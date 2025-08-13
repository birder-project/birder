"""
Inference-optimized multi-GPU parallelization

This module provides InferenceDataParallel, an inference-specific alternative to
torch.nn.DataParallel.
"""

import copy
import logging
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class InferenceDataParallel(nn.Module):
    """
    Distributes inference batches across multiple GPUs for optimal throughput.

    This implementation is inspired by torch.nn.DataParallel but optimized specifically
    for inference workloads. We've removed training-specific overhead and added
    inference-focused optimizations.

    Key differences from torch.nn.DataParallel:
    - Model replication happens only once at initialization (not every forward pass)
    - No gradient synchronization or backward pass support (inference only)
    - Optimized scatter/gather operations without gradient tracking
    - Native support for custom model methods (e.g., embedding, classify)
    - Reduced memory footprint and improved throughput

    Important
    ---------
    This class assumes the model is already configured for inference mode
    (i.e., loaded with inference=True in load_model or manually set to eval mode
    with requires_grad=False on all parameters).

    Examples
    --------
    >>> # Model should be loaded with inference=True
    >>> model, info = load_model(device, network, inference=True)
    >>> parallel_model = InferenceDataParallel(model, device_ids=[0, 1, 2])
    >>> output = parallel_model(input_batch)

    >>> # Also works with custom methods:
    >>> embeddings = parallel_model.embedding(input_batch)
    >>> logits = parallel_model.classify(embeddings)

    >>> # Compile 'forward' and 'embedding' method on each replica, output to CPU
    >>> parallel_model = InferenceDataParallel(model, device_ids=[0, 1], output_device="cpu",
    ...                                        compile_replicas=True, compile_methods=["embedding"])
    >>> embeddings = parallel_model.embedding(input_batch)
    """

    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[list[int]] = None,
        output_device: Optional[int | str | torch.device] = None,
        compile_replicas: bool = False,
        compile_methods: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
            logger.debug(f"Using all available GPUs: {device_ids}")

        if len(device_ids) == 0:
            raise ValueError("At least one device id must be provided")

        self.device_ids = device_ids
        self.src_device = torch.device(f"cuda:{self.device_ids[0]}")
        if output_device is None:
            self.output_device = self.src_device
        elif isinstance(output_device, int):
            self.output_device = torch.device(f"cuda:{output_device}")
        elif isinstance(output_device, str):
            self.output_device = torch.device(output_device)
        elif isinstance(output_device, torch.device):
            self.output_device = output_device
        else:
            raise TypeError(
                f"Invalid type for output_device: {type(output_device)}. "
                "Expected int (CUDA device ID), str ('cpu', 'cuda:X'), or torch.device."
            )

        self.compile_replicas = compile_replicas
        compile_methods = compile_methods if compile_methods is not None else []

        # Move main module to first device
        self.module = module.to(self.src_device)

        # Replicate model to other devices
        if len(self.device_ids) > 1:
            logger.info(f"Replicating model to devices: {self.device_ids}")
            self._replicate_model()
        else:
            self.replicas = [self.module]

        # Compile each replica individually after they are all on their respective devices
        if compile_replicas is True:
            for i, replica in enumerate(self.replicas):
                self.replicas[i] = torch.compile(replica)
                logger.debug(f"Replica on cuda:{self.device_ids[i]} compiled")

                for method_name in compile_methods:
                    if hasattr(replica, method_name) is True and callable(getattr(replica, method_name)) is True:
                        original_method = getattr(replica, method_name)
                        compiled_method = torch.compile(original_method)
                        setattr(replica, method_name, compiled_method)
                        logger.debug(f"Method '{method_name}' on replica on cuda:{self.device_ids[i]} compiled")
                    else:
                        logger.warning(f"Cannot compile method '{method_name}': not found or not callable")

    def _replicate_model(self) -> None:
        self.replicas = [self.module]

        for device_id in self.device_ids[1:]:
            replica = copy.deepcopy(self.module)
            replica = replica.to(f"cuda:{device_id}")
            self.replicas.append(replica)
            logger.debug(f"Model replicated to cuda:{device_id}")

    def _scatter(
        self, inputs: torch.Tensor, kwargs: dict[str, Any]
    ) -> list[tuple[Optional[torch.Tensor], dict[str, Any]]]:
        # Calculate chunk sizes for even distribution
        batch_size = inputs.size(0)
        num_devices = len(self.device_ids)

        base_chunk_size = batch_size // num_devices
        remainder = batch_size % num_devices

        # Calculate chunk sizes - distribute remainder to first devices
        chunk_sizes = [base_chunk_size + (1 if i < remainder else 0) for i in range(num_devices)]

        # Split inputs
        input_chunks = []
        kwargs_chunks = []
        offset = 0
        for chunk_size, device_id in zip(chunk_sizes, self.device_ids):
            if chunk_size > 0:
                # Slice and move input to target device
                input_chunk = inputs[offset : offset + chunk_size].to(f"cuda:{device_id}", non_blocking=True)
                input_chunks.append(input_chunk)

                # Handle kwargs - we need to scatter any tensor kwargs too
                kwargs_chunk = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        # Tensors with the same batch size as the main input are scattered.
                        # Other tensors (e.g., lookup tables, global tensors) are moved
                        # to the device without slicing, assuming they are replicated.
                        if value.size(0) == batch_size:
                            kwargs_chunk[key] = value[offset : offset + chunk_size].to(
                                f"cuda:{device_id}", non_blocking=True
                            )
                        else:
                            kwargs_chunk[key] = value.to(f"cuda:{device_id}", non_blocking=True)

                    else:
                        # Non-tensor kwargs are passed as-is
                        kwargs_chunk[key] = value

                kwargs_chunks.append(kwargs_chunk)
                offset += chunk_size

            else:
                # Empty chunk for this device
                input_chunks.append(None)
                kwargs_chunks.append({})

        return list(zip(input_chunks, kwargs_chunks))

    def _gather(self, outputs: list[Optional[torch.Tensor]]) -> torch.Tensor:
        # Filter out None outputs (from devices with no data)
        valid_outputs = [out for out in outputs if out is not None and out.size(0) > 0]

        if len(valid_outputs) == 0:
            raise ValueError("No valid outputs to gather")

        if len(valid_outputs) == 1:
            output = valid_outputs[0]
            return output.to(self.output_device)

        # Move all outputs to output device and concatenate
        non_blocking = self.output_device.type == "cuda"
        gathered = []
        for output in valid_outputs:
            gathered.append(output.to(self.output_device, non_blocking=non_blocking))

        if self.output_device.type == "cuda":
            torch.cuda.synchronize(self.output_device)

        return torch.concat(gathered, dim=0)

    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass distributed across GPUs

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch to process
        **kwargs
            Additional keyword arguments to pass to the underlying model's forward method

        Returns
        -------
        Gathered outputs on output_device
        """

        if len(self.device_ids) == 1:
            return self.module(inputs.to(self.src_device), **kwargs)

        scattered = self._scatter(inputs, kwargs)

        outputs = []
        for replica, (input_chunk, kwargs_chunk), device_id in zip(self.replicas, scattered, self.device_ids):
            if input_chunk is not None and input_chunk.size(0) > 0:
                with torch.cuda.device(device_id):
                    output = replica(input_chunk, **kwargs_chunk)
                    outputs.append(output)
            else:
                outputs.append(None)

        return self._gather(outputs)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the wrapped module
        """

        try:
            return super().__getattr__(name)
        except AttributeError:
            # Check if this is a method/attribute of the wrapped module
            if hasattr(self.module, name) is True:
                attr = getattr(self.module, name)
                if callable(attr) is True:
                    return self._make_parallel_method(name)

                return attr

            raise

    def _make_parallel_method(self, method_name: str) -> Callable[..., torch.Tensor]:
        """
        Creates a parallelized wrapper for a specified method of the wrapped module

        This allows custom methods (e.g., model.embedding()) to be called
        on the InferenceDataParallel instance, which then scatters inputs,
        calls the method on each replica, and gathers the results.

        Parameters
        ----------
        method_name
            The name of the method to parallelize (e.g., 'embedding', 'classify').

        Returns
        -------
        A new callable that performs the specified method in parallel across the available devices.
        """

        def parallel_method(inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            if len(self.device_ids) == 1:
                method = getattr(self.module, method_name)
                return method(inputs.to(self.src_device), **kwargs)

            scattered = self._scatter(inputs, kwargs)

            outputs = []
            for replica, (input_chunk, kwargs_chunk), device_id in zip(self.replicas, scattered, self.device_ids):
                if input_chunk is not None and input_chunk.size(0) > 0:
                    with torch.cuda.device(device_id):
                        method = getattr(replica, method_name)
                        output = method(input_chunk, **kwargs_chunk)
                        outputs.append(output)
                else:
                    outputs.append(None)

            return self._gather(outputs)

        return parallel_method

    def __repr__(self) -> str:
        return (
            f"InferenceDataParallel(\n"
            f"  devices={self.device_ids},\n"
            f"  output_device={self.output_device},\n"
            f"  src_device={self.src_device}\n"
            f")"
        )
