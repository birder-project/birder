"""
ModelHandler defines a base model handler for MMS service.
"""

import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import mxnet as mx


def check_input_shape(inputs: List[mx.nd.NDArray], signature: Dict[str, Any]):
    """
    Check input data shape consistency with signature
    """

    assert isinstance(inputs, list), "Input data must be a list"

    msg = (
        f"Input number mismatches with signature (expected {len(signature['inputs'])} but got {len(inputs)})"
    )
    assert len(inputs) == len(signature["inputs"]), msg

    for input_data, sig_input in zip(inputs, signature["inputs"]):
        assert isinstance(input_data, mx.nd.NDArray), "Each input must be NDArray"
        assert len(input_data.shape) == len(sig_input["data_shape"]), (
            f"Shape dimension of input {sig_input['data_name']} mismatches with "
            f"signature. {len(sig_input['data_shape'])} expected but got {len(input_data.shape)}"
        )
        for idx in range(len(input_data.shape)):
            if idx != 0 and sig_input["data_shape"][idx] != 0:
                assert sig_input["data_shape"][idx] == input_data.shape[idx], (
                    f"Input {sig_input['data_name']} has different shape with "
                    f"signature. {sig_input['data_shape']} expected but got {input_data.shape}"
                )


class ModelHandler:
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context) -> None:
        """
        Initialize model. This will be called during model loading time
        """

        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True

    def preprocess(self, batch: List[Dict[str, Any]]) -> Optional[List[mx.nd.NDArray]]:
        """
        Transform raw input into model input data.

        Parameters
        ----------
        batch
            List of raw requests, should match batch size.

        Returns
        -------
        list
            Preprocessed model input data.
        """

        raise NotImplementedError

    def inference(self, model_input: Optional[List[mx.nd.NDArray]]) -> Optional[List[mx.nd.NDArray]]:
        """
        Internal inference methods
        """

        raise NotImplementedError

    def postprocess(self, inference_output: Optional[List[mx.nd.NDArray]]) -> List[Any]:
        """
        Returns predict result in batch.

        Parameters
        ----------
        inference_output
            List of inference output.

        Returns
        -------
        list
            Predict results.
        """

        raise NotImplementedError

    def handle(self, data: List[Dict[str, Any]], context) -> List[Any]:
        """
        Custom service entry point function.

        Parameters
        ----------
        data
            List of objects, raw input from request.
        context
            MMS context object.

        Returns
        -------
        list
            List of outputs to be send back to client.
        """

        # Reset earlier errors
        self.error = None

        try:
            preprocess_start = time.time()
            _data = self.preprocess(data)
            inference_start = time.time()
            _data = self.inference(_data)
            postprocess_start = time.time()
            _data = self.postprocess(_data)
            end_time = time.time()

            metrics = context.metrics
            metrics.add_time("PreprocessTime", round((inference_start - preprocess_start) * 1000, 2))
            metrics.add_time("InferenceTime", round((postprocess_start - inference_start) * 1000, 2))
            metrics.add_time("PostprocessTime", round((end_time - postprocess_start) * 1000, 2))

            return _data

        # pylint: disable=broad-except
        except Exception as e:
            logging.exception("Exception at handle")
            request_processor = context.request_processor
            request_processor.report_status(500, "Unknown inference error")

            # Expose some insight of the error to the API client
            return [str(e)] * self._batch_size
