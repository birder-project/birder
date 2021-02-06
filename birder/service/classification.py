"""
ClassificationModelHandler defines a base model for classification service.

Based on template from:
https://github.com/awslabs/multi-model-server/tree/master/examples/model_service_template
"""

import json
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import mxnet as mx
import numpy as np

try:
    # MMS Runtime
    from base import ModelHandler
    from base import check_input_shape
    from preprocess import preprocess_image

except ImportError:
    from birder.common.preprocess import preprocess_image

    from .base import ModelHandler
    from .base import check_input_shape


def top_probability(data: mx.nd.NDArray, labels: List[str], top: int) -> List[Any]:
    """
    Get top probability prediction from NDArray
    """

    dim = len(data.shape)
    if dim > 2:
        data = mx.nd.array(np.squeeze(data.asnumpy(), axis=tuple(range(dim)[2:])))

    sorted_prob = mx.nd.argsort(data[0], is_ascend=False)
    top_prob = map(lambda x: int(x.asscalar()), sorted_prob[0:top])

    return [{"probability": float(data[0, i].asscalar()), "class": labels[i]} for i in top_prob]


class ClassificationModelHandler(ModelHandler):
    """
    ClassificationModelHandler defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-K labels are returned.
    Batching is not supported.
    """

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.mxnet_ctx: mx.Context = None
        self.mx_model: mx.module.Module = None
        self.labels: Optional[List[str]] = None
        self.signature: Any = None
        self.rgb_mean: Any = None
        self.rgb_std: Any = None
        self.rgb_scale: Any = None
        self.epoch = 0
        self.top_k = top_k

    # pylint: disable=too-many-locals
    def initialize(self, context) -> None:
        super().initialize(context)

        assert self._batch_size == 1, "Batch is not supported"

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")

        signature_file_path = os.path.join(model_dir, "signature.json")
        if os.path.isfile(signature_file_path) is False:
            raise RuntimeError("Missing signature.json file")

        with open(signature_file_path) as signature_file_handle:
            self.signature = json.load(signature_file_handle)

        self.rgb_mean = mx.nd.array(
            [self.signature["mean_r"], self.signature["mean_g"], self.signature["mean_b"]], dtype="float32"
        )
        self.rgb_std = mx.nd.array(
            [self.signature["std_r"], self.signature["std_g"], self.signature["std_b"]], dtype="float32"
        )
        self.rgb_scale = self.signature["scale"]

        model_files_prefix = context.manifest["model"]["modelName"]
        archive_synset = os.path.join(model_dir, "synset.txt")
        if os.path.isfile(archive_synset):
            synset = archive_synset
            with open(synset, "r") as synset_handle:
                self.labels = [line.strip() for line in synset_handle.readlines()]

        data_names = []
        data_shapes = []
        for input_data in self.signature["inputs"]:
            data_name = input_data["data_name"]
            data_shape = input_data["data_shape"]

            # Set batch size
            data_shape[0] = self._batch_size

            # Replace 0 entry in data shape with 1 for binding executor
            # pylint: disable=consider-using-enumerate
            for idx in range(len(data_shape)):
                if data_shape[idx] == 0:
                    data_shape[idx] = 1

            data_names.append(data_name)
            data_shapes.append((data_name, tuple(data_shape)))

        checkpoint_prefix = f"{model_dir}/{model_files_prefix}"

        # Load MXNet module
        if gpu_id is None:
            self.mxnet_ctx = mx.cpu()

        else:
            self.mxnet_ctx = mx.gpu(gpu_id)

        (sym, arg_params, aux_params) = mx.model.load_checkpoint(checkpoint_prefix, self.epoch)

        self.mx_model = mx.module.Module(
            symbol=sym, context=self.mxnet_ctx, data_names=data_names, label_names=None
        )
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    def preprocess(self, batch: List[Dict[str, Any]]) -> Optional[List[mx.nd.NDArray]]:
        """
        Decode all input images into mx.nd.NDArray and preprocess them.
        """

        img_list = []
        param_name = self.signature["inputs"][0]["data_name"]
        input_shape = self.signature["inputs"][0]["data_shape"]

        for data in batch:
            img = data.get(param_name)
            if img is None:
                img = data.get("body")

            if img is None:
                img = data.get("data")

            if img is None or len(img) == 0:
                self.error = "Empty image input"
                return None

            try:
                img_arr = mx.image.imdecode(img, to_rgb=True)

            # pylint: disable=broad-except
            except Exception:
                logging.exception("Corrupted image input")
                self.error = "Corrupted image input"
                return None

            # We are assuming input shape is NCHW
            (height, width) = input_shape[2:]
            img_arr = preprocess_image(img_arr, (height, width), self.rgb_mean, self.rgb_std, self.rgb_scale)

            img_list.append(img_arr)

        return img_list

    def inference(self, model_input: Optional[List[mx.nd.NDArray]]) -> Optional[List[mx.nd.NDArray]]:
        """
        Internal inference methods for MXNet. Run forward computation and
        return output.

        Parameters
        ----------
        model_input
            Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        """

        if self.error is not None:
            return None

        # Check input shape
        check_input_shape(model_input, self.signature)
        model_input = [item.as_in_context(self.mxnet_ctx) for item in model_input]
        self.mx_model.forward(mx.io.DataBatch(model_input))
        outputs = self.mx_model.get_outputs()

        # Bypass lazy evaluation get_outputs either returns a list of nd arrays a list of list of NDArray
        for output in outputs:
            if isinstance(output, list):
                for y in output:
                    if isinstance(y, mx.nd.NDArray):
                        y.wait_to_read()

            elif isinstance(output, mx.nd.NDArray):
                output.wait_to_read()

        return outputs

    def postprocess(self, inference_output: Optional[List[mx.nd.NDArray]]) -> List[Any]:
        if self.error is not None:
            return [self.error] * self._batch_size

        assert isinstance(self.labels, list)

        return [top_probability(d, self.labels, top=self.top_k) for d in inference_output]


_service = ClassificationModelHandler()


def handle(data, context):
    if _service.initialized is False:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
