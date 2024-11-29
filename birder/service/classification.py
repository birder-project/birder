import base64
import io
import json
import logging
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


# Must be in sync with birder.transforms.classification.inference_preset
def inference_preset(
    size: tuple[int, int], rgv_values: dict[str, list[float]], center_crop: float = 1.0
) -> Callable[..., torch.Tensor]:
    mean = rgv_values["mean"]
    std = rgv_values["std"]

    base_size = (int(size[0] / center_crop), int(size[1] / center_crop))
    return v2.Compose(  # type: ignore
        [
            v2.Resize(base_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(size),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


class ImageClassifier(BaseHandler):
    def __init__(self) -> None:
        super().__init__()
        self.model_yaml_config: dict[str, Any] = {}
        self.transforms: Callable[..., torch.Tensor]
        self.top_k = 3

    def initialize(self, context: Context) -> None:
        if context is not None and hasattr(context, "model_yaml_config") is True:
            self.model_yaml_config = context.model_yaml_config

        properties = context.system_properties
        if torch.cuda.is_available() is True and properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(self.map_location + ":" + str(properties.get("gpu_id")))
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        if properties.get("limit_max_image_pixels") is False:
            Image.MAX_IMAGE_PIXELS = None

        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)
        if os.path.isfile(model_path) is False:
            raise RuntimeError("Missing the model file")

        (self.model, self.mapping, self.transforms) = self._load_model(model_path, self.device, self.model_yaml_config)

        self.initialized = True

    def _load_model(
        self, path: str, device: torch.device, model_yaml_config: dict[str, Any]
    ) -> tuple[torch.ScriptModule | torch.nn.Module, dict[int, str], Callable[..., torch.Tensor]]:
        extra_files = {"task": "", "class_to_idx": "", "signature": "", "rgb_stats": ""}
        if path.endswith("pts") is True:
            model = torch.jit.load(path, map_location=device, _extra_files=extra_files)
            model.eval()
        elif path.endswith("pt2") is True:
            model = torch.export.load(path, extra_files=extra_files).module()
            model.to(device)
            # model.eval()  # Use when GraphModule add support for 'eval'
        else:
            raise RuntimeError(f"Unknown model type {path}")

        if extra_files["task"] != "image_classification":
            raise RuntimeError(f"Model type mismatch, task={extra_files['task']}")

        for param in model.parameters():
            param.requires_grad = False

        class_to_idx: dict[str, int] = json.loads(extra_files["class_to_idx"])
        signature = json.loads(extra_files["signature"])
        rgb_stats = json.loads(extra_files["rgb_stats"])

        size = signature["inputs"][0]["data_shape"][2:4]
        transforms = inference_preset(size, rgb_stats, 1.0)

        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))

        if model_yaml_config.get("compile", False) is True:
            logger.error("Compiling model")
            model = torch.compile(model)

        return (model, idx_to_class, transforms)

    def preprocess(self, data: list[Any]) -> torch.Tensor:
        images = []
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))

            else:
                # If the image is a list
                image = torch.FloatTensor(image)

            images.append(self.transforms(image))

        return torch.stack(images).to(self.device)

    def inference(self, data: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        with torch.inference_mode():
            results = self.model(data)

        return results

    def postprocess(self, data: torch.Tensor) -> list[dict[str, float]]:
        ps = F.softmax(data, dim=1)
        (probs, classes) = torch.topk(ps, self.top_k, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()

        results = []
        for s_probs, s_classes in zip(probs, classes):
            result: dict[str, float] = {}
            for prob, c in zip(s_probs, s_classes):
                class_name = self.mapping[c]
                result[class_name] = prob

            results.append(result)

        return results
