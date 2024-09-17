from birder.common.fs_ops import load_pretrained_model
from birder.common.lib import get_channels_from_signature
from birder.common.lib import get_size_from_signature
from birder.core.inference.classification import evaluate as evaluate_classification
from birder.core.transforms.classification import inference_preset as classification_transform
from birder.model_registry.model_registry import list_pretrained_models

__version__ = "v0.0.5a32"

__all__ = [
    "classification_transform",
    "evaluate_classification",
    "get_channels_from_signature",
    "get_size_from_signature",
    "list_pretrained_models",
    "load_pretrained_model",
]
