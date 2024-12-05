# pylint: disable=wrong-import-position
dependencies = ["torch"]

from functools import partial as _partial  # noqa: E402

import torch  # noqa: E402

import birder  # noqa: E402

_model_dict = {}
for model in birder.list_pretrained_models():
    dst = torch.hub.get_dir() + f"/{model}.pt"
    _model_dict[model.replace("-", "_")] = _partial(birder.load_pretrained_model, model, dst=dst)

globals().update(_model_dict)
