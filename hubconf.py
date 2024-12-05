# pylint: disable=wrong-import-position
dependencies = ["torch"]

from functools import partial as _partial  # noqa: E402

import birder  # noqa: E402

_model_dict = {}
for model in birder.list_pretrained_models():
    _model_dict[model.replace("-", "_")] = _partial(birder.load_pretrained_model, model)

globals().update(_model_dict)
