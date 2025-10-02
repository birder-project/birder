---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for regnet_z_4g_eu-common

A RegNet Z 4g image classification model. This model was trained on the `eu-common` dataset containing common European bird species.

The species list is derived from the Collins bird guide [^1].

[^1]: Svensson, L., Mullarney, K., & Zetterström, D. (2022). Collins bird guide (3rd ed.). London, England: William Collins.

Note: A 256 x 256 variant of this model is available as `regnet_z_4g_eu-common256px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 28.5
    - Input image size: 384 x 384
- **Dataset:** eu-common (707 classes)

- **Papers:**
    - Fast and Accurate Model Scaling: <https://arxiv.org/abs/2103.06877>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("regnet_z_4g_eu-common", inference=True)
# Note: A 256x256 variant is available as "regnet_z_4g_eu-common256px"

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 707), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("regnet_z_4g_eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1536)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("regnet_z_4g_eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 48, 96, 96])),
#  ('stage2', torch.Size([1, 104, 48, 48])),
#  ('stage3', torch.Size([1, 240, 24, 24])),
#  ('stage4', torch.Size([1, 528, 12, 12]))]
```

## Citation

```bibtex
@misc{dollár2021fastaccuratemodelscaling,
      title={Fast and Accurate Model Scaling},
      author={Piotr Dollár and Mannat Singh and Ross Girshick},
      year={2021},
      eprint={2103.06877},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2103.06877},
}
```
