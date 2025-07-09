---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for efficientvim_m1_il-common

A EfficientViM image classification model. This model was trained on the `il-common` dataset, which contains common bird species found in Israel.

The species list is derived from data available at <https://www.israbirding.com/checklist/>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 6.1
    - Input image size: 256 x 256
- **Dataset:** il-common (371 classes)

- **Papers:**
    - EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality: <https://arxiv.org/abs/2411.15241>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("efficientvim_m1_il-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 371), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("efficientvim_m1_il-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 320)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("efficientvim_m1_il-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 128, 16, 16])),
#  ('stage2', torch.Size([1, 192, 8, 8])),
#  ('stage3', torch.Size([1, 320, 4, 4]))]
```

## Citation

```bibtex
@misc{lee2025efficientvimefficientvisionmamba,
      title={EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality},
      author={Sanghyeok Lee and Joonmyung Choi and Hyunwoo J. Kim},
      year={2025},
      eprint={2411.15241},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.15241},
}
```
