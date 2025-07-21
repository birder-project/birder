---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for hornet_tiny_7x7_danube-delta

A HorNet image classification model. This model was trained on the `danube-delta` dataset (all the relevant bird species found int the Danube Delta region).

The species list is derived from data available at <https://www.discoverdanubedelta.com/wp-content/uploads/2023/01/BirdsList-ian-2023.pdf>.

Note: this is a subset of the `eu-common` dataset.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 22.1
    - Input image size: 256 x 256
- **Dataset:** danube-delta (368 classes)

- **Papers:**
    - HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions: <https://arxiv.org/abs/2207.14284>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("hornet_tiny_7x7_danube-delta", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 368), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("hornet_tiny_7x7_danube-delta", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 512)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("hornet_tiny_7x7_danube-delta", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 64, 64, 64])),
#  ('stage2', torch.Size([1, 128, 32, 32])),
#  ('stage3', torch.Size([1, 256, 16, 16])),
#  ('stage4', torch.Size([1, 512, 8, 8]))]
```

## Citation

```bibtex
@misc{rao2022hornetefficienthighorderspatial,
      title={HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions},
      author={Yongming Rao and Wenliang Zhao and Yansong Tang and Jie Zhou and Ser-Nam Lim and Jiwen Lu},
      year={2022},
      eprint={2207.14284},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2207.14284},
}
```
