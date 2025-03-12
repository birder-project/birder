---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for efficientnet_v2_s_arabian-peninsula

A Resnet v1 image classification model. This model was trained on the `arabian-peninsula` dataset (all the relevant bird species found in the Arabian peninsula inc. rarities).

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 21.1
    - Input image size: 256 x 256
- **Dataset:** arabian-peninsula (735 classes)

- **Papers:**
    - EfficientNetV2: Smaller Models and Faster Training: <https://arxiv.org/abs/2104.00298>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("efficientnet_v2_s_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 735), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("efficientnet_v2_s_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1280)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("efficientnet_v2_s_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 48, 64, 64])),
#  ('stage2', torch.Size([1, 64, 32, 32])),
#  ('stage3', torch.Size([1, 160, 16, 16])),
#  ('stage4', torch.Size([1, 256, 8, 8]))]
```

## Citation

```bibtex
@misc{tan2021efficientnetv2smallermodelsfaster,
      title={EfficientNetV2: Smaller Models and Faster Training},
      author={Mingxing Tan and Quoc V. Le},
      year={2021},
      eprint={2104.00298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2104.00298},
}
```
