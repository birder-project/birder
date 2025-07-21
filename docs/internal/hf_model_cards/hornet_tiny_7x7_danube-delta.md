---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for hornet_tiny_7x7_danube-delta

<SHORT_DESCRIPTION>

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 22.1
    - Input image size: 256 x 256
- **Dataset:** <DATASET_NAME> (368 classes)

- **Papers:**
    - <PAPER_TITLE>: <PAPER_LINK>

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
<BibTeX>
```
