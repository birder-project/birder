---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for resnet_v1_50_arabian-peninsula

A Resnet v1 image classification model. This model was trained on the `arabian-peninsula` dataset (all the relevant bird species found in the Arabian peninsula inc. rarities).
The training followed RSB procedure A2.

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 25.0
    - Input image size: 256 x 256
- **Dataset:** arabian-peninsula (735 classes)

- **Papers:**
    - Deep Residual Learning for Image Recognition: <https://arxiv.org/abs/1512.03385>
    - ResNet strikes back: An improved training procedure in timm: <https://arxiv.org/abs/2110.00476>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("resnet_v1_50_arabian-peninsula", inference=True)

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

(net, model_info) = birder.load_pretrained_model("resnet_v1_50_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 2048)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("resnet_v1_50_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 256, 64, 64])),
#  ('stage2', torch.Size([1, 512, 32, 32])),
#  ('stage3', torch.Size([1, 1024, 16, 16])),
#  ('stage4', torch.Size([1, 2048, 8, 8]))]
```

## Citation

```bibtex
@misc{he2015deepresiduallearningimage,
      title={Deep Residual Learning for Image Recognition},
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1512.03385},
}

@misc{wightman2021resnetstrikesbackimproved,
      title={ResNet strikes back: An improved training procedure in timm},
      author={Ross Wightman and Hugo Touvron and Hervé Jégou},
      year={2021},
      eprint={2110.00476},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2110.00476},
}
```
