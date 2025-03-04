---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for convnext_v2_tiny_intermediate-il-common

A ConvNext v2 image classification model. The model follows a two-stage training process: first undergoing intermediate training on a large-scale dataset containing diverse bird species from around the world, then fine-tuned specifically on the `il-common` dataset containing common bird species found in Israel.

The species list is derived from data available at <https://www.israbirding.com/checklist/>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 28.2
    - Input image size: 256 x 256
- **Dataset:** il-common (371 classes)
    - Intermediate training involved ~4000 species from asia, europe and eastern africa

- **Papers:**
    - ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders: <https://arxiv.org/abs/2301.00808>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("convnext_v2_tiny_intermediate-il-common", inference=True)

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

(net, model_info) = birder.load_pretrained_model("convnext_v2_tiny_intermediate-il-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 768)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("convnext_v2_tiny_intermediate-il-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 96, 64, 64])),
#  ('stage2', torch.Size([1, 192, 32, 32])),
#  ('stage3', torch.Size([1, 384, 16, 16])),
#  ('stage4', torch.Size([1, 768, 8, 8]))]
```

## Citation

```bibtex
@misc{woo2023convnextv2codesigningscaling,
      title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
      author={Sanghyun Woo and Shoubhik Debnath and Ronghang Hu and Xinlei Chen and Zhuang Liu and In So Kweon and Saining Xie},
      year={2023},
      eprint={2301.00808},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2301.00808},
}
```
