---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for resnet_v2_50_inat21

A ResNet v2 image classification model. This model was trained on the `iNaturalist 2021` dataset - <https://github.com/visipedia/inat_comp/tree/master/2021>.
The model was trained using an adapted procedure from ResNet Strikes Back (RSB) A2.

Note: A 256 x 256 variant of this model is available as `resnet_v2_50_inat21-256px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 44.0
    - Input image size: 384 x 384
- **Dataset:** iNaturalist 2021 (10000 classes)

- **Papers:**
    - Identity Mappings in Deep Residual Networks: <https://arxiv.org/abs/1603.05027>
    - ResNet strikes back: An improved training procedure in timm: <https://arxiv.org/abs/2110.00476>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("resnet_v2_50_inat21", inference=True)
# Note: A 256x256 variant is available as "resnet_v2_50_inat21-256px"

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("resnet_v2_50_inat21", inference=True)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
out, _ = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 10000), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("resnet_v2_50_inat21", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("resnet_v2_50_inat21", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 2048)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("resnet_v2_50_inat21", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 256, 96, 96])),
#  ('stage2', torch.Size([1, 512, 48, 48])),
#  ('stage3', torch.Size([1, 1024, 24, 24])),
#  ('stage4', torch.Size([1, 2048, 12, 12]))]
```

## Citation

```bibtex
@misc{he2016identitymappingsdeepresidual,
      title={Identity Mappings in Deep Residual Networks},
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2016},
      eprint={1603.05027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1603.05027},
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
