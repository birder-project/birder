---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for maxvit_s_il-all

A MaxViT image classification model. This model was trained on the `il-all` dataset, encompassing all relevant bird species found in Israel, including rarities.

The species list is derived from data available at <https://www.israbirding.com/checklist/>.

Note: A 256 x 256 variant of this model is available as `maxvit_s_il-all256px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 31.1
    - Input image size: 384 x 384
- **Dataset:** il-all (550 classes)

- **Papers:**
    - MaxViT: Multi-Axis Vision Transformer: <https://arxiv.org/abs/2204.01697>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("maxvit_s_il-all", inference=True)
# Note: A 256x256 variant is available as "maxvit_s_il-all256px"

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("maxvit_s_il-all", inference=True)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
out, _ = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 550), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("maxvit_s_il-all", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("maxvit_s_il-all", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 512)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("maxvit_s_il-all", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 96, 96, 96])),
#  ('stage2', torch.Size([1, 128, 48, 48])),
#  ('stage3', torch.Size([1, 256, 24, 24])),
#  ('stage4', torch.Size([1, 512, 12, 12]))]
```

## Citation

```bibtex
@misc{tu2022maxvitmultiaxisvisiontransformer,
      title={MaxViT: Multi-Axis Vision Transformer},
      author={Zhengzhong Tu and Hossein Talebi and Han Zhang and Feng Yang and Peyman Milanfar and Alan Bovik and Yinxiao Li},
      year={2022},
      eprint={2204.01697},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.01697},
}
```
