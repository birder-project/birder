---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for mvit_v2_t_il-all

A MViTv2 image classification model. This model was trained on the `il-all` dataset, encompassing all relevant bird species found in Israel, including rarities.

The species list is derived from data available at <https://www.israbirding.com/checklist/>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 23.9
    - Input image size: 384 x 384
- **Dataset:** il-all (550 classes)

- **Papers:**
    - MViTv2: Improved Multiscale Vision Transformers for Classification and Detection: <https://arxiv.org/abs/2112.01526>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("mvit_v2_t_il-all", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 550), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("mvit_v2_t_il-all", inference=True)

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

(net, model_info) = birder.load_pretrained_model("mvit_v2_t_il-all", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 96, 96, 96])),
#  ('stage2', torch.Size([1, 192, 48, 48])),
#  ('stage3', torch.Size([1, 384, 24, 24])),
#  ('stage4', torch.Size([1, 768, 12, 12]))]
```

## Citation

```bibtex
@misc{li2022mvitv2improvedmultiscalevision,
      title={MViTv2: Improved Multiscale Vision Transformers for Classification and Detection},
      author={Yanghao Li and Chao-Yuan Wu and Haoqi Fan and Karttikeya Mangalam and Bo Xiong and Jitendra Malik and Christoph Feichtenhofer},
      year={2022},
      eprint={2112.01526},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2112.01526},
}
```
