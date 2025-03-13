---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for focalnet_b_lrf_intermediate-eu-common

A FocalNet image classification model. The model follows a two-stage training process: first undergoing intermediate training on a large-scale dataset containing diverse bird species from around the world, then fine-tuned specifically on the `eu-common` dataset (all the relevant bird species found in the Arabian peninsula inc. rarities).

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 88.4
    - Input image size: 384 x 384
- **Dataset:** eu-common (707 classes)
    - Intermediate training involved ~5500 species from asia, europe and eastern africa

- **Papers:**
    - Focal Modulation Networks: <https://arxiv.org/abs/2203.11926>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("focalnet_b_lrf_intermediate-eu-common", inference=True)

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

(net, model_info) = birder.load_pretrained_model("focalnet_b_lrf_intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1024)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("focalnet_b_lrf_intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 128, 96, 96])),
#  ('stage2', torch.Size([1, 256, 48, 48])),
#  ('stage3', torch.Size([1, 512, 24, 24])),
#  ('stage4', torch.Size([1, 1024, 12, 12]))]
```

## Citation

```bibtex
@misc{yang2022focalmodulationnetworks,
      title={Focal Modulation Networks},
      author={Jianwei Yang and Chunyuan Li and Xiyang Dai and Lu Yuan and Jianfeng Gao},
      year={2022},
      eprint={2203.11926},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2203.11926},
}
```
