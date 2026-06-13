---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- facebook/sam-vit-base
---

# Model Card for vit_sam_b16_sam-sa1b

A ViT SAM b16 image encoder from the SAM project, converted to the Birder format for image feature extraction.
This version preserves the original model weights and architecture for downstream tasks.

See <https://huggingface.co/facebook/sam-vit-base> for further details.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 89.7
    - Input image size: 1024 x 1024

- **Papers:**
    - Segment Anything: <https://arxiv.org/abs/2304.02643>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("vit_sam_b16_sam-sa1b", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("vit_sam_b16_sam-sa1b", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 256)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("vit_sam_b16_sam-sa1b", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 256, 64, 64]))]
```

## Citation

```bibtex
@misc{kirillov2023segment,
      title={Segment Anything},
      author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
      year={2023},
      eprint={2304.02643},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.02643},
}
```
