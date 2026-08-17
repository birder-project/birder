---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: mit
base_model:
- microsoft/Florence-2-base
---

# Model Card for davit_fl_base_florence-v2

A DaViT FL Base image encoder from Microsoft's Florence-2 model, converted to the Birder format for image feature extraction.
This version preserves the original vision backbone weights and architecture for downstream tasks.

See <https://huggingface.co/microsoft/Florence-2-base> for further details.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 90.4
    - Input image size: 768 x 768

- **Papers:**
    - DaViT: Dual Attention Vision Transformers: <https://arxiv.org/abs/2204.03645>
    - Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks: <https://arxiv.org/abs/2311.06242>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("davit_fl_base_florence-v2", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("davit_fl_base_florence-v2", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1024)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("davit_fl_base_florence-v2", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 128, 192, 192])),
#  ('stage2', torch.Size([1, 256, 96, 96])),
#  ('stage3', torch.Size([1, 512, 48, 48])),
#  ('stage4', torch.Size([1, 1024, 24, 24]))]
```

## Citation

```bibtex
@misc{ding2022davitdualattentionvision,
      title={DaViT: Dual Attention Vision Transformers},
      author={Mingyu Ding and Bin Xiao and Noel Codella and Ping Luo and Jingdong Wang and Lu Yuan},
      year={2022},
      eprint={2204.03645},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.03645}
}

@misc{xiao2023florence2advancingunifiedrepresentation,
      title={Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
      author={Bin Xiao and Haiping Wu and Weijian Xu and Xiyang Dai and Houdong Hu and Yumao Lu and Michael Zeng and Ce Liu and Lu Yuan},
      year={2023},
      eprint={2311.06242},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.06242},
}
```
