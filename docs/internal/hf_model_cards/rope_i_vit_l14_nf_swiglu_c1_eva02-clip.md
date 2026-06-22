---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- QuanSun/EVA-CLIP
---

# Model Card for rope_i_vit_l14_nf_swiglu_c1_eva02-clip

A RoPE ViT-L14 image encoder from the EVA02 CLIP model by Sun et al., converted to the Birder format for image feature extraction.
This version retains the original model weights and architecture.
It is a general-purpose visual backbone.

See: <https://huggingface.co/QuanSun/EVA-CLIP> for further details.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 304.5
    - Input image size: 336 x 336

- **Papers:**
    - EVA-02: A Visual Representation for Neon Genesis: <https://arxiv.org/abs/2303.11331>
    - EVA-CLIP: Improved Training Techniques for CLIP at Scale: <https://arxiv.org/abs/2303.15389>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("rope_i_vit_l14_nf_swiglu_c1_eva02-clip", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("rope_i_vit_l14_nf_swiglu_c1_eva02-clip", inference=True)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
out, _ = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 768), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("rope_i_vit_l14_nf_swiglu_c1_eva02-clip", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("rope_i_vit_l14_nf_swiglu_c1_eva02-clip", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1024)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("rope_i_vit_l14_nf_swiglu_c1_eva02-clip", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 1024, 24, 24]))]
```

## Citation

```bibtex
@article{Fang_2024,
   title={EVA-02: A visual representation for neon genesis},
   volume={149},
   ISSN={0262-8856},
   url={http://dx.doi.org/10.1016/j.imavis.2024.105171},
   DOI={10.1016/j.imavis.2024.105171},
   journal={Image and Vision Computing},
   publisher={Elsevier BV},
   author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
   year={2024},
   month=Sept, pages={105171}
}
@misc{sun2023evaclipimprovedtrainingtechniques,
      title={EVA-CLIP: Improved Training Techniques for CLIP at Scale},
      author={Quan Sun and Yuxin Fang and Ledell Wu and Xinlong Wang and Yue Cao},
      year={2023},
      eprint={2303.15389},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2303.15389},
}
```
