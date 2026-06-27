---
tags:
- image-feature-extraction
- birder
- pytorch
- biology
library_name: birder
license: apache-2.0
---

# Model Card for rope_vit_s14_swiglu_avg_eva-bio

A RoPE ViT s14 image encoder pretrained using EVA-style Masked Image Modeling (MIM) distillation from a BioCLIP v2 ViT l14 teacher. This model has *not* been fine-tuned for a specific classification task and is intended to be used as a general-purpose feature extractor or a backbone for downstream tasks like object detection, segmentation, or custom classification.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 28.7
    - Input image size: 224 x 224
- **Dataset:** Trained on a diverse dataset of approximately 31M images, including:
    - TreeOfLife-10M-EOL-NaturalImages
    - iNaturalist 2021
    - BIOSCAN-5M (pretrain split)
    - TreeOfLife-200M (subset)
    - IP102 v1.1
    - iWildCam 2022 (subset)
    - The Birder dataset (private dataset)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Rotary Position Embedding for Vision Transformer: <https://arxiv.org/abs/2403.13298>
    - EVA-02: A Visual Representation for Neon Genesis: <https://arxiv.org/abs/2303.11331>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("rope_vit_s14_swiglu_avg_eva-bio", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("rope_vit_s14_swiglu_avg_eva-bio", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 384)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("rope_vit_s14_swiglu_avg_eva-bio", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 384, 16, 16]))]
```

## Citation

```bibtex
@misc{dosovitskiy2021imageworth16x16words,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2021},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2010.11929},
}

@misc{heo2024rotarypositionembeddingvision,
      title={Rotary Position Embedding for Vision Transformer},
      author={Byeongho Heo and Song Park and Dongyoon Han and Sangdoo Yun},
      year={2024},
      eprint={2403.13298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.13298},
}

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
```
