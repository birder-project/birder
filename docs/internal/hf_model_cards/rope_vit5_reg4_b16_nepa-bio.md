---
tags:
- image-feature-extraction
- birder
- pytorch
- biology
library_name: birder
license: apache-2.0
---

# Model Card for rope_vit5_reg4_b16_nepa-bio

A RoPE ViT-5 b16 image encoder pretrained using NEPA. This model has *not* been fine-tuned for a specific classification task and is intended to be used as a general-purpose feature extractor or a backbone for downstream tasks like object detection, segmentation, or custom classification.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 85.8
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
    - ViT-5: Vision Transformers for The Mid-2020s: <https://arxiv.org/abs/2602.08071>
    - Next-Embedding Prediction Makes Strong Vision Learners: <https://arxiv.org/abs/2512.16922>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("rope_vit5_reg4_b16_nepa-bio", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("rope_vit5_reg4_b16_nepa-bio", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 768)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("rope_vit5_reg4_b16_nepa-bio", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 768, 14, 14]))]
```

## Citation

```bibtex
@misc{wang2026vit5visiontransformersmid2020s,
      title={ViT-5: Vision Transformers for The Mid-2020s},
      author={Feng Wang and Sucheng Ren and Tiezheng Zhang and Predrag Neskovic and Anand Bhattad and Cihang Xie and Alan Yuille},
      year={2026},
      eprint={2602.08071},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.08071},
}

@misc{xu2025nextembeddingpredictionmakesstrong,
      title={Next-Embedding Prediction Makes Strong Vision Learners},
      author={Sihan Xu and Ziqiao Ma and Wenhao Chai and Xuweiyi Chen and Weiyang Jin and Joyce Chai and Saining Xie and Stella X. Yu},
      year={2025},
      eprint={2512.16922},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.16922},
}
```
