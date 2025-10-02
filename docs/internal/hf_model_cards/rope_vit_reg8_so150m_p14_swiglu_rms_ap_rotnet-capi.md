---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- birder-project/rope_vit_reg8_so150m_p14_swiglu_rms_avg_capi
---

# Model Card for rope_vit_reg8_so150m_p14_swiglu_rms_ap_rotnet-capi

A RoPE SoViT 150m p14 image orientation estimation model. The model follows a two-stage training process: first, CAPI pretraining, then trained to estimate image orientation.

Given an input image, the model predicts whether it is correctly oriented (0°), or rotated by 90°, 180°, or 270°.  
It is primarily intended for use in image curation pipelines, enabling automatic correction of mis-rotated images.

## Model Details

- **Model Type:** Image orientation estimation (4-way classification: 0°, 90°, 180°, 270°)
- **Model Stats:**
    - Params (M): 178.5
    - Input image size: 252 x 252
- **Dataset:** Mixture of public datasets
    - iNaturalist 2021
    - ImageNet-1K
    - Places365
    - Country211
    - ADE20K 2016
    - VOC2012
    - IndoorCVPR 2009

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Rotary Position Embedding for Vision Transformer: <https://arxiv.org/abs/2403.13298>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design: <https://arxiv.org/abs/2305.13035>
    - Cluster and Predict Latent Patches for Improved Masked Image Modeling: <https://arxiv.org/abs/2502.08769>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rope_vit_reg8_so150m_p14_swiglu_rms_ap_rotnet-capi", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 4), representing class probabilities.
#   index 0 -> 0° (upright)
#   index 1 -> 90° rotation
#   index 2 -> 180° rotation
#   index 3 -> 270° rotation
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rope_vit_reg8_so150m_p14_swiglu_rms_ap_rotnet-capi", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 896)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("rope_vit_reg8_so150m_p14_swiglu_rms_ap_rotnet-capi", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 896, 18, 18]))]
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

@misc{darcet2024visiontransformersneedregisters,
      title={Vision Transformers Need Registers},
      author={Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2024},
      eprint={2309.16588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.16588},
}

@misc{alabdulmohsin2024gettingvitshapescaling,
      title={Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design},
      author={Ibrahim Alabdulmohsin and Xiaohua Zhai and Alexander Kolesnikov and Lucas Beyer},
      year={2024},
      eprint={2305.13035},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2305.13035},
}

@misc{darcet2025clusterpredictlatentpatches,
      title={Cluster and Predict Latent Patches for Improved Masked Image Modeling},
      author={Timothée Darcet and Federico Baldassarre and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2025},
      eprint={2502.08769},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.08769},
}
```
