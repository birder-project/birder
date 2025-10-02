---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for rope_vit_reg8_so150m_p14_swiglu_rms_avg_capi

A RoPE SoViT 150m p14 image encoder pre-trained using CAPI. This model has *not* been fine-tuned for a specific classification task and is intended to be used as a general-purpose feature extractor or a backbone for downstream tasks like object detection, segmentation, or custom classification.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 171.0
    - Input image size: 224 x 224
- **Dataset:** Trained on a diverse dataset of approximately 21M images, including:
    - iNaturalist 2021 (~2.7M)
    - imagenet-w21-webp-wds (~2.4M random subset)
    - Objects365-2020 (~1.8M)
    - WebVision-2.0 (~1.5M random subset)
    - imagenet-1k-webp (~1.3M)
    - BIOSCAN-1M (~1.1M)
    - GLDv2 (~820K random subset of 100 chunks)
    - VMMR (~285K)
    - SA-1B (~220K random subset of 20 chunks)
    - TreeOfLife-10M (~200K random subset)
    - Places365 (~170K random subset)
    - Open Images V4 (~160K random subset)
    - COCO (~120K) x2
    - SUN397 (~100K)
    - Food-101 (~100K)
    - IP102 v1.1 (~75K)
    - NABirds (~48K)
    - Birdsnap v1.1 (~44K)
    - Country211 (~31K)
    - ADE20K 2016 (~22K) x2
    - VOC2012 (~17K)
    - IndoorCVPR 2009 (~15K)
    - VisDrone2019-DET (~10K) x4
    - BLIP3o-Pretrain-Short-Caption (~10K random subset)
    - comma10k (~10K) x2
    - FGVC-Aircraft 2013 (~10K) x2
    - Oxford-IIIT Pet (~7K) x4
    - CUB-200 2011 (~6K)
    - Flowers102 (~1K) x16
    - The Birder dataset (~8M, private dataset)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Rotary Position Embedding for Vision Transformer: <https://arxiv.org/abs/2403.13298>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design: <https://arxiv.org/abs/2305.13035>
    - Cluster and Predict Latent Patches for Improved Masked Image Modeling: <https://arxiv.org/abs/2502.08769>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rope_vit_reg8_so150m_p14_swiglu_rms_avg_capi", inference=True)

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

(net, model_info) = birder.load_pretrained_model("rope_vit_reg8_so150m_p14_swiglu_rms_avg_capi", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 896, 16, 16]))]
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
