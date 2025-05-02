---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for vit_l16_mim

A ViT l16 image encoder pre-trained using Masked Image Modeling (MIM). This model has *not* been fine-tuned for a specific classification task and is intended to be used as a general-purpose feature extractor or a backbone for downstream tasks like object detection, segmentation, or custom classification.

## Model Details

- **Model Type:** Image encoder
- **Model Stats:**
    - Params (M): 303.3
    - Input image size: 224 x 224
- **Dataset:** Trained on a diverse dataset of approximately 11M images, including:
    - iNaturalist 2021 (~3.3M)
    - WebVision-2.0 (~1.5M random subset)
    - imagenet-w21-webp-wds (~1M random subset)
    - SA-1B (~220K random subset of 20 chunks)
    - COCO (~120K)
    - NABirds (~48K)
    - Birdsnap v1.1 (~44K)
    - CUB-200 2011 (~11K)
    - The Birder dataset (~5M, private dataset)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Masked Autoencoders Are Scalable Vision Learners: <https://arxiv.org/abs/2111.06377>

## Model Usage

### Image Embeddings

```python
import torch
import birder
from PIL import Image

(net, model_info) = birder.load_pretrained_model("vit_l16_mim_400", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
input_tensor = transform(image).unsqueeze(dim=0)
with torch.inference_mode():
    embedding = net.embedding(input_tensor)
    # embedding is a tensor with shape of (1, 1024)
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

@misc{he2021maskedautoencodersscalablevision,
      title={Masked Autoencoders Are Scalable Vision Learners},
      author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Dollár and Ross Girshick},
      year={2021},
      eprint={2111.06377},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2111.06377},
}
```
