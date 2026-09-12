---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for naflex_vit_b16_nepa-generic

A NaFlex ViT B/16 image encoder pretrained using NEPA with variable patch sizes from 14 to 32 pixels and variable image resolutions from 192px to 320px, subject to a sequence-length budget of 36-400 tokens.
This model has *not* been fine-tuned for a specific classification task and is intended to be used as a general-purpose feature extractor or a backbone for downstream tasks like object detection, segmentation, or custom classification.

## Model Details

- **Model Type:** Image encoder and detection backbone
- **Model Stats:**
    - Params (M): 85.8
    - Input image size: 256 x 256 (natural aspect ratio)
- **Dataset:** Trained on a diverse dataset of approximately 20M images, including:
    - imagenet-w21-webp-wds (~13.1M)
    - Objects365-2020 (~1.8M)
    - GLDv2 (~3M subset)
    - Places365 (~1.8M)
    - COCO (~120K)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features: <https://arxiv.org/abs/2502.14786>
    - FlexiViT: One Model for All Patch Sizes: <https://arxiv.org/abs/2212.08013>
    - Next-Embedding Prediction Makes Strong Vision Learners: <https://arxiv.org/abs/2512.16922>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("naflex_vit_b16_nepa-generic", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create a NaFlex inference transform
patch_size = net.stem_stride
max_seq_len = (size[0] // patch_size) * (size[1] // patch_size)
transform = birder.naflex_transform(patch_size, max_seq_len, model_info.rgb_stats)

# Option 2: helper (quick start with NaFlex preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform(
    "naflex_vit_b16_nepa-generic",
    inference=True,
    naflex=True,
)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 768)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("naflex_vit_b16_nepa-generic", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 768, 16, 16]))]
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

@misc{tschannen2025siglip2multilingualvisionlanguage,
      title={SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features},
      author={Michael Tschannen and Alexey Gritsenko and Xiao Wang and Muhammad Ferjad Naeem and Ibrahim Alabdulmohsin and Nikhil Parthasarathy and Talfan Evans and Lucas Beyer and Ye Xia and Basil Mustafa and Olivier Hénaff and Jeremiah Harmsen and Andreas Steiner and Xiaohua Zhai},
      year={2025},
      eprint={2502.14786},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.14786},
}

@misc{beyer2023flexivitmodelpatchsizes,
      title={FlexiViT: One Model for All Patch Sizes},
      author={Lucas Beyer and Pavel Izmailov and Alexander Kolesnikov and Mathilde Caron and Simon Kornblith and Xiaohua Zhai and Matthias Minderer and Michael Tschannen and Ibrahim Alabdulmohsin and Filip Pavetic},
      year={2023},
      eprint={2212.08013},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2212.08013},
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
