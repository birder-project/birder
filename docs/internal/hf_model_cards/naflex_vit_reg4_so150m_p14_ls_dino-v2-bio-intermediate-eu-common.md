---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio
---

# Model Card for naflex_vit_reg4_so150m_p14_ls_dino-v2-bio-intermediate-eu-common

A NaFlex SoViT Reg4 150M/14 image classification model initialized from [Bio-DINO](https://huggingface.co/birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio).
The model follows a three-stage training process: first, DINOv2-style self-supervised pretraining on natural biological images, next intermediate training as a NaFlex ViT on a large-scale dataset containing diverse bird species from around the world, finally fine-tuned specifically on the `eu-common` dataset containing common European bird species.

NaFlex preprocessing preserves the natural aspect ratio of the input image within a patch-token budget.
NaFlex training used variable sequence lengths derived from reference resolutions ranging from 252px to 364px, corresponding to budgets of 324-676 image tokens with 14 x 14 patches.
The default inference reference size is 336 x 336, corresponding to a budget of 576 image tokens.

The species list is derived from the Collins bird guide [^1].

[^1]: Svensson, L., Mullarney, K., & Zetterström, D. (2022). Collins bird guide (3rd ed.). London, England: William Collins.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 134.5
    - Input image size: 336 x 336 (natural aspect ratio)
- **Dataset:** eu-common (707 classes)
    - Intermediate training on diverse bird species from around the world
    - Bio-DINO pretraining on approximately 31M natural biological images

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features: <https://arxiv.org/abs/2502.14786>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design: <https://arxiv.org/abs/2305.13035>
    - DINOv2: Learning Robust Visual Features without Supervision: <https://arxiv.org/abs/2304.07193>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("naflex_vit_reg4_so150m_p14_ls_dino-v2-bio-intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create a NaFlex inference transform
patch_size = net.stem_stride
max_seq_len = (size[0] // patch_size) * (size[1] // patch_size)
transform = birder.naflex_transform(patch_size, max_seq_len, model_info.rgb_stats)

# Option 2: helper (quick start with NaFlex preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform(
    "naflex_vit_reg4_so150m_p14_ls_dino-v2-bio-intermediate-eu-common",
    inference=True,
    naflex=True,
)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
out, _ = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 707), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("naflex_vit_reg4_so150m_p14_ls_dino-v2-bio-intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create a NaFlex inference transform
patch_size = net.stem_stride
max_seq_len = (size[0] // patch_size) * (size[1] // patch_size)
transform = birder.naflex_transform(patch_size, max_seq_len, model_info.rgb_stats)

# Option 2: helper (quick start with NaFlex preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform(
    "naflex_vit_reg4_so150m_p14_ls_dino-v2-bio-intermediate-eu-common",
    inference=True,
    naflex=True,
)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 896)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform(
    "naflex_vit_reg4_so150m_p14_ls_dino-v2-bio-intermediate-eu-common",
    inference=True,
)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 896, 24, 24]))]
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

@misc{oquab2024dinov2learningrobustvisual,
      title={DINOv2: Learning Robust Visual Features without Supervision},
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2024},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.07193},
}
```
