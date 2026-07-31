---
tags:
- image-feature-extraction
- birder
- pytorch
- biology
library_name: birder
license: apache-2.0
base_model:
- birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio
---

# Model Card for rope_deit3_m14_dino-v2-dist-bio

`rope_deit3_m14_dino-v2-dist-bio` is a compact Bio-DINO image encoder distilled from the larger [Bio-DINO SoViT-150M/14 model](https://huggingface.co/birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio).
It keeps the same natural-photography biodiversity scope as the teacher model, but uses a much smaller DeiT3-M/14-style student with 38.3M backbone parameters and 512-dimensional embeddings.

For the training-data description, intended scope, limitations, broader evaluation design and background on the Bio-DINO release, see the [teacher model card](https://huggingface.co/birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio).
This card focuses only on what is specific to the distilled model.

## Model Details

- **Model Type:** Image encoder and detection backbone
- **Model Stats:**
    - Params (M): 38.3
    - Input image size: 252 x 252
- **Dataset:** Trained on a diverse dataset of approximately 31M images, including:
    - TreeOfLife-10M-EOL-NaturalImages
    - iNaturalist 2021
    - BIOSCAN-5M (pretrain split)
    - TreeOfLife-200M (subset)
    - IP102 v1.1
    - iWildCam 2022 (subset)
    - The Birder dataset (private dataset)

- **Papers:**
    - DeiT III: Revenge of the ViT: <https://arxiv.org/abs/2204.07118>
    - Rotary Position Embedding for Vision Transformer: <https://arxiv.org/abs/2403.13298>
    - DINOv2: Learning Robust Visual Features without Supervision: <https://arxiv.org/abs/2304.07193>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("rope_deit3_m14_dino-v2-dist-bio", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("rope_deit3_m14_dino-v2-dist-bio", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 512)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("rope_deit3_m14_dino-v2-dist-bio", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 512, 18, 18]))]
```

## Citation

```bibtex
@misc{touvron2022deitiiirevengevit,
      title={DeiT III: Revenge of the ViT},
      author={Hugo Touvron and Matthieu Cord and Hervé Jégou},
      year={2022},
      eprint={2204.07118},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.07118},
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
