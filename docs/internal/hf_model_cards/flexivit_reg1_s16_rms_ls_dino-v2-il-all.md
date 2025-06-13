---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for flexivit_reg1_s16_rms_ls_dino-v2-il-all

FlexiViT reg1 s16 RMS norm with layer scaling classification model pre-trained using DINOv2 on the `il-all` dataset and then fine-tuned on the `il-all` dataset.

The species list is derived from data available at <https://www.israbirding.com/checklist/>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 21.9
    - Input image size: 240 x 240
- **Dataset:** il-all (550 classes)

- **Papers:**
    - FlexiViT: One Model for All Patch Sizes: <https://arxiv.org/abs/2212.08013>
    - DINOv2: Learning Robust Visual Features without Supervision: <https://arxiv.org/abs/2304.07193>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("flexivit_reg1_s16_rms_ls_dino-v2-il-all", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 550), representing class probabilities.

# Use the flexible patch size of FlexiViT
(out, _) = infer_image(net, image, transform, patch_size=24)
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("flexivit_reg1_s16_rms_ls_dino-v2-il-all", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 384)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("flexivit_reg1_s16_rms_ls_dino-v2-il-all", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 384, 15, 15]))]
```

## Citation

```bibtex
@misc{beyer2023flexivitmodelpatchsizes,
      title={FlexiViT: One Model for All Patch Sizes},
      author={Lucas Beyer and Pavel Izmailov and Alexander Kolesnikov and Mathilde Caron and Simon Kornblith and Xiaohua Zhai and Matthias Minderer and Michael Tschannen and Ibrahim Alabdulmohsin and Filip Pavetic},
      year={2023},
      eprint={2212.08013},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2212.08013},
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
