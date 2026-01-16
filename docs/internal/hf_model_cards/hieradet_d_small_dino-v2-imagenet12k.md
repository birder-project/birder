---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- birder-project/hieradet_d_small_dino-v2
datasets:
- timm/imagenet-12k-wds
---

# Model Card for hieradet_d_small_dino-v2-imagenet12k

HieraDet (dynamic window size) small image classification model. The model follows a two-stage training process: first, DINOv2 pretraining, then fine-tuned on the `ImageNet-12K` dataset.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 43.0
    - Input image size: 256 x 256
- **Dataset:** ImageNet-12K (11821 classes)

- **Papers:**
    - Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles: <https://arxiv.org/abs/2306.00989>
    - SAM 2: Segment Anything in Images and Videos: <https://arxiv.org/abs/2408.00714>
    - DINOv2: Learning Robust Visual Features without Supervision: <https://arxiv.org/abs/2304.07193>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("hieradet_d_small_dino-v2-imagenet12k", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 11821), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("hieradet_d_small_dino-v2-imagenet12k", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 768)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("hieradet_d_small_dino-v2-imagenet12k", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 96, 64, 64])),
#  ('stage2', torch.Size([1, 192, 32, 32])),
#  ('stage3', torch.Size([1, 384, 16, 16])),
#  ('stage4', torch.Size([1, 768, 8, 8]))]
```

## Citation

```bibtex
@misc{ryali2023hierahierarchicalvisiontransformer,
      title={Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
      author={Chaitanya Ryali and Yuan-Ting Hu and Daniel Bolya and Chen Wei and Haoqi Fan and Po-Yao Huang and Vaibhav Aggarwal and Arkabandhu Chowdhury and Omid Poursaeed and Judy Hoffman and Jitendra Malik and Yanghao Li and Christoph Feichtenhofer},
      year={2023},
      eprint={2306.00989},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2306.00989},
}

@misc{ravi2024sam2segmentimages,
      title={SAM 2: Segment Anything in Images and Videos},
      author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman Rädle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollár and Christoph Feichtenhofer},
      year={2024},
      eprint={2408.00714},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00714},
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
