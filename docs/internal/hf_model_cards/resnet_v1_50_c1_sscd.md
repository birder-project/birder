---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: mit
---

# Model Card for resnet_v1_50_c1_sscd

A ResNet v1 model designed to be used for image copy detection, converted to the Birder format for image feature extraction. This version retains the original model weights. The model produces 512-dimensional L2 normalized descriptors for each input image.

The similarity between two images, represented by their descriptors a and b, can be effectively measured using descriptor cosine similarity `a.dot(b)`, where higher values indicate greater similarity.
Alternatively, Euclidean distance `torch.linalg.vector_norm(a-b)` can be used, with lower values indicating greater similarity.
For reference, descriptor cosine similarity greater than 0.75 indicates copies with 90% precision.

For optimal performance, particularly when sample images from the target distribution are available, additional descriptor post-processing is recommended.
This includes techniques such as centering (subtracting the mean) followed by L2 normalization, or whitening followed by L2 normalization, both of which can enhance accuracy.
Furthermore, applying score normalization can lead to more consistent similarity measurements and improve global accuracy metrics, although it does not impact ranking metrics.

For further information see: <https://github.com/facebookresearch/sscd-copy-detection>

## Model Details

- **Model Type:** Image copy detection
- **Model Stats:**
    - Params (M): 24.6
    - Input image size: 320 x 320
- **Dataset:** DISC21: Dataset for the Image Similarity Challenge 2021

- **Papers:**
    - Deep Residual Learning for Image Recognition: <https://arxiv.org/abs/1512.03385>
    - A Self-Supervised Descriptor for Image Copy Detection: <https://arxiv.org/abs/2202.10261>

## Model Usage

### Image Copy Detection

```python
import torch
import torch.nn.functional as F
from PIL import Image

import birder
from birder.inference.classification import infer_image
from birder.net.ssl.sscd import SSCD

(backbone, model_info) = birder.load_pretrained_model("resnet_v1_50_c1_sscd", inference=True)
net = SSCD(backbone)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image1 = Image.open("path/to/image1.jpeg")
image2 = Image.open("path/to/image2.jpeg")
out1 = net(transform(image1).unsqueeze(dim=0))
out2 = net(transform(image2).unsqueeze(dim=0))
# Both out1 and out2 have torch.Size([1, 512])

# Calculate cosine similarity (higher = more similar, range: -1 to 1)
F.cosine_similarity(out1, out2, dim=1)

# Calculate Euclidean distance (lower = more similar)
torch.linalg.vector_norm(out1 - out2, dim=1)
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("resnet_v1_50_c1_sscd", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 2048)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("resnet_v1_50_c1_sscd", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 256, 80, 80])),
#  ('stage2', torch.Size([1, 512, 40, 40])),
#  ('stage3', torch.Size([1, 1024, 20, 20])),
#  ('stage4', torch.Size([1, 2048, 10, 10]))]
```

## Citation

```bibtex
@misc{he2015deepresiduallearningimage,
      title={Deep Residual Learning for Image Recognition},
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1512.03385},
}

@misc{pizzi2022selfsuperviseddescriptorimagecopy,
      title={A Self-Supervised Descriptor for Image Copy Detection},
      author={Ed Pizzi and Sreya Dutta Roy and Sugosh Nagavara Ravindra and Priya Goyal and Matthijs Douze},
      year={2022},
      eprint={2202.10261},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2202.10261},
}
```
