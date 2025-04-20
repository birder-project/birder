---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- birder-project/hiera_abswin_base_mim
---

# Model Card for hiera_abswin_base_mim-intermediate-eu-common

A Hiera image classification model. The model follows a three-stage training process: first, masked image modeling, next intermediate training on a large-scale dataset containing diverse bird species from around the world, finally fine-tuned specifically on the `eu-common` dataset.

The species list is derived from the Collins bird guide [^1].

[^1]: Svensson, L., Mullarney, K., & Zetterström, D. (2022). Collins bird guide (3rd ed.). London, England: William Collins.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 51.1
    - Input image size: 384 x 384
- **Dataset:** eu-common (707 classes)
    - Intermediate training involved ~6000 species from asia, europe and africa

- **Papers:**
    - Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles: <https://arxiv.org/abs/2306.00989>
    - Window Attention is Bugged: How not to Interpolate Position Embeddings: <https://arxiv.org/abs/2311.05613>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("hiera_abswin_base_mim-intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 707), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("hiera_abswin_base_mim-intermediate-eu-common", inference=True)

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

(net, model_info) = birder.load_pretrained_model("hiera_abswin_base_mim-intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 96, 96, 96])),
#  ('stage2', torch.Size([1, 192, 48, 48])),
#  ('stage3', torch.Size([1, 384, 24, 24])),
#  ('stage4', torch.Size([1, 768, 12, 12]))]
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

@misc{bolya2023windowattentionbuggedinterpolate,
      title={Window Attention is Bugged: How not to Interpolate Position Embeddings},
      author={Daniel Bolya and Chaitanya Ryali and Judy Hoffman and Christoph Feichtenhofer},
      year={2023},
      eprint={2311.05613},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.05613},
}
```
