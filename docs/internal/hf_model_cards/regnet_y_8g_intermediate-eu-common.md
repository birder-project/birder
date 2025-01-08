---
tags:
- image-classification
- birder
library_name: birder
license: apache-2.0
---

# Model Card for regnet_y_8g_intermediate-eu-common

RegNet Y image classification model. The model follows a two-stage training process: first undergoing intermediate training on a large-scale dataset containing diverse bird species from around the world, then fine-tuned specifically on the `eu-common` dataset containing common European bird species.

The species list is derived from the Collins bird guide [^1].

[^1]: Svensson, L., Mullarney, K., & Zetterström, D. (2022). Collins bird guide (3rd ed.). London, England: William Collins.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 38.8
    - Input image size: 384 x 384
- **Dataset:** eu-common (707 classes)

- **Papers:**
    - Designing Network Design Spaces: <https://arxiv.org/abs/2003.13678>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("regnet_y_8g_intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(signature)

# Create an inference transform
transform = birder.classification_transform(size, rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, num_classes)
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("regnet_y_8g_intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(signature)

# Create an inference transform
transform = birder.classification_transform(size, rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, embedding_size)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("regnet_y_8g_intermediate-eu-common", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(signature)

# Create an inference transform
transform = birder.classification_transform(size, rgb_stats)

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
@misc{radosavovic2020designingnetworkdesignspaces,
      title={Designing Network Design Spaces}, 
      author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
      year={2020},
      eprint={2003.13678},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2003.13678}, 
}
```
