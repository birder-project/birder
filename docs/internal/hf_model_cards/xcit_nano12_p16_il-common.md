---
tags:
- image-classification
- birder
library_name: birder
license: apache-2.0
---

# Model Card for xcit_nano12_p16_il-common

XCiT image classification model. This model was trained on the `il-common` dataset (common bird species found in Israel).

The species list is derived from data available at <https://www.israbirding.com/checklist/>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
  - Params (M): 3.0
  - Input image size: 256 x 256
- **Dataset:** il-common (371 classes)

- **Papers:**
  - XCiT: Cross-Covariance Image Transformers: <https://arxiv.org/abs/2106.09681>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("xcit_nano12_p16_il-common", inference=True)

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

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("xcit_nano12_p16_il-common", inference=True)

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

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("xcit_nano12_p16_il-common", inference=True)

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
@misc{elnouby2021xcitcrosscovarianceimagetransformers,
      title={XCiT: Cross-Covariance Image Transformers}, 
      author={Alaaeldin El-Nouby and Hugo Touvron and Mathilde Caron and Piotr Bojanowski and Matthijs Douze and Armand Joulin and Ivan Laptev and Natalia Neverova and Gabriel Synnaeve and Jakob Verbeek and Hervé Jegou},
      year={2021},
      eprint={2106.09681},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2106.09681}, 
}
```
