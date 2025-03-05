---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for van_b2_arabian-peninsula

An VAN image classification model. This model was trained on the `arabian-peninsula` dataset (all the relevant bird species found in the Arabian peninsula inc. rarities).

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

Note: A 256 x 256 variant of this model is available as `van_b2_arabian-peninsula256px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 26.4
    - Input image size: 384 x 384
- **Dataset:** arabian-peninsula (735 classes)

- **Papers:**
    - Visual Attention Network: <https://arxiv.org/abs/2202.09741>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("van_b2_arabian-peninsula", inference=True)
# Note: A 256x256 variant is available as "van_b2_arabian-peninsula256px"

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 735), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("van_b2_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 512)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("van_b2_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 64, 64, 64])),
#  ('stage2', torch.Size([1, 128, 32, 32])),
#  ('stage3', torch.Size([1, 320, 16, 16])),
#  ('stage4', torch.Size([1, 512, 8, 8]))]
```

## Citation

```bibtex
@misc{guo2022visualattentionnetwork,
      title={Visual Attention Network},
      author={Meng-Hao Guo and Cheng-Ze Lu and Zheng-Ning Liu and Ming-Ming Cheng and Shi-Min Hu},
      year={2022},
      eprint={2202.09741},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2202.09741},
}
```
