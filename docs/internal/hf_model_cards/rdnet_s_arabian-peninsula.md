---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for rdnet_s_arabian-peninsula

An RDNet image classification model. This model was trained on the `arabian-peninsula` dataset (all the relevant bird species found in the Arabian peninsula inc. rarities).

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

Note: A 256 x 256 variant of this model is available as `rdnet_s_arabian-peninsula256px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 50.1
    - Input image size: 384 x 384
- **Dataset:** arabian-peninsula (735 classes)

- **Papers:**
    - DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs: <https://arxiv.org/abs/2403.19588>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rdnet_s_arabian-peninsula", inference=True)
# Note: A 256x256 variant is available as "rdnet_s_arabian-peninsula256px"

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

(net, model_info) = birder.load_pretrained_model("rdnet_s_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1264)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("rdnet_s_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 264, 96, 96])),
#  ('stage2', torch.Size([1, 512, 48, 48])),
#  ('stage3', torch.Size([1, 760, 24, 24])),
#  ('stage4', torch.Size([1, 1264, 12, 12]))]
```

## Citation

```bibtex
@misc{kim2024densenetsreloadedparadigmshift,
      title={DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs},
      author={Donghyun Kim and Byeongho Heo and Dongyoon Han},
      year={2024},
      eprint={2403.19588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.19588},
}
```
