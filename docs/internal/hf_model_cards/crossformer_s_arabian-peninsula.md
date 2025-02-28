---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for crossformer_s_arabian-peninsula

An CrossFormer image classification model. This model was trained on the `arabian-peninsula` dataset (all the relevant bird species found in the Arabian peninsula inc. rarities).

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

Note: A 256 x 256 variant of this model is available as `crossformer_s_arabian-peninsula256px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 30.5
    - Input image size: 384 x 384
- **Dataset:** arabian-peninsula (735 classes)

- **Papers:**
    - CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention: <https://arxiv.org/abs/2108.00154>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("crossformer_s_arabian-peninsula", inference=True)
# Note: A 256x256 variant is available as "crossformer_s_arabian-peninsula256px"

# Get the image size the model was trained on
size = birder.get_size_from_signature(signature)

# Create an inference transform
transform = birder.classification_transform(size, rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, num_classes), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("crossformer_s_arabian-peninsula", inference=True)

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

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("crossformer_s_arabian-peninsula", inference=True)

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
@misc{wang2021crossformerversatilevisiontransformer,
      title={CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention},
      author={Wenxiao Wang and Lu Yao and Long Chen and Binbin Lin and Deng Cai and Xiaofei He and Wei Liu},
      year={2021},
      eprint={2108.00154},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2108.00154},
}
```
