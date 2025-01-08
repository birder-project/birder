---
tags:
- image-classification
- birder
library_name: birder
license: apache-2.0
---

# Model Card for iformer_s_arabian-peninsula

iFormer image classification model. This model was trained on the `arabian-peninsula` dataset (all the relevant bird species found in the Arabian peninsula inc. rarities).

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 20.8
    - Input image size: 384 x 384
- **Dataset:** arabian-peninsula (735)

- **Papers:**
    - Inception Transformer: <https://arxiv.org/abs/2205.12956>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("iformer_s_arabian-peninsula", inference=True)

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

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("iformer_s_arabian-peninsula", inference=True)

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

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("iformer_s_arabian-peninsula", inference=True)

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
@misc{si2022inceptiontransformer,
      title={Inception Transformer}, 
      author={Chenyang Si and Weihao Yu and Pan Zhou and Yichen Zhou and Xinchao Wang and Shuicheng Yan},
      year={2022},
      eprint={2205.12956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2205.12956}, 
}
```
