---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for se_resnext_50_arabian-peninsula

A SE ResNeXt image classification model. This model was trained on the `arabian-peninsula` dataset (all the relevant bird species found in the Arabian peninsula inc. rarities).

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 27.0
    - Input image size: 256 x 256
- **Dataset:** arabian-peninsula (735 classes)

- **Papers:**
    - Aggregated Residual Transformations for Deep Neural Networks: <https://arxiv.org/abs/1611.05431>
    - Squeeze-and-Excitation Networks: <https://arxiv.org/abs/1709.01507>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("se_resnext_50_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("se_resnext_50_arabian-peninsula", inference=True)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
out, _ = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 735), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("se_resnext_50_arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("se_resnext_50_arabian-peninsula", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 2048)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("se_resnext_50_arabian-peninsula", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 256, 64, 64])),
#  ('stage2', torch.Size([1, 512, 32, 32])),
#  ('stage3', torch.Size([1, 1024, 16, 16])),
#  ('stage4', torch.Size([1, 2048, 8, 8]))]
```

## Citation

```bibtex
@misc{xie2017aggregatedresidualtransformationsdeep,
      title={Aggregated Residual Transformations for Deep Neural Networks},
      author={Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He},
      year={2017},
      eprint={1611.05431},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1611.05431},
}

@misc{hu2019squeezeandexcitationnetworks,
      title={Squeeze-and-Excitation Networks},
      author={Jie Hu and Li Shen and Samuel Albanie and Gang Sun and Enhua Wu},
      year={2019},
      eprint={1709.01507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1709.01507},
}
```
