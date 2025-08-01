---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: mit
base_model:
- openai/clip-vit-large-patch14
---

# Model Card for vit_l14_pn_quick_gelu_openai-clip

A ViT l14 image encoder from the original OpenAI CLIP model by Radford et al., converted to the Birder format for image feature extraction.
This version preserves the original model weights and architecture, including the CLIP projection layer for further downstream tasks.

See: <https://huggingface.co/openai/clip-vit-large-patch14> and <https://github.com/openai/CLIP> for further details.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 304.0
    - Input image size: 224 x 224

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Learning Transferable Visual Models From Natural Language Supervision: <https://arxiv.org/abs/2103.00020>

## Model Usage

### CLIP Projections

```python
from PIL import Image

import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("vit_l14_pn_quick_gelu_openai-clip", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Method 1: Using the convenience function
image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(projection, _) = infer_image(net, image, transform, return_logits=True)
# projection is a NumPy array with shape of (1, 768)

# Method 2: Direct model usage
image = Image.open("path/to/image.jpeg")
projection = net(transform(image).unsqueeze(dim=0))
# projection is a torch.Tensor with torch.Size([1, 768])
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("vit_l14_pn_quick_gelu_openai-clip", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1024)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("vit_l14_pn_quick_gelu_openai-clip", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 1024, 16, 16]))]
```

## Citation

```bibtex
@misc{dosovitskiy2021imageworth16x16words,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2021},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2010.11929},
}

@misc{radford2021learningtransferablevisualmodels,
      title={Learning Transferable Visual Models From Natural Language Supervision},
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2103.00020},
}
```
