---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- google/siglip2-so400m-patch14-224
---

# Model Card for vit_so400m_p14_ap_siglip-v2-webli

A ViT so400m image encoder from the SigLIP-v2 model by Tschannen et al., converted to the Birder format for image feature extraction.
This version preserves the original model weights and architecture for downstream tasks.

See: <https://huggingface.co/google/siglip2-so400m-patch14-224> for further details.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 427.7
    - Input image size: 224 x 224

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design: <https://arxiv.org/abs/2305.13035>
    - SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features: <https://arxiv.org/abs/2502.14786>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("vit_so400m_p14_ap_siglip-v2-webli", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1152)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("vit_so400m_p14_ap_siglip-v2-webli", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 1152, 16, 16]))]
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

@misc{alabdulmohsin2024gettingvitshapescaling,
      title={Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design},
      author={Ibrahim Alabdulmohsin and Xiaohua Zhai and Alexander Kolesnikov and Lucas Beyer},
      year={2024},
      eprint={2305.13035},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2305.13035},
}

@misc{tschannen2025siglip2multilingualvisionlanguage,
      title={SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features},
      author={Michael Tschannen and Alexey Gritsenko and Xiao Wang and Muhammad Ferjad Naeem and Ibrahim Alabdulmohsin and Nikhil Parthasarathy and Talfan Evans and Lucas Beyer and Ye Xia and Basil Mustafa and Olivier Hénaff and Jeremiah Harmsen and Andreas Steiner and Xiaohua Zhai},
      year={2025},
      eprint={2502.14786},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.14786},
}
```
