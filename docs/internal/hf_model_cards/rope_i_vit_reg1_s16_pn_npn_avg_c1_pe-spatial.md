---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- facebook/PE-Spatial-S16-512
---

# Model Card for rope_i_vit_reg1_s16_pn_npn_avg_c1_pe-spatial

A ViT-S16 image encoder from the PE-Spatial model by Bolya et al., converted to the Birder format for image feature extraction.
This version retains the original model weights and architecture, with the exception of treating the original CLS token as a REG token.
It is a general-purpose visual backbone.

See: <https://huggingface.co/facebook/PE-Spatial-S16-512> for further details.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 22.0
    - Input image size: 512 x 512

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Rotary Position Embedding for Vision Transformer: <https://arxiv.org/abs/2403.13298>
    - Perception Encoder: The best visual embeddings are not at the output of the network: <https://arxiv.org/abs/2504.13181>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rope_i_vit_reg1_s16_pn_npn_avg_c1_pe-spatial", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 384)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("rope_i_vit_reg1_s16_pn_npn_avg_c1_pe-spatial", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 384, 32, 32]))]
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

@misc{heo2024rotarypositionembeddingvision,
      title={Rotary Position Embedding for Vision Transformer},
      author={Byeongho Heo and Song Park and Dongyoon Han and Sangdoo Yun},
      year={2024},
      eprint={2403.13298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.13298},
}

@misc{bolya2025perceptionencoderbestvisual,
      title={Perception Encoder: The best visual embeddings are not at the output of the network},
      author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Dollár and Christoph Feichtenhofer},
      year={2025},
      eprint={2504.13181},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.13181},
}
```
