---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- birder-project/vit_reg4_m16_rms_avg_i-jepa
---

# Model Card for vit_reg4_m16_rms_avg_i-jepa-inat21

A ViT image classification model. The model follows a two-stage training process: first, I-JEPA pretraining, then fine-tuned on the `iNaturalist 2021` dataset - <https://github.com/visipedia/inat_comp/tree/master/2021>.

The model's class-to-index mapping uses original scientific names with full taxonomic rank,
a partial mapping to common names can be found here: <https://gitlab.com/birder/birder/-/blob/main/public_datasets_metadata/inat21-mapping.json>

Note: A 256 x 256 variant of this model is available as `vit_reg4_m16_rms_avg_i-jepa-inat21-256px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 43.6
    - Input image size: 384 x 384
- **Dataset:** iNaturalist 2021 (10000 classes)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture: <https://arxiv.org/abs/2301.08243>

- **Metrics:**
    - Top-1 accuracy of 256px model @ 224: 83.39%
    - Top-1 accuracy @ 384: 86.85%

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("vit_reg4_m16_rms_avg_i-jepa-inat21", inference=True)
# Note: A 224x224 variant is available as "vit_reg4_m16_rms_avg_i-jepa-inat21-256px"

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 10000), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("vit_reg4_m16_rms_avg_i-jepa-inat21", inference=True)

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

(net, model_info) = birder.load_pretrained_model("vit_reg4_m16_rms_avg_i-jepa-inat21", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 512, 24, 24]))]
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

@misc{darcet2024visiontransformersneedregisters,
      title={Vision Transformers Need Registers},
      author={Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2024},
      eprint={2309.16588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.16588},
}

@misc{assran2023selfsupervisedlearningimagesjointembedding,
      title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
      author={Mahmoud Assran and Quentin Duval and Ishan Misra and Piotr Bojanowski and Pascal Vincent and Michael Rabbat and Yann LeCun and Nicolas Ballas},
      year={2023},
      eprint={2301.08243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2301.08243},
}
```
