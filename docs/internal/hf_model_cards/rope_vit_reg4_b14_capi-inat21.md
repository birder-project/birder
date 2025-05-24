---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- birder-project/rope_vit_reg4_b14_capi
---

# Model Card for rope_vit_reg4_b14_capi-inat21

A RoPE ViT image classification model. The model follows a two-stage training process: first, CAPI pretraining, then fine-tuned on the `iNaturalist 2021` dataset - <https://github.com/visipedia/inat_comp/tree/master/2021>.

Note: A 224 x 224 variant of this model is available as `rope_vit_reg4_b14_capi-inat21-224px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 93.6
    - Input image size: 336 x 336
- **Dataset:** iNaturalist 2021 (10000 classes)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Rotary Position Embedding for Vision Transformer: <https://arxiv.org/abs/2403.13298>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - Cluster and Predict Latent Patches for Improved Masked Image Modeling: <https://arxiv.org/abs/2502.08769>

- **Metrics:**
    - Top-1 accuracy of 224px model @ 224: 87.15%
    - Top-1 accuracy of 224px model @ 336: 88.62%
    - Top-1 accuracy @ 336: 90.01%

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rope_vit_reg4_b14_capi-inat21", inference=True)
# Note: A 224x224 variant is available as "rope_vit_reg4_b14_capi-inat21-224px"

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

(net, model_info) = birder.load_pretrained_model("rope_vit_reg4_b14_capi-inat21", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 768)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("rope_vit_reg4_b14_capi-inat21", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 768, 24, 24]))]
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

@misc{darcet2024visiontransformersneedregisters,
      title={Vision Transformers Need Registers}, 
      author={Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2024},
      eprint={2309.16588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.16588}, 
}

@misc{darcet2025clusterpredictlatentpatches,
      title={Cluster and Predict Latent Patches for Improved Masked Image Modeling},
      author={Timothée Darcet and Federico Baldassarre and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2025},
      eprint={2502.08769},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.08769},
}
```
