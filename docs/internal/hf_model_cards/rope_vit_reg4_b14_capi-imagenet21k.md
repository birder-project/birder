---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- birder-project/rope_vit_reg4_b14_capi
datasets:
- timm/imagenet-w21-webp-wds
---

# Model Card for rope_vit_reg4_b14_capi-imagenet21k

A RoPE ViT image classification model. The model follows a two-stage training process: first, CAPI pretraining, then fine-tuned on the `ImageNet-21K` dataset.

## RoPE Configuration

This model implements EVA-style Rotary Position Embedding (RoPE). When working with resolutions different from the training resolution (224x224), the model behavior can be optimized by configuring the `pt_grid_size` parameter:

- For inference at higher resolutions or when performing "shallow" fine-tuning, it's recommended to explicitly set `pt_grid_size=(16, 16)` (the default grid size during pretraining).
- For aggressive fine-tuning at higher resolutions, leave `pt_grid_size` as `None` to allow the model to adapt to the new resolution.

Setting `pt_grid_size` during inference:

```sh
# When running inference with a custom resolution (e.g., 336x336)
python predict.py --network rope_vit_reg4_b14 -t capi-imagenet21k --model-config '{"pt_grid_size":[16, 16]}' --size 336 ...
```

Converting the model with explicit RoPE configuration:

```sh
python tool.py convert-model --network rope_vit_reg4_b14 -t capi-imagenet21k --add-config '{"pt_grid_size":[16, 16]}'
```

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 100.5
    - Input image size: 224 x 224
- **Dataset:** ImageNet-21K (19167 classes)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Rotary Position Embedding for Vision Transformer: <https://arxiv.org/abs/2403.13298>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - Cluster and Predict Latent Patches for Improved Masked Image Modeling: <https://arxiv.org/abs/2502.08769>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rope_vit_reg4_b14_capi-imagenet21k", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 19167), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rope_vit_reg4_b14_capi-imagenet21k", inference=True)

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

(net, model_info) = birder.load_pretrained_model("rope_vit_reg4_b14_capi-imagenet21k", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 768, 16, 16]))]
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
