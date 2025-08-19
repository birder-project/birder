---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for hiera_abswin_base_mim

A Hiera with absolute window position embedding strategy image encoder pre-trained using Masked Image Modeling (MIM). This model has *not* been fine-tuned for a specific classification task and is intended to be used as a general-purpose feature extractor or a backbone for downstream tasks like object detection, segmentation, or custom classification.

The full MAE model is available at <https://huggingface.co/birder-project/mae_hiera_hiera_abswin_base>

## Model Details

- **Model Type:** Image encoder and detection backbone
- **Model Stats:**
    - Params (M): 50.5
    - Input image size: 224 x 224
- **Dataset:** Trained on a diverse dataset of approximately 12M images, including:
    - iNaturalist 2021 (~2.7M)
    - WebVision-2.0 (~1.5M random subset)
    - imagenet-w21-webp-wds (~1M random subset)
    - SA-1B (~220K random subset of 20 chunks)
    - COCO (~120K)
    - NABirds (~48K)
    - GLDv2 (~40K random subset of 6 chunks)
    - Birdsnap v1.1 (~44K)
    - CUB-200 2011 (~11K)
    - The Birder dataset (~6M, private dataset)

- **Papers:**
    - Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles: <https://arxiv.org/abs/2306.00989>
    - Window Attention is Bugged: How not to Interpolate Position Embeddings: <https://arxiv.org/abs/2311.05613>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("hiera_abswin_base_mim", inference=True)

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

(net, model_info) = birder.load_pretrained_model("hiera_abswin_base_mim", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 96, 56, 56])),
#  ('stage2', torch.Size([1, 192, 28, 28])),
#  ('stage3', torch.Size([1, 384, 14, 14])),
#  ('stage4', torch.Size([1, 768, 7, 7]))]
```

## Citation

```bibtex
@misc{ryali2023hierahierarchicalvisiontransformer,
      title={Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
      author={Chaitanya Ryali and Yuan-Ting Hu and Daniel Bolya and Chen Wei and Haoqi Fan and Po-Yao Huang and Vaibhav Aggarwal and Arkabandhu Chowdhury and Omid Poursaeed and Judy Hoffman and Jitendra Malik and Yanghao Li and Christoph Feichtenhofer},
      year={2023},
      eprint={2306.00989},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2306.00989},
}

@misc{bolya2023windowattentionbuggedinterpolate,
      title={Window Attention is Bugged: How not to Interpolate Position Embeddings},
      author={Daniel Bolya and Chaitanya Ryali and Judy Hoffman and Christoph Feichtenhofer},
      year={2023},
      eprint={2311.05613},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.05613},
}
```
