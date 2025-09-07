---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for vit_parallel_s16_18x2_ls_avg_data2vec-intermediate-arabian-peninsula

A ViT Parallel s16 18x2 image classification model. The model follows a three-stage training process: first, data2vec pretraining, next intermediate training on a large-scale dataset containing diverse bird species from around the world, finally fine-tuned specifically on the `arabian-peninsula` dataset.

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 64.7
    - Input image size: 384 x 384
- **Dataset:** arabian-peninsula (735 classes)
    - Intermediate training involved ~8000 species from all over the world

- **Papers:**
    - Three things everyone should know about Vision Transformers: <https://arxiv.org/abs/2203.09795>
    - data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language: <https://arxiv.org/abs/2202.03555>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("vit_parallel_s16_18x2_ls_avg_data2vec-intermediate-arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 735), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("vit_parallel_s16_18x2_ls_avg_data2vec-intermediate-arabian-peninsula", inference=True)

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

(net, model_info) = birder.load_pretrained_model("vit_parallel_s16_18x2_ls_avg_data2vec-intermediate-arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('neck', torch.Size([1, 384, 24, 24]))]
```

## Citation

```bibtex
@misc{touvron2022thingsknowvisiontransformers,
      title={Three things everyone should know about Vision Transformers},
      author={Hugo Touvron and Matthieu Cord and Alaaeldin El-Nouby and Jakob Verbeek and Hervé Jégou},
      year={2022},
      eprint={2203.09795},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2203.09795},
}

@misc{https://doi.org/10.48550/arxiv.2202.03555,
      title={data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language},
      author={Alexei Baevski and Wei-Ning Hsu and Qiantong Xu and Arun Babu and Jiatao Gu and Michael Auli},
      year={2022},
      eprint={2202.03555},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2202.03555},
}
```
