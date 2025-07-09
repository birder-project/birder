---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for pvt_v2_b2_mmcr-il-all

A PVT v2 image classification model. The model follows a two-stage training process: first undergoing self-supervised training (MMCR) on the `il-all` dataset, then fine-tuned on the same dataset. The dataset, encompassing all relevant bird species found in Israel, including rarities.

The species list is derived from data available at <https://www.israbirding.com/checklist/>.

Note: A 256 x 256 variant of this model is available as `pvt_v2_b2_mmcr-il-all256px`.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 25.1
    - Input image size: 384 x 384
- **Dataset:** il-all (550 classes)

- **Papers:**
    - PVT v2: Improved Baselines with Pyramid Vision Transformer: <https://arxiv.org/abs/2106.13797>
    - Learning Efficient Coding of Natural Images with Maximum Manifold Capacity Representations: <https://arxiv.org/abs/2303.03307>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("pvt_v2_b2_mmcr-il-all", inference=True)
# Note: A 256x256 variant is available as "pvt_v2_b2_mmcr-il-all256px"

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 550), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("pvt_v2_b2_mmcr-il-all", inference=True)

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

(net, model_info) = birder.load_pretrained_model("pvt_v2_b2_mmcr-il-all", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 64, 96, 96])),
#  ('stage2', torch.Size([1, 128, 48, 48])),
#  ('stage3', torch.Size([1, 320, 24, 24])),
#  ('stage4', torch.Size([1, 512, 12, 12]))]
```

## Citation

```bibtex
@article{Wang_2022,
   title={PVT v2: Improved baselines with pyramid vision transformer},
   volume={8},
   ISSN={2096-0662},
   url={http://dx.doi.org/10.1007/s41095-022-0274-8},
   DOI={10.1007/s41095-022-0274-8},
   number={3},
   journal={Computational Visual Media},
   publisher={Tsinghua University Press},
   author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
   year={2022},
   month=sep, pages={415–424}
}

@misc{yerxa2023learningefficientcodingnatural,
      title={Learning Efficient Coding of Natural Images with Maximum Manifold Capacity Representations},
      author={Thomas Yerxa and Yilun Kuang and Eero Simoncelli and SueYeon Chung},
      year={2023},
      eprint={2303.03307},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2303.03307},
}
```
