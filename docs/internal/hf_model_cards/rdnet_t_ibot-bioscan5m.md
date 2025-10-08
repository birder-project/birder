---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
datasets:
- bioscan-ml/BIOSCAN-5M
---

# Model Card for rdnet_t_ibot-bioscan5m

A RDNet tiny image encoder pre-trained using iBOT.

The model is primarily a feature extractor. Separately trained linear probing classification heads for various taxonomic levels (order, family, genus, species) are available for classification tasks.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 22.8
    - Input image size: 224 x 224
- **Dataset:** BIOSCAN-5M (pretrain split)

- **Papers:**
    - DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs: <https://arxiv.org/abs/2403.19588>
    - iBOT: Image BERT Pre-Training with Online Tokenizer: <https://arxiv.org/abs/2111.07832>

## Linear Probing Results

The following table shows the Top-1 Accuracy (%) achieved by training a linear classification head on top of the frozen `rdnet_t_ibot-bioscan5m` encoder.
The linear probing was conducted using 289,203 samples for all taxonomic levels, and the model was evaluated on the validation (14,757 samples) and test (39,373 samples) splits of the BIOSCAN-5M dataset.

| Taxonomic Level | Classes (N) | Val Top-1 Acc. (%) | Test Top-1 Acc. (%) |
|-----------------|-------------|--------------------|---------------------|
| Order           | 42          | 99.36              | 99.01               |
| Family          | 606         | 95.79              | 92.89               |
| Genus           | 4930        | 88.09              | 78.51               |
| Species         | 11846       | 79.74              | 65.26               |

## Unsupervised Evaluation (Adjusted Mutual Information)

The quality of the image embeddings was also evaluated intrinsically using Adjusted Mutual Information (AMI) following the setup of Lowe et al., 2024a ([An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders](https://arxiv.org/abs/2406.02465)):

1. Extract embeddings from the pretrained encoder.
1. Reduce dimensionality to 50 with [UMAP](https://arxiv.org/abs/1802.03426) (McInnes et al., 2018).
1. Cluster reduced embeddings using Agglomerative Clustering (Ward's method).
1. Compare against ground-truth taxonomic labels using AMI (Vinh et al., 2010).

The AMI score reflects how well the learned representations align with ground-truth taxonomy in an unsupervised setting.

| Taxonomic Level | AMI Score (%) |
|-----------------|---------------|
| Genus           | 39.14         |
| Species         | 26.91         |

## Model Usage

### Image Classification (with Linear Probing Head)

To use the model for classification, you must load the encoder and then load a specific pre-trained classification head for the desired taxonomic level. Heads are available for `order`, `family`, `genus`, and `species`.

```python
import torch
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rdnet_t_ibot-bioscan5m", inference=True)

# Load a linear probing classification head (e.g., for 'family')
head_data = torch.load("models/rdnet_t_ibot-bioscan5m-family.head.pt")

# Reset the classifier layer and load the head weights
net.reset_classifier(len(head_data["class_to_idx"]))
net.classifier.load_state_dict(head_data["state"])

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, N_CLASSES) for the chosen level, representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("rdnet_t_ibot-bioscan5m", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1040)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("rdnet_t_ibot-bioscan5m", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 256, 56, 56])),
#  ('stage2', torch.Size([1, 440, 28, 28])),
#  ('stage3', torch.Size([1, 744, 14, 14])),
#  ('stage4', torch.Size([1, 1040, 7, 7]))]
```

## Citation

```bibtex
@misc{kim2024densenetsreloadedparadigmshift,
      title={DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs},
      author={Donghyun Kim and Byeongho Heo and Dongyoon Han},
      year={2024},
      eprint={2403.19588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.19588},
}

@misc{zhou2022ibotimagebertpretraining,
      title={iBOT: Image BERT Pre-Training with Online Tokenizer},
      author={Jinghao Zhou and Chen Wei and Huiyu Wang and Wei Shen and Cihang Xie and Alan Yuille and Tao Kong},
      year={2022},
      eprint={2111.07832},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2111.07832},
}

@inproceedings{gharaee2024bioscan5m,
    title={{BIOSCAN-5M}: A Multimodal Dataset for Insect Biodiversity},
    booktitle={Advances in Neural Information Processing Systems},
    author={Zahra Gharaee and Scott C. Lowe and ZeMing Gong and Pablo Millan Arias
        and Nicholas Pellegrino and Austin T. Wang and Joakim Bruslund Haurum
        and Iuliia Zarubiieva and Lila Kari and Dirk Steinke and Graham W. Taylor
        and Paul Fieguth and Angel X. Chang
    },
    editor={A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
    pages={36285--36313},
    publisher={Curran Associates, Inc.},
    year={2024},
    volume={37},
    url={https://proceedings.neurips.cc/paper_files/paper/2024/file/3fdbb472813041c9ecef04c20c2b1e5a-Paper-Datasets_and_Benchmarks_Track.pdf},
}

```
