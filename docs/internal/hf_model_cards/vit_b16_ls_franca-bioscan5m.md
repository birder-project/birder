---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
datasets:
- bioscan-ml/BIOSCAN-5M
---

# Model Card for vit_b16_ls_franca-bioscan5m

A ViT b16 image encoder pre-trained using Franca.

The model is primarily a feature extractor. Separately trained linear probing classification heads for various taxonomic levels (order, family, genus, species) are available for classification tasks.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 85.8
    - Input image size: 224 x 224
- **Dataset:** BIOSCAN-5M (pretrain split)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning: <https://arxiv.org/abs/2507.14137>

## Linear Probing Results

The following table shows the Top-1 Accuracy (%) achieved by training a linear classification head on top of the frozen `vit_b16_ls_franca-bioscan5m` encoder.
The linear probing was conducted using 289,203 samples for all taxonomic levels, and the model was evaluated on the validation (14,757 samples) and test (39,373 samples) splits of the BIOSCAN-5M dataset.

| Taxonomic Level | Classes (N) | Val Top-1 Acc. (%) | Test Top-1 Acc. (%) |
|-----------------|-------------|--------------------|---------------------|
| Order           | 42          | 99.56              | 99.47               |
| Family          | 606         | 97.61              | 96.44               |
| Genus           | 4930        | 94.36              | 90.14               |
| Species         | 11846       | 89.46              | 82.30               |

## Unsupervised Evaluation (Adjusted Mutual Information)

The quality of the image embeddings was also evaluated intrinsically using Adjusted Mutual Information (AMI) following the setup of Lowe et al., 2024a ([An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders](https://arxiv.org/abs/2406.02465)):

1. Extract embeddings from the pretrained encoder.
1. Reduce dimensionality to 50 with [UMAP](https://arxiv.org/abs/1802.03426) (McInnes et al., 2018).
1. Cluster reduced embeddings using Agglomerative Clustering (Ward's method).
1. Compare against ground-truth taxonomic labels using AMI (Vinh et al., 2010).

The AMI score reflects how well the learned representations align with ground-truth taxonomy in an unsupervised setting.

| Taxonomic Level | AMI Score (%) |
|-----------------|---------------|
| Genus           | 62.92         |
| Species         | 43.75         |

## Model Usage

### Image Classification (with Linear Probing Head)

To use the model for classification, you must load the encoder and then load a specific pre-trained classification head for the desired taxonomic level. Heads are available for `order`, `family`, `genus`, and `species`.

```python
import torch
import birder
from birder.inference.classification import infer_image

(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_b16_ls_franca-bioscan5m", inference=True)

# Load a linear probing classification head (e.g., for 'family')
head_data = torch.load("models/vit_b16_ls_franca-bioscan5m-family.head.pt")

# Reset the classifier layer and load the head weights
net.reset_classifier(len(head_data["class_to_idx"]))
net.classifier.load_state_dict(head_data["state"])

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, N_CLASSES) for the chosen level, representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
(net, model_info) = birder.load_pretrained_model("vit_b16_ls_franca-bioscan5m", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_b16_ls_franca-bioscan5m", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 768)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_b16_ls_franca-bioscan5m", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 768, 14, 14]))]
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

@misc{venkataramanan2026francanestedmatryoshkaclustering,
      title={Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning},
      author={Shashanka Venkataramanan and Valentinos Pariza and Mohammadreza Salehi and Lukas Knobel and Spyros Gidaris and Elias Ramzi and Andrei Bursuc and Yuki M. Asano},
      year={2026},
      eprint={2507.14137},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.14137},
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
