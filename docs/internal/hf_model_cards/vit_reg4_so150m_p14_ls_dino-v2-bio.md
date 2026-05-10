---
tags:
- image-feature-extraction
- birder
- pytorch
- biology
library_name: birder
license: apache-2.0
base_model:
- birder-project/dino_v2_vit_reg4_so150m_p14_ls_bio
---

# Model Card for vit_reg4_so150m_p14_ls_dino-v2-bio

`vit_reg4_so150m_p14_ls_dino-v2-bio` is a Bio-DINO image encoder for natural photographs of living organisms.
It uses a SoViT-150M/14 Vision Transformer with 4 register tokens and 133.6M backbone parameters, trained with a DINOv2-style self-supervised objective on approximately 31 million curated images spanning plants, fungi, insects, fish, corals, birds, mammals and other biodiversity subjects.

The model is released as a single repository with three checkpoints: 224 x 224, 252 x 252 and 336 x 336.
The 252px checkpoint is the recommended default, the 224px checkpoint is a faster and lower-cost variant, and the 336px checkpoint is a short high-resolution adaptation for users who want to trade latency for more input detail.
All checkpoints are intended primarily as frozen image encoders for embeddings, retrieval, clustering, probing and transfer learning, rather than as ready-made species classifiers.

This repository releases the Birder backbone checkpoints for direct use as image encoders.
The full DINO training weights, including the DINO head and training state needed for continued self-supervised work, are released separately in the [full DINO training weights companion repository](https://huggingface.co/birder-project/dino_v2_vit_reg4_so150m_p14_ls_bio).

## Model Details

- **Model Type:** Image encoder and detection backbone
- **Model Stats:**
    - Params (M): 133.6
    - Input image size: 252 x 252
    - Additional released sizes: 224 x 224, 336 x 336
- **Dataset:** Trained on a diverse dataset of approximately 31M images, including:
    - TreeOfLife-10M-EOL-NaturalImages
    - iNaturalist 2021
    - BIOSCAN-5M (pretrain split)
    - TreeOfLife-200M (subset)
    - IP102 v1.1
    - iWildCam 2022 (subset)
    - The Birder dataset (private dataset)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design: <https://arxiv.org/abs/2305.13035>
    - DINOv2: Learning Robust Visual Features without Supervision: <https://arxiv.org/abs/2304.07193>

## Released Checkpoints / Model Variants

All three backbone checkpoints are released in this model repository.

| Checkpoint | Resolution | Training stage | Recommended use |
| ---------- | ---------: | -------------- | --------------- |
| `vit_reg4_so150m_p14_ls_dino-v2-bio-224px` | 224 x 224 | Initial DINOv2 training resolution | Fastest Bio-DINO checkpoint, useful when throughput or memory is the main constraint |
| `vit_reg4_so150m_p14_ls_dino-v2-bio-252px` | 252 x 252 | Continued training from 224px | Recommended default checkpoint for most embedding and transfer workflows |
| `vit_reg4_so150m_p14_ls_dino-v2-bio-336px` | 336 x 336 | Short high-resolution adaptation from 252px | Higher-input-detail checkpoint, useful when latency and memory are less constrained |
| [Full DINO training weights](https://huggingface.co/birder-project/dino_v2_vit_reg4_so150m_p14_ls_bio) | varies | - | Continued self-supervised training and research |

## Intended Use And Scope

This model is intended for natural photographs of living organisms and biodiversity subjects: organismal field photography, camera-trap imagery, collection-style photographs, underwater photography and similar real-world images of living systems.
It is best used as a general-purpose image encoder for representation-learning workflows, especially when the downstream task benefits from visual similarity, fine-grained biological structure or transfer from broad biodiversity imagery.
Image retrieval and visual similarity search are primary use cases for Bio-DINO.
The model was evaluated heavily as a frozen embedding model and is particularly well suited to finding visually related biological images across large collections.

### Suitable Uses

Suitable uses include:

- Image embedding extraction for natural photographs of living organisms.
- Nearest-neighbor retrieval, visual similarity search and reference-image matching.
- Clustering, dataset exploration, and biodiversity image organization.
- Dataset deduplication, near-duplicate review, and collection triage.
- Frozen-embedding evaluation with linear probes, k-NN, SimpleShot, SVM, MLP probes, or retrieval metrics.
- Lightweight supervised transfer for taxonomic, ecological, trait, or visual-recognition tasks.
- Fine-tuning or initialization for downstream biodiversity image models.
- Continued self-supervised training and representation-learning research.

### Out-of-Scope Domains

The model is not intended for microscopy, pathology, medical imaging, X-ray/radiology, satellite imagery, maps, diagrams, charts, drawings, document understanding or other non-photographic visual domains.
It was trained for natural photography, not controlled lab imagery, scientific instrument imagery or clinical imagery.

## Training Data

The model was trained on approximately 31 million curated images focused on natural photographs of living organisms and biodiversity subjects.
The training mixture combines public biodiversity datasets, selected subsets of larger public image collections, and curated private Birder data sources.
The goal of the corpus was broad visual coverage of the biological world rather than a fixed taxonomy, a supervised class list or a balanced benchmark-style dataset.

At a high level, the curation process follows the same concept as the Birder Vision Data Curation walkthrough for [TreeOfLife-10M](https://gitlab.com/birder/vision-data-curation/-/blob/main/docs/example/tree_of_life_10m.md): input validation, near-duplicate removal, example-based filtering of non-natural images, quality/aesthetic filtering, and diversity-oriented sampling.
The filtering process was designed to emphasize real photographs of living organisms and reduce illustrations, documents, maps, charts, diagrams, screenshots, lab imagery, medical imagery and other content outside the intended domain.

The final corpus includes broad biological coverage across plants, fungi, insects, fish, corals, birds, mammals and other organismal groups.

Known public components and source families used in the training mixture include:

- TreeOfLife-10M-EOL-NaturalImages
- iNaturalist 2021
- BIOSCAN-5M pretrain split
- TreeOfLife-200M subset
- IP102 v1.1
- iWildCam 2022 subset

## Training Procedure

The model was trained with a DINOv2-style self-supervised objective using a SoViT-150M/14 backbone with 4 register tokens.
The longest training stage was run at 224 x 224 for efficiency, then continued at 252 x 252 for the main release checkpoint and finished with a short 336 x 336 high-resolution adaptation.

### Resolution Schedule

| Stage | Resolution | Release checkpoint                         | Notes                                                      |
| ----: | ---------: | ------------------------------------------ | ---------------------------------------------------------- |
| 1     | 224 x 224  | `vit_reg4_so150m_p14_ls_dino-v2-bio-224px` | Initial and longest training stage                         |
| 2     | 252 x 252  | `vit_reg4_so150m_p14_ls_dino-v2-bio-252px` | Continued training from 224px, recommended default release |
| 3     | 336 x 336  | `vit_reg4_so150m_p14_ls_dino-v2-bio-336px` | Short high-resolution adaptation from 252px                |

Detailed launch/configuration files for the self-supervised runs are published with the release.
The 224px launch/configuration file is available here: [224px Slurm file](train_distributed_dino-v2_bio.sh).

## Evaluation Overview

The evaluation is designed to assess representation quality across diverse biological and biodiversity image tasks.

Evaluation protocols vary by dataset. They are intended to test whether frozen representations are useful for separability, neighborhood structure, trait prediction, retrieval, clustering and lightweight transfer, rather than to optimize an end-to-end classifier for a single benchmark.
For this reason, the benchmarks use simple probes and non-parametric methods over frozen features wherever possible.

The reference set includes biology-focused CLIP models such as BioCLIP v1, BioCLIP v2 and BioCLIP v2.5, generic CLIP/VLM models such as PE-Core and SigLIP, generic self-supervised encoders such as DINOv2 and DINOv3, a supervised ConvNeXt ImageNet-22K reference and the spatially focused TIPS-v2 model.
These references are included to contextualize embedding quality across different training paradigms, not as a controlled ablation of architecture, data or supervision.

### Evaluation Protocols

| Protocol | Purpose | Notes |
| -------- | ------- | ----- |
| Linear probe | Measures class separability in frozen embeddings | Used for Plankton and SnakeCLEF2023 |
| k-NN | Measures local neighborhood quality in embedding space | Used for NABirds, Butterflies and Moths Austria, FungiCLEF2023, and CCT category classification |
| SimpleShot | Measures prototype-style transfer with frozen embeddings | Used for Flowers102, ImageNet-1K, PlantDoc, PlantNet-300K, and CCT non-empty classification |
| SVM | Measures shallow supervised separability across many small tasks | Used for NeWT binary tasks |
| MLP probe | Measures lightweight supervised transfer from frozen embeddings | Used for FishNet multi-label trait prediction and AwA2 attribute prediction |
| AMI clustering | Measures unsupervised clustering structure | Used for BIOSCAN-5M with UMAP reduction and L2 normalization |
| Retrieval | Measures image retrieval and similarity-search quality | Used for CUB-200-2011 with mAP and Recall@K |
| Full linear probe | Measures supervised transfer with a frozen backbone and trained classification head | Used for iNaturalist 2021 |

### Benchmark Results

The benchmark suite was used both to select release checkpoints and to understand how the representation improved during training.
The charts below show Bio-DINO checkpoints across training epochs against contextual reference models.
The three released checkpoints correspond to 224px at epoch 185, 252px at epoch 245, and 336px at epoch 250.

![Aggregate benchmark progression](img/benchmark_summary_ref_minmax_raw.png)

Aggregate benchmark progression over Bio-DINO training. Points show Bio-DINO checkpoints across training epochs, with the released 224px, 252px and 336px checkpoints marked by stars. Dashed lines show reference encoders evaluated under the same local protocols. The aggregate score is computed by normalizing each benchmark against the reference-model range and averaging across benchmarks, so it should be read as a relative summary of this evaluation suite rather than an absolute measure of biological expertise.

![CUB-200-2011 retrieval benchmark progression](img/benchmark_cub200.png)

CUB-200-2011 retrieval benchmark progression over Bio-DINO training. Points show Bio-DINO checkpoints across training epochs, with the released 224px, 252px, and 336px checkpoints marked by stars. The metric is mean average precision, using frozen image embeddings for visual retrieval.

![NeWT benchmark progression](img/benchmark_newt.png)

NeWT benchmark progression. Accuracy is measured with SVM classifiers over the NeWT binary task suite using frozen image embeddings.

![SnakeCLEF2023 benchmark progression](img/benchmark_snakeclef.png)

SnakeCLEF2023 benchmark progression. Accuracy is measured with a linear probe over frozen image embeddings.

The iNaturalist 2021 result uses a fuller linear-probing setup than the frozen-embedding evaluations above: the backbone is frozen, but the classification head is trained for multiple epochs with light image augmentation.

| Model                         | Resolution | Params (M) | Accuracy | Top-3 accuracy | Macro F1 |
| ----------------------------- | ---------: | ---------: | -------: | -------------: | -------: |
| Bio-DINO 224px                |        224 |      142.6 |   0.8572 |         0.9420 |   0.8564 |
| Bio-DINO 252px                |        252 |      142.6 |   0.8709 |         0.9510 |   0.8702 |
| Bio-DINO 336px                |        336 |      142.6 |   0.8807 |         0.9567 |   0.8800 |
| BioCLIP v1 ViT-B/16           |        224 |       93.5 |   0.7890 |         0.9005 |   0.7874 |
| BioCLIP v2 ViT-L/14           |        224 |      313.4 |   0.9169 |         0.9745 |   0.9164 |
| BioTrove-CLIP-O ViT-B/16      |        224 |       93.5 |   0.8351 |         0.9290 |   0.8334 |
| DINOv2 ViT-B/14               |        224 |       93.4 |   0.7492 |         0.8645 |   0.7457 |
| DINOv3 RoPE ViT-B/16          |        256 |       93.4 |   0.7826 |         0.8889 |   0.7795 |
| DINOv3 RoPE ViT-L/16          |        256 |      313.4 |   0.8333 |         0.9235 |   0.8309 |
| PE-Core RoPE ViT-B/16         |        224 |      100.6 |   0.6831 |         0.8299 |   0.6788 |
| ConvNeXt v1 L ImageNet-22K    |        224 |      211.6 |   0.7317 |         0.8647 |   0.7289 |

For the iNaturalist 2021 linear-probe table, parameter counts include the frozen image encoder plus the trained 10k-class linear classification head.

The reference models are included for context only.
They differ in architecture, model size, pretraining data, supervision type, intended use, preprocessing and input resolution, so these comparisons should be read as local benchmark context rather than as controlled ablations.

Results reflect the local protocols above, not absolute biological expertise.
They may change with preprocessing, crop policy, input resolution, probe hyperparameters, random seed and embedding extraction layer.

## Efficiency / Latency vs Quality

The three Bio-DINO checkpoints expose a practical speed/quality tradeoff.
The 252px checkpoint is the recommended default, the 224px checkpoint is faster and lower-cost, and the 336px checkpoint can improve some detail-sensitive tasks at a substantial latency cost.

The figure below shows the CUDA latency vs accuracy Pareto view from the iNaturalist 2021 linear-probe evaluation.
It is intended as a deployment-oriented view of the same kind of tradeoff shown in the broader evaluation above.

![iNaturalist 2021 accuracy vs CUDA latency](img/performance-inat21.png)

Accuracy/latency tradeoff on the iNaturalist 2021 linear-probe evaluation. Accuracy is measured with a frozen image encoder and trained classification head. Latency is measured as CUDA milliseconds per sample with `torch.compile`, batch size 512 and AMP enabled. Bio-DINO checkpoints are shown at 224px, 252px and 336px, illustrating the tradeoff between input resolution, accuracy and inference cost.

The full interactive latency/quality chart is available in the [Birder Leaderboard Space](https://huggingface.co/spaces/birder-project/leaderboard).

## Relationship To BioCLIP And Other Work

BioCLIP, BioCLIP 2 and BioCLIP 2.5 are important reference points for this release.
They advanced biological image understanding with large-scale biodiversity-focused CLIP training, strong Tree of Life image-text datasets and open model releases.
They remain very strong baselines in the evaluations reported here, especially on fine-grained biological recognition tasks.

Bio-DINO is intended to be complementary.
It is an image-only, self-supervised encoder released for embedding extraction, image retrieval, transfer learning and continued DINO-style research.
It does not include a text encoder and should not be read as a replacement for BioCLIP-style image-text models.
The companion release of the full DINO training weights is intended to make the training state reusable for researchers who want to continue or adapt the self-supervised training process.
The full training state is available in the [full DINO training weights companion repository](https://huggingface.co/birder-project/dino_v2_vit_reg4_so150m_p14_ls_bio).

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
(net, model_info) = birder.load_pretrained_model("vit_reg4_so150m_p14_ls_dino-v2-bio-252px", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_reg4_so150m_p14_ls_dino-v2-bio-252px", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 896)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_reg4_so150m_p14_ls_dino-v2-bio-252px", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 896, 18, 18]))]
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

@misc{alabdulmohsin2024gettingvitshapescaling,
      title={Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design},
      author={Ibrahim Alabdulmohsin and Xiaohua Zhai and Alexander Kolesnikov and Lucas Beyer},
      year={2024},
      eprint={2305.13035},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2305.13035},
}

@misc{oquab2024dinov2learningrobustvisual,
      title={DINOv2: Learning Robust Visual Features without Supervision},
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2024},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.07193},
}
```

## Acknowledgements

This release builds on the broader open-source vision and biodiversity machine-learning ecosystem.
In particular, the BioCLIP, BioCLIP 2 and BioCLIP 2.5 work by the Imageomics team helped move biological image foundation models forward and provided strong open reference models for this release.

This work benefited from computational resources provided by The Stein Faculty of Computer and Information Science, Ben-Gurion University of the Negev, through a collaboration with the [STARdbi](https://stardbi.cs.bgu.ac.il/home/welcome) project led by Chen Keasar.
The released models and evaluations are intended to support biodiversity computer-vision research, including visual identification, classification, retrieval, and diversity analysis of insects and other organisms.
