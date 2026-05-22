---
tags:
- self-supervised learning
- birder
- pytorch
- biology
library_name: birder
license: apache-2.0
---

# Model Card for dino_v2_vit_reg4_so150m_p14_ls_bio

This repository contains the full Bio-DINO DINOv2 training weights for a SoViT-150M/14 Vision Transformer trained on natural photographs of living organisms.
It is the companion release to the Birder backbone checkpoints at <https://huggingface.co/birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio>.

**Important:** this repository is intended for continued DINOv2 self-supervised training, research and inspection of the training state.
If you want to extract embeddings, run retrieval, train probes, or initialize downstream models, use the backbone checkpoints instead.
These checkpoints include the DINO/iBOT training heads and related training state, not only the exported image encoder.

## Model Details

- **Model Type:** DINOv2 self-supervised training checkpoint
- **Backbone:** SoViT-150M/14 Vision Transformer with 4 register tokens
- **Backbone Params (M):** 133.6
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

## Released Checkpoints

| Checkpoint     | Resolution | Training stage                            |
| -------------- | ---------: | ----------------------------------------- |
| Bio-DINO 224px | 224 x 224  | Initial and longest DINOv2 training stage |
| Bio-DINO 252px | 252 x 252  | Continued training from the 224px stage   |

The 252px checkpoint is the recommended starting point for most continued-training experiments.
The 224px checkpoint is useful when continuing from the lower-cost training stage.
The 336px DINO training weights are not available.
The 336px model is available only as an exported backbone checkpoint in the companion Bio-DINO encoder repository.

## Intended Use

This release is intended for:

- Continued DINOv2-style self-supervised training.
- Research on biological image representation learning.
- Inspecting or adapting the DINO/iBOT heads and training state.

This release is not the recommended path for normal inference or embedding extraction.
For those workflows, use the companion Bio-DINO backbone checkpoints.

## Usage

Use the Birder DINOv2 training script to continue training from these checkpoints.
The exact command depends on the target dataset, resolution, batch size and cluster setup.

### Example Run

First download the model weights:

```sh
python -m birder.tools download-model dino_v2_vit_reg4_so150m_p14_ls_bio-252px
```

Then run training:

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_dino_v2 --network vit_reg4_so150m_p14_ls --tag bio-252px --dino-out-dim 98304 --head-bottleneck-dim 320 --ibot-separate-head --ibot-out-dim 98304 --momentum-teacher 0.998 --warmup-teacher-temp-epochs 15 --freeze-last-layer-epochs 0 --local-crop-size 112 --batch-size 64 --opt adamw --opt-fused --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0001 --lr-scale 1024 --lr-scale-type sqrt --wd 0.1 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --steps-per-epoch 5000 --size 252 --rgb-mode centered --fast-matmul --compile --resume-epoch 0 --distributed-mode fsdp --fsdp-sharding-strategy shard-grad-op --fsdp-param-dtype bfloat16 --fsdp-reduce-dtype float32 --no-broadcast-buffers --data-path data/some_training_data
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
