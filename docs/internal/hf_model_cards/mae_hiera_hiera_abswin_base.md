---
tags:
- masked-image-modeling
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for mae_hiera_hiera_abswin_base

This is a Masked Autoencoder (MAE) model based on the Hiera (Hierarchical Vision Transformer) architecture.

**Important:** This model is specifically designed for continued masked image modeling (MIM) pre-training only. If you're looking to fine-tune on downstream tasks, please use the pre-trained encoder directly at <https://huggingface.co/birder-project/hiera_abswin_base_mim>

## Model Details

- **Model Type:** Masked image modeling
- **Model Stats:**
    - Params (M): 85.9
    - Input image size: 224 x 224
- **Dataset:** Trained on a diverse dataset of approximately 12M images, including:
    - iNaturalist 2021 (~2.6M)
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

The Birder CLI provides the easiest way to continue MIM training with this model. However, if you need to integrate the model into your own training scripts, you can use the Python API to load the checkpoint as shown below.

### Continue Training with Birder CLI

First download the model weights:

```sh
python -m birder.tools download-model mae_hiera_hiera_abswin_base
```

Then run training:

```sh
python -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_base --pretrained --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 512 --wd 0.05 --encoder-model-config drop_path_rate=0.2 --amp --compile --compile-opt --find-unused-parameters --data-path data/training
```

### Python API

```python
import torch
from birder.common import fs_ops

device = torch.device("cuda")
(net, training_states) = fs_ops.load_mim_checkpoint(
    device,
    "mae_hiera",
    encoder="hiera_abswin_base",
    epoch=None,
)
```

The Python code snippet above loads the model architecture but does not download weights - use the Birder CLI download command to get the actual trained parameters

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
