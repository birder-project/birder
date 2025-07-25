# Reference Pre-training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## Image Pre-training

- [CrossMAE](#crossmae)
- [FCMAE](#fcmae)
- [MAE Hiera](#mae-hiera)
- [MAE ViT](#mae-vit)
- [SimMIM](#simmim)

### CrossMAE

#### CrossMAE: ViT reg4 b14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network crossmae --encoder vit_reg4_b14 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 512 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b14 --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body
```

#### CrossMAE: SoViT reg4 150m p14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network crossmae --encoder vit_reg4_so150m_p14_ap --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 384 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --rgb-mode none --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_so150m_p14_ap --tag mim-intermediate --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --rgb-mode none --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_so150m_p14_ap --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --rgb-mode none --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

### FCMAE

#### FCMAE: ConvNeXt v2 Atto

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder convnext_v2_atto --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 512 --epochs 400 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --fast-matmul --compile --find-unused-parameters --data-path data/training
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_atto --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.2 --aug-level 4 --fast-matmul --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_atto --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 600 --size 256 --wd 0.3 --smoothing-alpha 0.2 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --layer-decay 0.9 --resume-epoch 0
```

#### FCMAE: ConvNeXt v2 Nano

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder convnext_v2_nano --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 512 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### FCMAE: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder convnext_v2_tiny --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 256 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 15 --batch-size 256 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --amp --compile --compile-opt --layer-decay 0.9 --resume-epoch 10 --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: fine-tuning, second stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim-intermediate --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 384 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim-intermediate --opt adamw --lr 0.000025 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 64 --epochs 60 --size 448 --wd 1e-8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --compile-opt --layer-decay 0.9 --resume-epoch 10
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 320 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 128 --epochs 210 --size 320 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --compile-opt --layer-decay 0.9 --resume-epoch 10
```

At epoch 120 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 64 --epochs 210 --size 384 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --compile-opt --layer-decay 0.9 --resume-epoch 120 --load-states
```

At epoch 170 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 64 --epochs 210 --size 448 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --compile-opt --layer-decay 0.9 --resume-epoch 170 --load-states
```

#### FCMAE: ConvNeXt v2 Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder convnext_v2_base --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 128 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 15 --batch-size 128 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --amp --compile --compile-opt --layer-decay 0.8 --resume-epoch 10 --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: fine-tuning, second stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim-intermediate --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 384 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim-intermediate --opt adamw --lr 0.000025 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 32 --epochs 40 --size 448 --wd 1e-8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --compile-opt --layer-decay 0.8 --resume-epoch 10
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 320 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

#### FCMAE: RegNet Y 8 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder regnet_y_8g --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 128 --wd 0.05 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --lr 0.1 --lr-scheduler cosine --warmup-epochs 15 --batch-size 128 --size 256 --epochs 110 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --amp --compile --compile-opt --layer-decay 0.9 --resume-epoch 10 --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --lr 0.001 --lr-scheduler cosine --batch-size 128 --size 256 --epochs 30 --wd 0.0005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --amp --compile --compile-opt --layer-decay 0.8 --resume-epoch 10
```

### MAE Hiera

#### MAE Hiera: Hiera Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_small --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 512 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE Hiera: Hiera AbsWin Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_tiny --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 512 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_tiny --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_tiny --tag mim --opt adamw --lr 0.002 --lr-scale 1024 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 15 --epochs 310 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --amp --compile --compile-opt --layer-decay 0.65 --resume-epoch 10 --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_tiny --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

#### MAE Hiera: Hiera AbsWin Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_small --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 512 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE Hiera: Hiera AbsWin Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_base --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 512 --wd 0.05 --encoder-model-config drop_path_rate=0.2 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Frozen representation linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 50 --size 256 --smoothing-alpha 0.1 --aug-level 4 --amp --compile--resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim-intermediate --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim-intermediate --opt adamw --lr 0.002 --lr-scale 1024 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 192 --warmup-epochs 5 --epochs 100 --size 256 --wd 0.05 --norm-wd 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --amp --compile --compile-opt --layer-decay 0.7 --resume-epoch 0 --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim-intermediate --opt adamw --lr 0.0003 --lr-scale 1024 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 80 --epochs 30 --size 384 --wd 0.05 --norm-wd 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --amp --compile --compile-opt --layer-decay 0.7 --resume-epoch 0 --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 100 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --ra-sampler --ra-reps 2 --amp --compile --compile-opt --layer-decay 0.7 --resume-epoch 0
```

#### MAE Hiera: Hiera AbsWin Base Plus

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_base_plus --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 256 --wd 0.05 --encoder-model-config drop_path_rate=0.2 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### MAE ViT

#### MAE ViT: Simple ViT b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder simple_vit_b16 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 256 --wd 0.0001 --clip-grad-norm 1 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE ViT: ViT SAM b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder vit_sam_b16 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 128 --wd 0.0001 --clip-grad-norm 1 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE ViT: ViT reg4 b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder vit_reg4_b16 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 256 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim-intermediate --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim-intermediate --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 15 --epochs 110 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --clip-grad-norm 1 --amp --compile --compile-opt --layer-decay 0.65 --resume-epoch 10 --save-frequency 1 --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Optional intermediate training: increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim-intermediate --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 15 --epochs 110 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --clip-grad-norm 1 --amp --compile --compile-opt --layer-decay 0.75 --resume-epoch 80 --load-scheduler --save-frequency 1 --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 15 --epochs 110 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.65 --resume-epoch 10
```

At epoch 80 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim --opt adamw --lr 0.00004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --epochs 110 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --compile-opt --layer-decay 0.65 --resume-epoch 80
```

#### MAE ViT: ViT l16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder vit_l16 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 256 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_l16 --tag mim --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_l16 --tag mim --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

Number of epochs relays on the amount of samples in training (1M -> 50 epochs), try larger number for smaller datasets.

```sh
torchrun --nproc_per_node=2 train.py --network vit_l16 --tag mim --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 5e-7 --batch-size 64 --warmup-epochs 5 --epochs 50 --size 256 --wd 0.05 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --clip-grad-norm 1 --amp --compile --compile-opt --layer-decay 0.75 --resume-epoch 0
```

#### MAE ViT: SoViT reg4 150m p14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder vit_reg4_so150m_p14_ap --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 256 --wd 0.05 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_so150m_p14_ap --tag mim-intermediate --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 252 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file data/intermediate_packed/classes.txt --wds-info data/intermediate_packed/_info.json
```

### SimMIM

#### SimMIM: MaxViT Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder maxvit_t --opt adamw --lr 0.0001 --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_t --tag mim --opt adamw --lr 0.00005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --size 256 --epochs 10 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --clip-grad-norm 1 --amp --resume-epoch 0 --reset-head --freeze-body
```

#### SimMIM: NextViT Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder nextvit_s --opt adamw --lr 0.0001 --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: NextViT Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder nextvit_b --opt adamw --lr 0.0001 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --encoder-model-config drop_path_rate=0.0 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: RegNet Y 4 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder regnet_y_4g --opt adamw --lr 0.0005 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: Swin Transformer v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder swin_transformer_v2_s --opt adamw --lr 0.0001 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --encoder-model-config drop_path_rate=0.0 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --tag mim --opt adamw --lr 0.00005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --size 256 --epochs 10 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --clip-grad-norm 5 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --tag mim --opt adamw --lr 0.0006 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 256 --warmup-epochs 20 --epochs 110 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --compile-opt --layer-decay 0.9 --resume-epoch 10
```

#### SimMIM: Swin Transformer v2 Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder swin_transformer_v2_b --opt adamw --lr 0.0001 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --encoder-model-config drop_path_rate=0.0 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: Swin Transformer v2 w2 Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder swin_transformer_v2_w2_b --opt adamw --lr 0.0001 --lr-scheduler cosine --warmup-epochs 10 --batch-size 64 --wd 0.05 --clip-grad-norm 1 --encoder-model-config drop_path_rate=0.0 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_w2_b --tag mim --opt adamw --lr 0.00005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --size 256 --epochs 10 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --clip-grad-norm 5 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```
