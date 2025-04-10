# Reference Self-supervised Training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## SSL Pre-training

- [Barlow Twins](#barlow-twins)
- [BYOL](#byol)
- [CAPI](#capi)
- [DINO v1](#dino-v1)
- [I-JEPA](#i-jepa)
- [iBOT](#ibot)
- [VICReg](#vicreg)

### Barlow Twins

#### Barlow Twins: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_barlow_twins --network efficientnet_v2_s --opt lars --lr 0.2 --lr-scale 256 --lr-scheduler cosine --warmup-epochs 10 --batch-size 192 --epochs 800 --wd 0.000001 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### BYOL

Use `--sync-bn` when batch size is 32 or below.

#### BYOL: RegNet X 4 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_byol --network regnet_x_4g --opt lars --lr 0.2 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --epochs 600 --wd 0.0000015 --norm-wd 0 --bias-weight-decay 0 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_x_4g --tag byol --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

### CAPI

#### CAPI: Hiera AbsWin Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network hiera_abswin_small --kept-mask-ratio 0.15 --opt adamw --lr 0.001 --lr-scale 512 --opt-betas 0.9 0.95 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 512 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_small --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

#### CAPI: Hiera AbsWin Large

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network hiera_abswin_large --kept-mask-ratio 0.15 --opt adamw --lr 0.001 --lr-scale 512 --opt-betas 0.9 0.95 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 128 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --wds --wds-info data/ssl/_info.json
```

DGX A100 training

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_capi --network hiera_abswin_large --kept-mask-ratio 0.15 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 512 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --find-unused-parameters --keep-last 4 --wds --wds-info data/ssl/_info.json
```

#### CAPI: RoPE ViTReg4 s14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vitreg4_s14 --opt adamw --lr 0.001 --lr-scale 512 --opt-betas 0.9 0.95 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 256 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vitreg4_s14 --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --fast-matmul --compile --resume-epoch 0 --reset-head --freeze-body
```

#### CAPI: RoPE ViTReg4 m14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vitreg4_m14 --opt adamw --lr 0.001 --lr-scale 512 --opt-betas 0.9 0.95 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 256 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vitreg4_m14 --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --fast-matmul --compile --resume-epoch 0 --reset-head --freeze-body
```

#### CAPI: RoPE SoViT reg8 150m p14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vitreg8_so150m_p14_ap --opt adamw --lr 0.001 --lr-scale 512 --opt-betas 0.9 0.95 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 192 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vitreg8_so150m_p14_ap --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

### DINO v1

#### DINO v1: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network convnext_v2_tiny --use-bn-in-head --norm-last-layer --local-crops-number 6 --teacher-temp 0.07 --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 10 --batch-size 128 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### DINO v1: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network efficientnet_v2_s --use-bn-in-head --norm-last-layer --local-crops-number 6 --teacher-temp 0.07 --opt lars --lr 0.3 --lr-scheduler cosine --lr-cosine-min 0.001 --epochs 800 --warmup-epochs 10 --batch-size 128 --wd 0.000001 --norm-wd 0 --bias-weight-decay 0 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag dino-v1 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

#### DINO v1: RoPE DeiT3 Reg4 m14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network rope_deit3_reg4_m14 --norm-last-layer --local-crops-number 10 --local-crop-size 98 --teacher-temp 0.07 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 600 --warmup-epochs 10 --batch-size 80 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 0.5 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### DINO v1: XCiT small-12 p16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network xcit_small12_p16 --local-crops-number 10 --teacher-temp 0.07 --opt adamw --lr 0.00025 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 10 --batch-size 96 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network xcit_small12_p16 --tag dino-v1 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

### I-JEPA

#### I-JEPA: Simple ViT s14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network simple_vit_s14 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 40 --batch-size 192 --wd 0.04 --wd-end 0.4 --norm-wd 0 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### I-JEPA: ViTReg4 b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network vitreg4_b16 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 40 --batch-size 192 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_b16 --tag i-jepa --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body
```

### iBOT

#### iBOT: ConvNeXt v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network convnext_v2_small --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --freeze-last-layer-epochs 1 --epochs 800 --warmup-epochs 10 --batch-size 64 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 3 --amp --compile --data-path data/training
```

Large scale training

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network convnext_v2_small --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 20 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --freeze-last-layer-epochs 1 --epochs 80 --warmup-epochs 5 --batch-size 64 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 3 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### iBOT: RegNet Y 1.6 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network regnet_y_1_6g --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --sync-bn --freeze-last-layer-epochs 1 --epochs 800 --warmup-epochs 10 --batch-size 128 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 3 --amp --compile-teacher --data-path data/training
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_1_6g --tag ibot --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --fast-matmul --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_1_6g --tag ibot --opt adamw --lr 0.000125 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 60 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --compile --layer-decay 0.9 --resume-epoch 0
```

#### iBOT: RegNet Y 4 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network regnet_y_4g --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --sync-bn --freeze-last-layer-epochs 1 --epochs 800 --warmup-epochs 10 --batch-size 80 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 3 --amp --compile-teacher --data-path data/training
```

#### iBOT: Swin Transformer v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network swin_transformer_v2_t --shared-head --local-crops-number 10 --pred-start-epoch 50 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --freeze-last-layer-epochs 1 --epochs 300 --warmup-epochs 10 --batch-size 64 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 3 --amp --compile --data-path data/training
```

#### iBOT: ViT s16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network vit_s16 --shared-head --local-crops-number 10 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --freeze-last-layer-epochs 1 --epochs 800 --warmup-epochs 10 --batch-size 64 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 3 --amp --compile --data-path data/training
```

#### iBOT: ViT b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network vit_b16 --shared-head --norm-last-layer --local-crops-number 10 --teacher-temp 0.07 --warmup-teacher-temp-epochs 50 --opt adamw --lr 0.00075 --lr-scheduler cosine --lr-cosine-min 1e-6 --freeze-last-layer-epochs 3 --epochs 400 --warmup-epochs 10 --batch-size 48 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 0.3 --amp --compile --data-path data/training
```

Large scale training

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network vit_b16 --shared-head --norm-last-layer --local-crops-number 10 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --freeze-last-layer-epochs 3 --epochs 80 --warmup-epochs 5 --batch-size 48 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --clip-grad-norm 0.3 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### VICReg

Use `--sync-bn` when batch size is 32 or below.

#### VICReg: EfficientNet v2 Medium

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_vicreg --network efficientnet_v2_m --opt lars --lr 0.2 --lr-scale 256 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --epochs 400 --wd 0.000001 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag vicreg --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag vicreg --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 128 --epochs 200 --size 256 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.98 --resume-epoch 0
```
