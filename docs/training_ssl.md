# Reference Self-supervised Training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## SSL Pre-training

- [Barlow Twins](#barlow-twins)
- [BYOL](#byol)
- [CAPI](#capi)
- [Data2Vec](#data2vec)
- [DINO v1](#dino-v1)
- [DINO v2](#dino-v2)
- [DINO v2 Dist](#dino-v2-dist)
- [I-JEPA](#i-jepa)
- [iBOT](#ibot)
- [MMCR](#mmcr)
- [SimCLR](#simclr)
- [VICReg](#vicreg)

### Barlow Twins

#### Barlow Twins: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_barlow_twins --network efficientnet_v2_s --opt lars --lr 0.2 --bias-lr 0.0048 --lr-scale 256 --lr-scheduler-update iter --lr-scheduler cosine --warmup-epochs 10 --batch-size 192 --sync-bn --epochs 800 --wd 0.000001 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### Barlow Twins: RegNet Y 1.6 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_barlow_twins --network regnet_y_1_6g --projector-dims 4096 4096 4096 --opt lars --lr 0.2 --bias-lr 0.0048 --lr-scale 256 --lr-scheduler-update iter --lr-scheduler cosine --warmup-epochs 10 --batch-size 256 --sync-bn --epochs 800 --wd 0.000001 --amp --compile --data-path data/training
```

### BYOL

Use `--sync-bn` when batch size is 32 or below.

#### BYOL: RegNet X 4 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_byol --network regnet_x_4g --opt lars --lr 0.2 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --epochs 600 --wd 0.0000015 --norm-wd 0 --bias-weight-decay 0 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_x_4g --tag byol --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body
```

### CAPI

#### CAPI: Hiera AbsWin Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network hiera_abswin_small --mask-ratio 0.6 --kept-mask-ratio 0.2 --decoder-layers 6 --decoder-dim 768 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 512 --epochs 600 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_small --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

#### CAPI: Hiera AbsWin Base Plus

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network hiera_abswin_base_plus --mask-ratio 0.6 --kept-mask-ratio 0.2 --decoder-layers 8 --decoder-dim 768 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 448 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

#### CAPI: Hiera AbsWin Large

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network hiera_abswin_large --mask-ratio 0.6 --kept-mask-ratio 0.2 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 128 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

DGX A100 training

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_capi --network hiera_abswin_large --mask-ratio 0.6 --kept-mask-ratio 0.2 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 512 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --keep-last 4 --wds --wds-info data/ssl_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_large --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

ImageNet 1K: fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_large --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

#### CAPI: RoPE ViT reg4 m14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vit_reg4_m14 --decoder-layers 8 --decoder-dim 512 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 256 --epochs 600 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg4_m14 --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 252 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --compile --resume-epoch 0 --reset-head --freeze-body
```

#### CAPI: RoPE ViT reg8 b14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vit_reg8_b14_ap --decoder-layers 8 --decoder-dim 768 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 192 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

ImageNet 1K fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet1k --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-type ra --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

ImageNet 1K next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet1k --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 320 --warmup-epochs 5 --epochs 100 --size 196 --wd 0.05 --grad-accum-steps 2 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-type 3aug --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.65 --resume-epoch 0 --save-frequency 1 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

ImageNet 1K next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet1k --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 150 --size 224 --wd 0.02 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-type ra --ra-magnitude 15 --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.65 --resume-epoch 100 --save-frequency 1 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

ImageNet 21K fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet21k --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-type ra --amp --compile --save-frequency 1 --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file public_datasets_metadata/imagenet-21k-classes.txt --wds-info ~/Datasets/imagenet-w21-webp-wds/_info.json --wds-training-split train
```

ImageNet 21K next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet21k --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 50 --size 224 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-type ra --re-prob 0.25 --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.65 --resume-epoch 0 --save-frequency 1 --wds --wds-class-file public_datasets_metadata/imagenet-21k-classes.txt --wds-info ~/Datasets/imagenet-w21-webp-wds/_info.json --wds-training-split train
```

ImageNet 1K fine-tuning of ImageNet 21K (after linear probing)

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet21k-imagenet1k --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 50 --size 224 --wd 0.05 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-type ra --ra-magnitude 12 --re-prob 0.25 --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.6 --resume-epoch 0 --save-frequency 1 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

iNaturalist 2021 fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-inat21 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

iNaturalist 2021 next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-inat21 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 100 --size 224 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.65 --resume-epoch 0 --save-frequency 1 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

iNaturalist 2021 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-inat21 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 40 --size 336 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.65 --resume-epoch 0 --save-frequency 1 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

Places 365 fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-places365 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --data-path ~/Datasets/Places365/training --val-path ~/Datasets/Places365/validation
```

Places 365 next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-places365 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 40 --size 224 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --resize-min-scale 0.3 --use-grayscale --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.65 --resume-epoch 0 --save-frequency 1 --data-path ~/Datasets/Places365/training --val-path ~/Datasets/Places365/validation
```

#### CAPI: RoPE SoViT reg8 150m p14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vit_reg8_so150m_p14_ap --decoder-layers 10 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 192 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --compile-opt --rgb-mode none --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

DGX A100 training

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_capi --network rope_vit_reg8_so150m_p14_ap --decoder-layers 10 --opt adamw --lr 0.001 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 1024 --epochs 400 --wd 0.1 --norm-wd 0.01 --amp --amp-dtype bfloat16 --compile --rgb-mode none --find-unused-parameters --keep-last 4 --wds --wds-info data/ssl_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_so150m_p14_ap --tag capi --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 252 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --rgb-mode none --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

### Data2Vec

#### Data2Vec: ViT Parallel s16 18x2 LS AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec --network vit_parallel_s16_18x2_ls_avg --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 192 --epochs 400 --wd 0.05 --clip-grad-norm 3 --model-config drop_path_rate=0.25 --amp --amp-dtype bfloat16 --compile --compile-opt --rgb-mode none --wds --wds-info data/ssl_micro_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_parallel_s16_18x2_ls_avg --tag data2vec --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --rgb-mode none --resume-epoch 0 --reset-head --freeze-body
```

#### Data2Vec: ViT reg1 s16 LS

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec --network vit_reg1_s16_ls --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 256 --epochs 400 --wd 0.05 --clip-grad-norm 3 --model-config drop_path_rate=0.1 --amp --amp-dtype bfloat16 --compile --rgb-mode none --data-path data/training
```

#### Data2Vec: ViT reg4 b16 AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec --network vit_reg4_b16_avg --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 192 --epochs 600 --wd 0.05 --clip-grad-norm 3 --model-config drop_path_rate=0.25 --amp --amp-dtype bfloat16 --compile --compile-opt --rgb-mode none --data-path data/training data/raw_data data/detection_data/training ~/Datasets
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
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag dino-v1 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body
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
torchrun --nproc_per_node=2 train.py --network xcit_small12_p16 --tag dino-v1 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body
```

### DINO v2

#### DINO v2: ConvNeXt v2 Nano

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network convnext_v2_nano --dino-out-dim 32768 --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --batch-size 128 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --data-path data/training
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_nano --tag dino-v2 --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body
```

#### DINO v2: DaViT Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network davit_small --ibot-separate-head --dino-out-dim 49152 --ibot-out-dim 49152 --centering sinkhorn_knopp --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --batch-size 64 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_micro_packed/_info.json
```

#### DINO v2: DaViT Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network davit_base --ibot-separate-head --centering sinkhorn_knopp --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 400 --warmup-epochs 50 --batch-size 56 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_packed/_info.json
```

#### DINO v2: HieraDet Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network hieradet_base --ibot-separate-head --centering sinkhorn_knopp --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --batch-size 64 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hieradet_base --tag dino-v2 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body
```

#### DINO v2: Next-ViT Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network nextvit_b --ibot-separate-head --dino-out-dim 131072 --ibot-out-dim 131072 --head-bottleneck-dim 384 --centering sinkhorn_knopp --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --batch-size 64 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --rgb-mode none --wds --wds-info data/ssl_packed/_info.json
```

#### DINO v2: ViT reg1 s16 rms LS

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network vit_reg1_s16_rms_ls --dino-out-dim 32768 --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 10 --batch-size 96 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --rgb-mode none --wds --wds-info data/ssl_micro_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg1_s16_rms_ls --tag dino-v2 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 240 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --resume-epoch 0 --reset-head --freeze-body --rgb-mode none
```

Transform to FlexiViT

```sh
inv flexivit-from-vit vit_reg1_s16_rms_ls --tag dino-v2 --epoch 0
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network flexivit_reg1_s16_rms_ls --tag dino-v2 --opt adamw --lr 0.0004 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 192 --warmup-epochs 10 --epochs 100 --size 240 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --model-config min_patch_size=10,max_patch_size=40 --amp --amp-dtype bfloat16 --compile --rgb-mode none --layer-decay 0.7 --resume-epoch 0
```

#### DINO v2: ViT reg4 m16 rms AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network vit_reg4_m16_rms_avg --dino-out-dim 32768 --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --batch-size 96 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --data-path data/training
```

#### DINO v2: RoPE SoViT reg8 150m p14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network rope_vit_reg8_so150m_p14_ap --ibot-separate-head --dino-out-dim 131072 --ibot-out-dim 131072 --head-bottleneck-dim 384 --centering sinkhorn_knopp --local-crop-size 98 --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --batch-size 32 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --model-config drop_path_rate=0.3 --amp --amp-dtype bfloat16 --compile --rgb-mode none --wds --wds-info data/ssl_packed/_info.json
```

### DINO v2 Dist

#### DINO v2 Dist: ViT reg1 t16 with a ViT reg1 s16 rms LS teacher

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2_dist --network vit_reg1_t16 --teacher vit_reg1_s16_rms_ls --teacher-epoch 200 --dino-out-dim 32768 --opt adamw --lr 0.0002 --lr-scheduler-update iter --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 10 --batch-size 96 --wd 0.04 --wd-end 0.2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --rgb-mode none --data-path data/training_il-all_packed
```

### I-JEPA

#### I-JEPA: Simple ViT s14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network simple_vit_s14 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 40 --batch-size 192 --wd 0.04 --wd-end 0.4 --norm-wd 0 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### I-JEPA: Simple ViT b14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network simple_vit_b14 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 40 --batch-size 128 --wd 0.04 --wd-end 0.4 --norm-wd 0 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

#### I-JEPA: ViT reg4 m16 rms AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network vit_reg4_m16_rms_avg --predictor-depth 6 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 5 --batch-size 320 --wd 0.04 --wd-end 0.4 --norm-wd 0 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_m16_rms_avg --tag i-jepa --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --reset-head --freeze-body
```

ImageNet 1K fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_m16_rms_avg --tag i-jepa-imagenet1k --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-type ra --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --reset-head --freeze-body --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

ImageNet 21K fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_m16_rms_avg --tag i-jepa-imagenet21k --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-type ra --amp --compile --save-frequency 1 --resume-epoch 0 --reset-head --freeze-body --wds --wds-class-file public_datasets_metadata/imagenet-21k-classes.txt --wds-info ~/Datasets/imagenet-w21-webp-wds/_info.json --wds-training-split train
```

ImageNet 21K next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_m16_rms_avg --tag i-jepa-imagenet21k --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --warmup-epochs 5 --epochs 50 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-type ra --re-prob 0.25 --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.65 --resume-epoch 0 --save-frequency 1 --wds --wds-class-file public_datasets_metadata/imagenet-21k-classes.txt --wds-info ~/Datasets/imagenet-w21-webp-wds/_info.json --wds-training-split train
```

ImageNet 1K fine-tuning of ImageNet 21K (after linear probing)

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_m16_rms_avg --tag i-jepa-imagenet21k-imagenet1k --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --warmup-epochs 10 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 9 --use-grayscale --resize-min-scale 0.1 --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.6 --resume-epoch 0 --save-frequency 1 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

iNaturalist 2021 fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_m16_rms_avg --tag i-jepa-inat21 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 224 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --reset-head --freeze-body --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

iNaturalist 2021 next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_m16_rms_avg --tag i-jepa-inat21 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --warmup-epochs 5 --epochs 100 --size 224 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --compile-opt --layer-decay 0.65 --resume-epoch 0 --save-frequency 1 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

#### I-JEPA: ViT reg8 b14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network vit_reg8_b14_ap --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 40 --batch-size 192 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --keep-last 10 --wds --wds-info data/ssl_packed/_info.json
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
torchrun --nproc_per_node=2 train.py --network regnet_y_1_6g --tag ibot --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --fast-matmul --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_1_6g --tag ibot --opt adamw --lr 0.000125 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 60 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --amp --compile --layer-decay 0.9 --resume-epoch 0
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

### MMCR

#### MMCR: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mmcr --network efficientnet_v2_s --opt lars --lr 0.6 --lr-scale 256 --lr-scheduler-update iter --lr-scheduler cosine --warmup-epochs 10 --batch-size 192 --sync-bn --epochs 100 --wd 0.000001 --amp --compile --data-path data/training
```

#### MMCR: PVT v2 B1

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mmcr --network pvt_v2_b1 --opt adamw --lr 0.0005 --opt-betas 0.9 0.95 --lr-scheduler-update iter --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --epochs 100 --wd 0.000001 --amp --amp-dtype bfloat16 --compile --data-path data/training
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b1 --tag mmcr --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body
```

### SimCLR

#### SimCLR: Resnet v1 50

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_simclr --network resnet_v1_50 --opt lars --lr 0.075 --lr-scale 4096 --lr-scale-type sqrt --lr-scheduler cosine --warmup-epochs 10 --batch-size 256 --sync-bn --epochs 100 --wd 0.0001 --amp --compile --data-path data/training
```

### VICReg

#### VICReg: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_vicreg --network convnext_v2_tiny --mlp-dim 4096 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 6 --batch-size 192 --epochs 60 --wd 0.000001 --amp --compile --wds --wds-info data/ssl_packed/_info.json
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag vicreg --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body
```

#### VICReg: EfficientNet v2 Medium

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_vicreg --network efficientnet_v2_m --opt lars --lr 0.2 --lr-scale 256 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --epochs 400 --wd 0.000001 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag vicreg --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag vicreg --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 128 --epochs 200 --size 256 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.9 --resume-epoch 0
```
