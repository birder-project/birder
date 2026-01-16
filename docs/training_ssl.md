# Reference Self-supervised Training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## SSL Pre-training

- [Barlow Twins](#barlow-twins)
- [BYOL](#byol)
- [CAPI](#capi)
- [Data2Vec](#data2vec)
- [Data2Vec2](#data2vec2)
- [DINO v1](#dino-v1)
- [DINO v2](#dino-v2)
- [DINO v2 Dist](#dino-v2-dist)
- [Franca](#franca)
- [I-JEPA](#i-jepa)
- [iBOT](#ibot)
- [MMCR](#mmcr)
- [RotNet](#rotnet)
- [SimCLR](#simclr)
- [VICReg](#vicreg)

### Barlow Twins

#### Barlow Twins: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_barlow_twins --network efficientnet_v2_s --batch-size 192 --opt lars --lr 0.2 --bias-lr 0.0048 --lr-scale 256 --wd 0.000001 --lr-scheduler-update step --lr-scheduler cosine --epochs 800 --warmup-epochs 10 --sync-bn --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### Barlow Twins: RegNet Y 1.6 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_barlow_twins --network regnet_y_1_6g --projector-dims 4096 4096 4096 --batch-size 256 --opt lars --lr 0.2 --bias-lr 0.0048 --lr-scale 256 --wd 0.000001 --lr-scheduler-update step --lr-scheduler cosine --epochs 800 --warmup-epochs 10 --sync-bn --amp --compile --data-path data/training
```

### BYOL

Use `--sync-bn` when batch size is 32 or below.

#### BYOL: RegNet X 4 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_byol --network regnet_x_4g --batch-size 128 --opt lars --lr 0.2 --wd 0.0000015 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --epochs 600 --warmup-epochs 10 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### CAPI

#### CAPI: Hiera AbsWin Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network hiera_abswin_small --decoder-layers 6 --decoder-dim 768 --mask-ratio 0.6 --kept-mask-ratio 0.2 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --grad-accum-steps 16 --lr 0.001 --wd 0.1 --norm-wd 0.01 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 600 --warmup-epochs 40 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data
```

#### CAPI: Hiera AbsWin Base Plus

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network hiera_abswin_base_plus --decoder-layers 8 --decoder-dim 768 --mask-ratio 0.6 --kept-mask-ratio 0.2 --batch-size 448 --opt adamw --opt-betas 0.9 0.95 --grad-accum-steps 16 --lr 0.001 --wd 0.1 --norm-wd 0.01 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 40 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

#### CAPI: Hiera AbsWin Large

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network hiera_abswin_large --mask-ratio 0.6 --kept-mask-ratio 0.2 --batch-size 128 --opt adamw --opt-betas 0.9 0.95 --grad-accum-steps 64 --lr 0.001 --wd 0.1 --norm-wd 0.01 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 40 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

DGX A100 training

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_capi --network hiera_abswin_large --mask-ratio 0.6 --kept-mask-ratio 0.2 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --grad-accum-steps 4 --lr 0.001 --wd 0.1 --norm-wd 0.01 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 40 --amp --amp-dtype bfloat16 --compile --keep-last 4 --wds --wds-info data/ssl_packed/_info.json
```

#### CAPI: RoPE ViT reg4 m14 AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vit_reg4_m14_avg --decoder-layers 6 --decoder-dim 512 --momentum-teacher 0.998 --sinkhorn-queue-size 256 --batch-size 256 --opt adamw --opt-fused --opt-betas 0.9 0.95 --grad-accum-steps 4 --lr 0.0015 --wd 0.1 --norm-wd 0.01 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 600 --warmup-epochs 60 --amp --amp-dtype bfloat16 --compile --data-path data/training data/raw_data
```

#### CAPI: RoPE ViT reg8 b14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vit_reg8_b14_ap --decoder-layers 8 --decoder-dim 768 --batch-size 192 --opt adamw --opt-betas 0.9 0.95 --grad-accum-steps 32 --lr 0.001 --wd 0.1 --norm-wd 0.01 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 40 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-intermediate --batch-size 192 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 5 --model-ema --size 252 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

ImageNet 1K, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet1k --batch-size 320 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 5 --size 196 --aug-type 3aug --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

Cont. ImageNet 1K, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet1k --batch-size 256 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0001 --wd 0.02 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 150 --warmup-epochs 5 --model-ema --size 224 --aug-type ra --ra-magnitude 15 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 100 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

ImageNet 21K, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet21k --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 50 --warmup-epochs 5 --model-ema --size 224 --aug-type ra --re-prob 0.25 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --wds --wds-info ~/Datasets/imagenet-w21-webp-wds/_info.json --wds-class-file public_datasets_metadata/imagenet-21k-classes.txt --wds-training-split train
```

ImageNet 1K fine-tuning of ImageNet 21K (after linear probing)

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet21k-imagenet1k --batch-size 256 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0001 --wd 0.05 --layer-decay 0.6 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 50 --warmup-epochs 5 --model-ema --size 224 --aug-type ra --ra-magnitude 12 --re-prob 0.25 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

iNaturalist 2021, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-inat21 --batch-size 256 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 5 --model-ema --size 224 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

iNaturalist 2021 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-inat21 --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 40 --warmup-epochs 5 --model-ema --size 336 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

Places 365, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-places365 --batch-size 256 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 40 --warmup-epochs 5 --model-ema --size 224 --aug-level 8 --use-grayscale --resize-min-scale 0.3 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/Places365/training --val-path ~/Datasets/Places365/validation
```

#### CAPI: RoPE SoViT reg8 150m p14 swiglu rms AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_capi --network rope_vit_reg8_so150m_p14_swiglu_rms_avg --decoder-layers 10 --decoder-dim 896 --batch-size 192 --opt adamw --opt-fused --opt-betas 0.9 0.95 --grad-accum-steps 32 --lr 0.001 --wd 0.1 --norm-wd 0.01 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 40 --rgb-mode none --amp --amp-dtype bfloat16 --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

DGX A100 training

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_capi --network rope_vit_reg8_so150m_p14_swiglu_rms_avg --decoder-layers 10 --batch-size 1024 --opt adamw --opt-betas 0.9 0.95 --grad-accum-steps 2 --lr 0.001 --wd 0.1 --norm-wd 0.01 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 40 --rgb-mode none --amp --amp-dtype bfloat16 --compile --keep-last 4 --wds --wds-info data/ssl_packed/_info.json
```

Optional: Train attention pooling head using DINO

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network rope_vit_reg8_so150m_p14_swiglu_rms_aps --tag capi --out-dim 131072 --teacher-temp 0.07 --local-crops-number 8 --local-crop-size 98 --backbone-epoch 0 --freeze-body --batch-size 512 --opt adamw --clip-grad-norm 0.5 --grad-accum-steps 4 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --rgb-mode none --amp --amp-dtype bfloat16 --compile --non-strict-weights --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional: Linear probing (ImageNet 1K)

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_so150m_p14_swiglu_rms_aps --tag dino-v1-capi-imagenet1k --reset-head --freeze-body --batch-size 512 --lr 0.1 --wd 0.0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 10 --size 224 --aug-level 1 --rgb-mode none --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

Use as backbone for RotNet

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_rotnet --network rope_vit_reg8_so150m_p14_swiglu_rms_ap --rotation-prob 0.5 --tag capi --freeze-body --unfreeze-features --batch-size 256 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 10 --size 224 --aug-level 4 --rgb-mode none --amp --compile --resume-epoch 0 --non-strict-weights --wds --wds-info data/ssl_micro_packed/_info.json
```

### Data2Vec

#### Data2Vec: ViT Parallel s16 18x2 LS AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec --network vit_parallel_s16_18x2_ls_avg --model-config drop_path_rate=0.25 --batch-size 192 --opt adamw --clip-grad-norm 3 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 20 --rgb-mode none --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_micro_packed/_info.json
```

Intermediate training training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit_parallel_s16_18x2_ls_avg --tag data2vec-intermediate --batch-size 192 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.75 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile --compile-opt --save-frequency 1 --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: full fine-tuning with layer-wise learning rate decay, increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit_parallel_s16_18x2_ls_avg --tag data2vec-intermediate --batch-size 96 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0001 --wd 0.05 --norm-wd 0 --layer-decay 0.75 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 50 --warmup-epochs 5 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile --compile-opt --save-frequency 1 --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: specific fine-tuning with layer-wise learning rate decay, increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit_parallel_s16_18x2_ls_avg --tag data2vec-intermediate-il-all --batch-size 96 --opt adamw --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.000075 --wd 0.05 --norm-wd 0 --layer-decay 0.5 --layer-decay-no-opt-scale 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 50 --warmup-epochs 5 --model-ema --size 384 --aug-level 9 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile --compile-opt --save-frequency 1 --resume-epoch 0 --data-path data/training_il-all_packed --val-path data/validation_il-all_packed
```

#### Data2Vec: ViT reg1 s16 LS

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec --network vit_reg1_s16_ls --model-config drop_path_rate=0.1 --batch-size 256 --opt adamw --clip-grad-norm 3 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 10 --rgb-mode none --amp --amp-dtype bfloat16 --compile --data-path data/training
```

#### Data2Vec: ViT reg4 b16 AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec --network vit_reg4_b16_avg --model-config drop_path_rate=0.25 --batch-size 192 --opt adamw --clip-grad-norm 3 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 600 --warmup-epochs 20 --rgb-mode none --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### Data2Vec2

#### Data2Vec2: ViT reg4 m16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec2 --network vit_reg4_m16 --batch-size 128 --opt adamw --opt-betas 0.9 0.95 --clip-grad-norm 4 --lr 0.0005 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --rgb-mode none --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training
```

#### Data2Vec2: ViT b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec2 --network vit_b16 --batch-size 96 --opt adamw --opt-betas 0.9 0.95 --clip-grad-norm 4 --grad-accum-steps 16 --lr 0.0005 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --rgb-mode none --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training
```

#### Data2Vec2: SoViT reg8 150m p14 swiglu

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec2 --network vit_reg8_so150m_p14_swiglu --average-layers 12 --decoder-layers 3 --decoder-kernel-size 5 --decoder-dim 896 --batch-size 32 --opt adamw --opt-betas 0.9 0.95 --clip-grad-norm 4 --lr 0.0004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --rgb-mode none --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

#### Data2Vec2: ViT Parallel s16 18x2 LS

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_data2vec2 --network vit_parallel_s16_18x2_ls --average-layers 12 --batch-size 96 --opt adamw --opt-betas 0.9 0.95 --clip-grad-norm 4 --lr 0.0005 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --rgb-mode none --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_micro_packed/_info.json
```

### DINO v1

#### DINO v1: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network convnext_v2_tiny --use-bn-in-head --norm-last-layer --teacher-temp 0.07 --local-crops-number 6 --batch-size 128 --opt adamw --lr 0.0008 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 10 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### DINO v1: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network efficientnet_v2_s --use-bn-in-head --norm-last-layer --teacher-temp 0.07 --local-crops-number 6 --batch-size 128 --opt lars --lr 0.3 --wd 0.000001 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 0.001 --epochs 800 --warmup-epochs 10 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### DINO v1: RoPE DeiT3 Reg4 m14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network rope_deit3_reg4_m14 --norm-last-layer --teacher-temp 0.07 --local-crops-number 10 --local-crop-size 98 --batch-size 80 --opt adamw --clip-grad-norm 0.5 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 600 --warmup-epochs 10 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### DINO v1: XCiT small-12 p16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network xcit_small12_p16 --teacher-temp 0.07 --local-crops-number 10 --batch-size 96 --opt adamw --lr 0.00025 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 10 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### DINO v2

#### DINO v2: ConvNeXt v1 Nano

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network convnext_v1_nano --dino-out-dim 32768 --batch-size 128 --opt adamw --clip-grad-norm 3 --grad-accum-steps 4 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --data-path data/training
```

#### DINO v2: DaViT Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network davit_small --dino-out-dim 49152 --ibot-separate-head --ibot-out-dim 49152 --centering sinkhorn_knopp --batch-size 64 --opt adamw --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_micro_packed/_info.json
```

#### DINO v2: DaViT Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network davit_base --ibot-separate-head --centering sinkhorn_knopp --batch-size 56 --opt adamw --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 400 --warmup-epochs 50 --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_packed/_info.json
```

#### DINO v2: HieraDet Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network hieradet_d_small --ibot-separate-head --centering sinkhorn_knopp --batch-size 64 --opt adamw --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_packed/_info.json
```

ImageNet 12K, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hieradet_d_small --tag dino-v2-imagenet12k --batch-size 256 --opt adamw --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 5e-7 --epochs 100 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --resize-min-scale 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --wds --wds-info ~/Datasets/imagenet-12k-wds/_info.json --wds-class-file public_datasets_metadata/imagenet-12k-classes.txt --wds-training-split train
```

ImageNet 1K fine-tuning of ImageNet 12K (after linear probing)

```sh
torchrun --nproc_per_node=2 train.py --network hieradet_d_small --tag dino-v2-imagenet12k-imagenet1k --batch-size 256 --opt adamw --grad-accum-steps 4 --lr 7.5e-5 --wd 0.05 --norm-wd 0 --layer-decay 0.6 --lr-scheduler cosine --lr-cosine-min 5e-7 --epochs 50 --warmup-epochs 10 --model-ema --model-ema-steps 1 --model-ema-decay 0.9998 --model-ema-warmup 0 --size 256 --aug-level 9 --resize-min-scale 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

iNaturalist 2021 next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hieradet_d_small --tag dino-v2-inat21 --batch-size 256 --opt adamw --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

iNaturalist 2021 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network hieradet_d_small --tag dino-v2-inat21 --batch-size 128 --opt adamw --grad-accum-steps 2 --lr 0.0001 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 50 --warmup-epochs 5 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile --compile-opt --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

#### DINO v2: HieraDet Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network hieradet_base --ibot-separate-head --centering sinkhorn_knopp --batch-size 64 --opt adamw --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_packed/_info.json
```

#### DINO v2: Next-ViT Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network nextvit_b --dino-out-dim 131072 --head-bottleneck-dim 384 --ibot-separate-head --ibot-out-dim 131072 --centering sinkhorn_knopp --batch-size 64 --opt adamw --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --rgb-mode none --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_packed/_info.json
```

#### DINO v2: ViT s16 LS

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network vit_s16_ls --dino-out-dim 49152 --warmup-teacher-temp-epochs 10 --centering sinkhorn_knopp --batch-size 96 --opt adamw --clip-grad-norm 3 --grad-accum-steps 16 --lr 0.0002 --lr-scale 1024 --lr-scale-type sqrt --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 15 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/intermediate_packed/_info.json
```

#### DINO v2: ViT reg1 s16 rms LS

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network vit_reg1_s16_rms_ls --dino-out-dim 32768 --batch-size 96 --opt adamw --clip-grad-norm 3 --grad-accum-steps 4 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 10 --rgb-mode none --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_micro_packed/_info.json
```

Transform to FlexiViT (after linear probing)

```sh
inv flexivit-from-vit vit_reg1_s16_rms_ls --tag dino-v2 --epoch 0
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network flexivit_reg1_s16_rms_ls --tag dino-v2 --model-config min_patch_size=10,max_patch_size=40 --batch-size 192 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0004 --wd 0.05 --norm-wd 0 --layer-decay 0.7 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 10 --model-ema --size 240 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile --resume-epoch 0
```

#### DINO v2: ViT reg4 m16 rms AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network vit_reg4_m16_rms_avg --dino-out-dim 32768 --batch-size 96 --opt adamw --clip-grad-norm 3 --grad-accum-steps 4 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --data-path data/training
```

#### DINO v2: SoViT reg4 150m p14 LS

BIO, DGX A100 training

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_dino_v2 --network vit_reg4_so150m_p14_ls --tag bio --dino-out-dim 98304 --head-bottleneck-dim 320 --ibot-separate-head --ibot-out-dim 98304 --local-crop-size 98 --batch-size 32 --opt adamw --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0007 --lr-scale 1024 --lr-scale-type sqrt --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --rgb-mode none --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_bio_packed/_info.json
```

(4000x8x8x32 = 8.2M epoch, 8.2Mx250 = 2B)

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_dino_v2 --network vit_reg4_so150m_p14_ls --tag bio --dino-out-dim 98304 --head-bottleneck-dim 320 --ibot-separate-head --ibot-out-dim 98304 --warmup-teacher-temp-epochs 20 --local-crop-size 98 --batch-size 32 --opt adamw --opt-fused --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0002 --lr-scale 1024 --lr-scale-type sqrt --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 250 --steps-per-epoch 4000 --warmup-epochs 40 --rgb-mode none --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_bio_packed/_info.json
```

#### DINO v2: RoPE SoViT reg8 150m p14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2 --network rope_vit_reg8_so150m_p14_ap --model-config drop_path_rate=0.3 --dino-out-dim 131072 --head-bottleneck-dim 384 --ibot-separate-head --ibot-out-dim 131072 --local-crop-size 98 --centering sinkhorn_knopp --sinkhorn-queue-size 768 --batch-size 32 --opt adamw --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0002 --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 10 --rgb-mode none --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_packed/_info.json
```

### DINO v2 Dist

#### DINO v2 Dist: ViT reg1 t16 with a ViT reg1 s16 rms LS teacher

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v2_dist --network vit_reg1_t16 --teacher vit_reg1_s16_rms_ls --teacher-epoch 200 --dino-out-dim 32768 --opt adamw --lr 0.0002 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 10 --batch-size 96 --wd 0.04 --wd-end 0.2 --grad-accum-steps 2 --clip-grad-norm 3 --amp --amp-dtype bfloat16 --compile --rgb-mode none --data-path data/training_il-all_packed
```

### Franca

#### Franca: ViT s16 LS

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_franca --network vit_s16_ls --dino-out-dim 49152 --ibot-separate-head --ibot-out-dim 49152 --momentum-teacher 0.994 --warmup-teacher-temp-epochs 15 --batch-size 64 --opt adamw --clip-grad-norm 3 --grad-accum-steps 16 --lr 0.00075 --lr-scale 1024 --lr-scale-type sqrt --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --data-path data/training
```

#### Franca: ViT b16 LS

DGX A100 training

```sh
torchrun --nproc_per_node=8 -m birder.scripts.train_franca --network vit_b16_ls --dino-out-dim 81920 --head-bottleneck-dim 320 --ibot-separate-head --ibot-out-dim 81920 --batch-size 32 --opt adamw --clip-grad-norm 3 --grad-accum-steps 8 --lr 0.0007 --lr-scale 1024 --lr-scale-type sqrt --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 64 --rgb-mode none --amp --amp-dtype bfloat16 --compile --wds --wds-info data/ssl_packed/_info.json
```

BIOSCAN-5M

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_franca --network vit_b16_ls --tag bioscan5m --dino-out-dim 65536 --head-bottleneck-dim 320 --ibot-separate-head --ibot-out-dim 65536 --nesting-levels 4 --sinkhorn-queue-size 1280 --batch-size 32 --opt adamw --opt-fused --clip-grad-norm 3 --grad-accum-steps 16 --lr 0.0007 --lr-scale 1024 --lr-scale-type sqrt --wd 0.04 --wd-end 0.2 --lr-scheduler-update step --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --steps-per-epoch 2000 --warmup-epochs 32 --rgb-mode none --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/BIOSCAN-5M/pretrain
```

### I-JEPA

#### I-JEPA: Simple ViT s14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network simple_vit_s14 --batch-size 192 --opt adamw --lr 0.001 --wd 0.04 --wd-end 0.4 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 40 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### I-JEPA: Simple ViT b14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network simple_vit_b14 --batch-size 128 --opt adamw --lr 0.001 --wd 0.04 --wd-end 0.4 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 40 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

#### I-JEPA: ViT reg4 m16 rms AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network vit_reg4_m16_rms_avg --predictor-depth 6 --batch-size 320 --opt adamw --lr 0.001 --wd 0.04 --wd-end 0.4 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 5 --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

#### I-JEPA: ViT reg8 b14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_i_jepa --network vit_reg8_b14_ap --batch-size 192 --opt adamw --lr 0.001 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 40 --amp --amp-dtype bfloat16 --compile --compile-opt --keep-last 10 --find-unused-parameters --wds --wds-info data/ssl_packed/_info.json
```

### iBOT

#### iBOT: ConvNeXt v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network convnext_v2_small --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --freeze-last-layer-epochs 1 --batch-size 64 --opt adamw --clip-grad-norm 3 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 10 --amp --compile --data-path data/training
```

Large scale training

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network convnext_v2_small --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 20 --freeze-last-layer-epochs 1 --batch-size 64 --opt adamw --clip-grad-norm 3 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 80 --warmup-epochs 5 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### iBOT: RDNet Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network rdnet_t --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --freeze-last-layer-epochs 1 --tag bioscan5m --batch-size 96 --opt adamw --clip-grad-norm 3 --grad-accum-steps 16 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 10 --size 192 --amp --compile-teacher --data-path ~/Datasets/BIOSCAN-5M/pretrain
```

#### iBOT: RegNet Y 1.6 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network regnet_y_1_6g --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --freeze-last-layer-epochs 1 --batch-size 128 --opt adamw --clip-grad-norm 3 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 10 --sync-bn --amp --compile-teacher --data-path data/training
```

Full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_1_6g --tag ibot --batch-size 256 --opt adamw --lr 0.000125 --wd 0.05 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 60 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile --resume-epoch 0
```

#### iBOT: RegNet Y 4 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network regnet_y_4g --shared-head --local-crops-number 8 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --freeze-last-layer-epochs 1 --batch-size 80 --opt adamw --clip-grad-norm 3 --grad-accum-steps 16 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 10 --sync-bn --amp --compile-teacher --data-path data/training
```

#### iBOT: Swin Transformer v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network swin_transformer_v2_t --shared-head --local-crops-number 10 --pred-start-epoch 50 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --freeze-last-layer-epochs 1 --batch-size 64 --opt adamw --clip-grad-norm 3 --grad-accum-steps 16 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 10 --amp --compile --data-path data/training
```

#### iBOT: ViT s16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network vit_s16 --shared-head --local-crops-number 10 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --freeze-last-layer-epochs 1 --batch-size 64 --opt adamw --clip-grad-norm 3 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 10 --amp --compile --data-path data/training
```

#### iBOT: ViT b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network vit_b16 --shared-head --norm-last-layer --local-crops-number 10 --teacher-temp 0.07 --warmup-teacher-temp-epochs 50 --freeze-last-layer-epochs 3 --batch-size 48 --opt adamw --clip-grad-norm 0.3 --lr 0.00075 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 400 --warmup-epochs 10 --amp --compile --data-path data/training
```

Large scale training

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_ibot --network vit_b16 --shared-head --norm-last-layer --local-crops-number 10 --teacher-temp 0.07 --warmup-teacher-temp-epochs 30 --freeze-last-layer-epochs 3 --batch-size 48 --opt adamw --clip-grad-norm 0.3 --lr 0.0005 --wd 0.04 --wd-end 0.4 --norm-wd 0 --bias-weight-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 80 --warmup-epochs 5 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### MMCR

#### MMCR: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mmcr --network efficientnet_v2_s --batch-size 192 --opt lars --lr 0.6 --lr-scale 256 --wd 0.000001 --lr-scheduler-update step --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --sync-bn --amp --compile --data-path data/training
```

#### MMCR: PVT v2 B1

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mmcr --network pvt_v2_b1 --batch-size 128 --opt adamw --opt-betas 0.9 0.95 --lr 0.0005 --wd 0.000001 --lr-scheduler-update step --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --data-path data/training
```

### RotNet

#### RotNet: RegNet Y 6.4 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_rotnet --network regnet_y_6_4g --rotation-prob 0.75 --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 200 --warmup-epochs 5 --size 288 --aug-level 7 --amp --compile --wds --wds-info data/ssl_packed/_info.json
```

After 100 epochs, reduce the rotation probability

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_rotnet --network regnet_y_6_4g --rotation-prob 0.65 --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 200 --warmup-epochs 5 --size 288 --aug-level 7 --amp --compile --resume-epoch 100 --load-states --wds --wds-info data/ssl_packed/_info.json
```

### SimCLR

#### SimCLR: Resnet v1 50

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_simclr --network resnet_v1_50 --batch-size 256 --opt lars --lr 0.075 --lr-scale 4096 --lr-scale-type sqrt --wd 0.0001 --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --sync-bn --amp --compile --data-path data/training
```

### VICReg

#### VICReg: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_vicreg --network convnext_v2_tiny --mlp-dim 4096 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 6 --batch-size 192 --epochs 60 --wd 0.000001 --amp --compile --wds --wds-info data/ssl_packed/_info.json
```

#### VICReg: EfficientNet v2 Medium

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_vicreg --network efficientnet_v2_m --batch-size 128 --opt lars --lr 0.2 --lr-scale 256 --wd 0.000001 --lr-scheduler cosine --epochs 400 --warmup-epochs 10 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```
