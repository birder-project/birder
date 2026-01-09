# Compact Networks Training Procedure

Examples use repo-root script names (e.g., `train.py`). If you installed Birder as a package, use the module form such as `python -m birder.scripts.train`.

## Compact Madness

### CaiT

#### CaiT: Extra Extra Small 24

```sh
torchrun --nproc_per_node=2 train.py --network cait_xxs24 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### CAS-ViT

#### CAS-ViT: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network cas_vit_xs --tag il-common --opt adamw --lr 0.006 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 384 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### CoaT

#### CoaT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network coat_tiny --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --clip-grad-norm 1 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### CoaT: Lite Tiny

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_tiny --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --clip-grad-norm 1 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Conv2Former

#### Conv2Former: Nano

```sh
torchrun --nproc_per_node=2 train.py --network conv2former_n --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### ConvNeXt v2

#### ConvNeXt v2: Atto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_atto --tag il-common --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 600 --size 256 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.2 --aug-level 8 --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### ConvNeXt v2: Femto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_femto --tag il-common --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 600 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.2 --mixup-alpha 0.3 --cutmix --aug-level 8 --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### CrossViT

#### CrossViT: 9 Dagger

```sh
torchrun --nproc_per_node=2 train.py --network crossvit_9d --tag il-common --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 30 --epochs 300 --size 240 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --clip-grad-norm 1 --amp --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### DeiT

#### DeiT: t16

```sh
torchrun --nproc_per_node=2 train.py --network deit_t16 --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### DeiT3

#### DeiT3: t16

```sh
torchrun --nproc_per_node=2 train.py --network deit3_t16 --tag il-common --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 512 --warmup-epochs 5 --epochs 600 --size 256 --wd 0.05 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --bce-loss --bce-threshold 0.05 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### DeiT3: reg4 t16

```sh
torchrun --nproc_per_node=2 train.py --network deit3_reg4_t16 --tag il-common --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 512 --warmup-epochs 5 --epochs 600 --size 256 --wd 0.05 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --bce-loss --bce-threshold 0.05 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EdgeNeXt

#### EdgeNeXt: Extra Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_xxs --tag il-common --opt adamw --lr 0.006 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### EdgeNeXt: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_xs --tag il-common --opt adamw --lr 0.006 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EdgeNeXt: Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_s --tag il-common --opt adamw --lr 0.006 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EdgeViT

#### EdgeViT: Extra Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_xxs --tag il-common --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 200 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### EdgeViT: Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_xs --tag il-common --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 200 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientFormer v1

#### EfficientFormer v1: L1

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v1_l1 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientFormer v2

#### EfficientFormer v2: S0

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v2_s0 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### EfficientFormer v2: S1

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v2_s1 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientNet Lite

#### EfficientNet Lite: 0

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_lite0 --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --rgb-mode none --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientNet v1

#### EfficientNet v1: B0

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b0 --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 128 --epochs 300 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### EfficientNet v1: B1

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b1 --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 128 --epochs 300 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### EfficientNet v1: B2

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b2 --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 128 --epochs 300 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientNet v2

#### EfficientNet v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientViM

#### EfficientViM: M1

```sh
torchrun --nproc_per_node=2 train.py --network efficientvim_m1 --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 20 --batch-size 512 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --model-ema-steps 1 --model-ema-decay 0.9995 --clip-grad-norm 0.02 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientViT MIT

#### EfficientViT MIT: B0

```sh
torchrun --nproc_per_node=2 train.py --network efficientvit_mit_b0 --tag il-common --opt adamw --lr 0.00025 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.1 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientViT MSFT

#### EfficientViT MSFT: M0

```sh
torchrun --nproc_per_node=2 train.py --network efficientvit_msft_m0 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### FasterNet

#### FasterNet: T0

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_t0 --tag il-common --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.005 --grad-accum-steps 2 --smoothing-alpha 0.1 --aug-level 6 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### FasterNet: T1

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_t1 --tag il-common --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.01 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### FastViT

#### FastViT: T8

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_t8 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### FastViT: T12

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_t12 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### FastViT: SA12

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_sa12 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 128 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### FastViT: MobileClip v1 i0

```sh
torchrun --nproc_per_node=2 train.py --network mobileclip_v1_i0 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### GhostNet v1

#### GhostNet v1: 0.5 (50)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v1_0_5 --tag il-common --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 512 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### GhostNet v1: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v1_1_0 --tag il-common --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### GhostNet v2

#### GhostNet v2: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2_1_0 --tag il-common --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --warmup-epochs 3 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### GroupMixFormer

#### GroupMixFormer: Mobile

```sh
torchrun --nproc_per_node=2 train.py --network groupmixformer_mobile --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --clip-grad-norm 5 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### HGNet v1

#### HGNet v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network hgnet_v1_tiny --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 5 --batch-size 128 --epochs 400 --size 256 --wd 0.00004 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### HGNet v2

#### HGNet v2: B0

```sh
torchrun --nproc_per_node=2 train.py --network hgnet_v2_b0 --tag il-common --lr 0.125 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 5 --batch-size 128 --epochs 400 --size 256 --wd 0.00004 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### InceptionNeXt

#### InceptionNeXt: Atto

```sh
torchrun --nproc_per_node=2 train.py --network inception_next_a --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 512 --epochs 450 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### LeViT

#### LeViT: 128s

```sh
torchrun --nproc_per_node=2 train.py --network levit_128s --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 512 --warmup-epochs 5 --epochs 400 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --model-ema-decay 0.9998 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### LeViT: 128

```sh
torchrun --nproc_per_node=2 train.py --network levit_128 --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 512 --warmup-epochs 5 --epochs 400 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --model-ema-decay 0.9998 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### LIT v1

#### LIT v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network lit_v1_t --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --custom-layer-wd offset_conv=0.0 --custom-layer-lr-scale offset_conv=0.01 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### MetaFormer

#### MetaFormer: PoolFormer v1 s12

```sh
torchrun --nproc_per_node=2 train.py --network poolformer_v1_s12 --tag il-common --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --size 256 --warmup-epochs 5 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### MetaFormer: PoolFormer v2 s12

```sh
torchrun --nproc_per_node=2 train.py --network poolformer_v2_s12 --tag il-common --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --size 256 --warmup-epochs 5 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### MnasNet

#### MnasNet: 0.5

```sh
torchrun --nproc_per_node=2 train.py --network mnasnet_0_5 --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 200 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### MnasNet: 1

```sh
torchrun --nproc_per_node=2 train.py --network mnasnet_1_0 --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 200 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v1

#### Mobilenet v1: Original

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1_0_5 --tag orig_il-common --opt rmsprop --lr 0.045 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.94 --batch-size 256 --aug-level 4 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### Mobilenet v1: v4 procedure

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1_0_5 --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 600 --wd 0.01 --smoothing-alpha 0.1 --aug-level 6 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v2

#### Mobilenet v2: 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v2_1_0 --tag il-common --opt rmsprop --lr 0.045 --lr-scheduler step --lr-step-size 1 --lr-step-gamma 0.98 --batch-size 128 --size 256 --epochs 300 --wd 0.00004 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v3 Large

#### Mobilenet v3 Large: 0.75

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large_0_75 --tag il-common --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### Mobilenet v3 Large: 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large_1_0 --tag il-common --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v3 Small

#### Mobilenet v3 Small: 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_small_1_0 --tag il-common --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v4

#### Mobilenet v4: Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_s --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 512 --size 256 --epochs 800 --wd 0.01 --smoothing-alpha 0.1 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### Mobilenet v4: Medium

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_m --tag il-common --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 500 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --clip-grad-norm 5 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### Mobilenet v4: Large

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_l --tag il-common --opt adamw --lr 0.00225 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 20 --batch-size 256 --size 256 --epochs 500 --wd 0.2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --clip-grad-norm 5 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v4 Hybrid

#### Mobilenet v4 Hybrid: Medium

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_hybrid_m --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 20 --batch-size 256 --size 256 --epochs 500 --wd 0.15 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --clip-grad-norm 5 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### MobileOne

#### MobileOne: s0

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s0 --tag il-common --lr 0.1 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### MobileOne: s1

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s1 --tag il-common --lr 0.1 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### MobileViT v1

#### MobileViT v1: Extra Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1_xxs --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 300 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### MobileViT v1: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1_xs --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 300 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### MobileViT v2

#### MobileViT v2: 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2_1_0 --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 20 --batch-size 256 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### MobileViT v2: 1.5

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2_1_5 --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 20 --batch-size 128 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### MogaNet

#### MogaNet: X-Tiny

```sh
torchrun --nproc_per_node=2 train.py --network moganet_xt --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.03 --smoothing-alpha 0.1 --mixup-alpha 0.1 --cutmix --aug-level 6 --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### PiT

#### PiT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network pit_t --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 6 --model-ema --clip-grad-norm 1 --fast-matmul --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### PVT v2

#### PVT v2: B0 Linear

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b0_li --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### PVT v2: B0

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b0 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### PVT v2: B1

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b1 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### RegionViT

#### RegionViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network regionvit_t --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 50 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### RegNet

#### RegNet: X 200 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_x_200m --tag il-common --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 140 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### RegNet: X 400 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_x_400m --tag il-common --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 140 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --ra-sampler --ra-reps 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### RegNet: Y 200 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_200m --tag il-common --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 140 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### RegNet: Y 400 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_400m --tag il-common --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 140 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --ra-sampler --ra-reps 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### RegNet: Y 600 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_600m --tag il-common --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 140 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --ra-sampler --ra-reps 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### RegNet: Y 1.6 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_1_6g --tag il-common --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 256 --size 256 --epochs 160 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### RegNet: Z 500 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_z_500m --tag il-common --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 140 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --ra-sampler --ra-reps 2 --fast-matmul --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### RepGhost

#### RepGhost: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network repghost_1_0 --tag il-common --lr 0.6 --lr-scheduler cosine --warmup-epochs 5 --batch-size 256 --size 256 --epochs 300 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --model-ema --model-ema-steps 1 --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### RepViT

#### RepViT: M0.6

```sh
torchrun --nproc_per_node=2 train.py --network repvit_m0_6 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### ResNeSt

#### ResNeSt: 14

```sh
torchrun --nproc_per_node=2 train.py --network resnest_14 --tag il-common --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 270 --batch-size 256 --size 256 --wd 0.0001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 8 --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### ShuffleNet v1

#### ShuffleNet v1: Groups 4

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v1_4 --tag il-common --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### ShuffleNet v1: Groups 8

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v1_8 --tag il-common --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### ShuffleNet v2

#### ShuffleNet v2: Width 1.0

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v2_1_0 --tag il-common --lr 0.5 --lr-scheduler cosine --warmup-epochs 5 --batch-size 256 --size 256 --epochs 300 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --model-ema --ra-sampler --ra-reps 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### ShuffleNet v2: Width 2.0

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v2_2_0 --tag il-common --lr 0.5 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### SMT

### SMT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network smt_t --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --clip-grad-norm 5 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### SqueezeNet

```sh
torchrun --nproc_per_node=2 train.py --network squeezenet --tag il-common --lr 0.04 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.95 --batch-size 256 --size 259 --wd 0.0002 --smoothing-alpha 0.1 --aug-level 4 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### SqueezeNext

#### SqueezeNext: Width 1.0

```sh
torchrun --nproc_per_node=2 train.py --network squeezenext_1_0 --tag il-common --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.8 --batch-size 256 --size 259 --wd 0.0001 --warmup-epochs 5 --epochs 120 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### SqueezeNext: Width 2.0

```sh
torchrun --nproc_per_node=2 train.py --network squeezenext_2_0 --tag il-common --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.8 --batch-size 128 --size 259 --wd 0.0001 --warmup-epochs 5 --epochs 120 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### StarNet

#### StarNet: ESM05

```sh
torchrun --nproc_per_node=2 train.py --network starnet_esm05 --tag il-common --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 512 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### StarNet: ESM10

```sh
torchrun --nproc_per_node=2 train.py --network starnet_esm10 --tag il-common --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 512 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### StarNet: S1

```sh
torchrun --nproc_per_node=2 train.py --network starnet_s1 --tag il-common --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 512 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### SwiftFormer

#### SwiftFormer: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network swiftformer_xs --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 0.1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Tiny ViT

#### Tiny ViT: 5M

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --aug-level 8 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### TransNeXt

#### TransNeXt: Micro

```sh
torchrun --nproc_per_node=2 train.py --network transnext_micro --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 128 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --clip-grad-norm 1 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### VAN

#### VAN: B0

```sh
torchrun --nproc_per_node=2 train.py --network van_b0 --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### VoVNet v1

#### VoVNet v1: 39

```sh
torchrun --nproc_per_node=2 train.py --network vovnet_v1_39 --tag il-common --lr-scheduler cosine --batch-size 128 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 6 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### VoVNet v2

#### VoVNet v2: 19

```sh
torchrun --nproc_per_node=2 train.py --network vovnet_v2_19 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### VoVNet v2: 39

```sh
torchrun --nproc_per_node=2 train.py --network vovnet_v2_39 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 8 --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### XCiT

### XCiT: nano p16

```sh
torchrun --nproc_per_node=2 train.py --network xcit_nano12_p16 --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 30 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### XCiT: nano p8

```sh
torchrun --nproc_per_node=2 train.py --network xcit_nano12_p8 --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 30 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

## Knowledge Distillation

### DeiT (kd)

#### DeiT t16 with a ConvNeXt v2 Tiny teacher

```sh
torchrun --nproc_per_node=2 train_kd.py --type deit --teacher convnext_v2_tiny --teacher-tag intermediate-il-common --student deit_t16 --student-tag dist-il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 8 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```
