# Compact Networks Training Procedure

## Compact Madness

### ConvNeXt v2

#### ConvNeXt v2: Atto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_atto --tag il-common --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 600 --size 256 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### CrossViT

#### CrossViT: 9 Dagger

```sh
torchrun --nproc_per_node=2 train.py --network crossvit_9d --tag il-common --opt adamw --lr 0.004 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --warmup-epochs 30 --epochs 300 --size 240 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --amp --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### DeiT

#### DeiT: t16

```sh
torchrun --nproc_per_node=2 train.py --network deit_t16 --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EdgeNeXt

#### EdgeNeXt: Extra Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_xxs --tag il-common --opt adamw --lr 0.006 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EdgeViT

#### EdgeViT: Extra Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_xxs --tag il-common --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 200 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### EdgeViT: Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_xs --tag il-common --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 200 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientFormer v1

#### EfficientFormer v1: L1

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v1_l1 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientFormer v2

#### EfficientFormer v2: S0

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v2_s0 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### EfficientFormer v2: S1

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v2_s1 --tag il-common --opt adamw --lr 0.001 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### EfficientNet v1

#### EfficientNet v1: B0

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b0 --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 128 --epochs 300 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### GhostNet v2

#### GhostNet v2: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2 --net-param 1 --tag il-common --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### MnasNet

```sh
torchrun --nproc_per_node=2 train.py --network mnasnet --net-param 0.5 --tag il-common --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 200 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v1

#### Mobilenet v1: Original

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1 --net-param 0.5 --tag orig_il-common --opt rmsprop --lr 0.045 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.94 --batch-size 256 --aug-level 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### Mobilenet v1: v4 procedure

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1 --net-param 0.5 --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 600 --wd 0.01 --smoothing-alpha 0.1 --aug-level 3 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v2

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v2 --net-param 1 --tag il-common --opt rmsprop --lr 0.045 --lr-scheduler step --lr-step-size 1 --lr-step-gamma 0.98 --batch-size 128 --size 256 --epochs 300 --wd 0.00004 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v3 Large

#### Mobilenet v3 Large: 0.75

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large --net-param 0.75 --tag il-common --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v3 Small

#### Mobilenet v3 Small: 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_small --net-param 1 --tag il-common --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### Mobilenet v4

#### Mobilenet v4: Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_s --tag il-common --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 512 --size 256 --epochs 800 --wd 0.01 --smoothing-alpha 0.1 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### Mobilenet v4: Medium

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_m --tag il-common --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 512 --size 256 --epochs 500 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### MobileOne

#### MobileOne: s0

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s0 --tag il-common --lr 0.1 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### MobileOne: s1

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s1 --tag il-common --lr 0.1 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### RegNet

#### RegNet: 0.2 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet --net-param 0.2 --tag il-common --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### ShuffleNet v1

#### ShuffleNet v1: Groups 4

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v1 --net-param 4 --tag il-common --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256  --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### ShuffleNet v2

#### ShuffleNet v2: Width 1.0

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v2 --net-param 1 --tag il-common --lr 0.5 --lr-scheduler cosine --warmup-epochs 5 --batch-size 256 --size 256 --epochs 300 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### ShuffleNet v2: Width 2.0

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v2 --net-param 2 --tag il-common --lr 0.5 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### ShuffleNet v1: Groups 8

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v1 --net-param 8 --tag il-common --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256  --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### SqueezeNet

```sh
torchrun --nproc_per_node=2 train.py --network squeezenet --tag il-common --lr 0.04 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.95 --batch-size 256 --size 259 --wd 0.0002 --smoothing-alpha 0.1 --aug-level 2 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### SqueezeNext

#### SqueezeNext: Width 1.0

```sh
torchrun --nproc_per_node=2 train.py --network squeezenext --net-param 1 --tag il-common --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.8 --batch-size 256 --size 259 --wd 0.0001 --warmup-epochs 5 --epochs 120 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

#### SqueezeNext: Width 2.0

```sh
torchrun --nproc_per_node=2 train.py --network squeezenext --net-param 2 --tag il-common --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.8 --batch-size 128 --size 259 --wd 0.0001 --warmup-epochs 5 --epochs 120 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```

### XCiT

### XCiT: nano p16

```sh
torchrun --nproc_per_node=2 train.py --network xcit_nano16 --tag il-common --opt adamw --lr 0.0005 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --warmup-epochs 30 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile --data-path data/training_il-common_packed --val-path data/validation_il-common_packed
```
