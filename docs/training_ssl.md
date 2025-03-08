# Reference Self-supervised Training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## SSL Pre-training

- [DINO v1](#dino-v1)
- [VICReg](#vicreg)

### DINO v1

#### DINO v1: XCiT small-12 p16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network xcit_small12_p16 --local-crops-number 10 --teacher-temp 0.07 --opt adamw --lr 0.00025 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 10 --batch-size 96 --wd 0.04 --norm-wd 0 --bias-weight-decay 0 --wd-end 0.4 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network xcit_small12_p16 --tag dino-v1 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

#### DINO v1: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_dino_v1 --network efficientnet_v2_s --use-bn-in-head --norm-last-layer --local-crops-number 6 --teacher-temp 0.07 --opt lars --lr 0.3 --lr-scheduler cosine --lr-cosine-min 0.001 --epochs 800 --warmup-epochs 10 --batch-size 128 --wd 0.000001 --norm-wd 0 --bias-weight-decay 0 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag dino-v1 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

### VICReg

Use `--sync-bn` when batch size is 32 or below.

#### VICReg: EfficientNet v2 Medium

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_vicreg --network efficientnet_v2_m --opt lars --lr 0.2 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --epochs 400 --wd 0.000001 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag vicreg --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag vicreg --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 128 --epochs 200 --size 256 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.98 --resume-epoch 0
```
