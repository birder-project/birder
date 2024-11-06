# Reference Pre-training Procedure

Training script and procedures adapted from PyTorch vision reference
<https://github.com/pytorch/vision/tree/main/references/classification>

Set `OMP_NUM_THREADS`.

## Image Pre-training

* [FCMAE](#fcmae)
* [MAE ViT](#mae-vit)
* [SimMIM](#simmim)

### FCMAE

#### FCMAE: ConvNeXt v2 Nano

```sh
torchrun --nproc_per_node=2 train_mim.py --network fcmae --encoder convnext_v2_nano --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 512 --wd 0.05 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### FCMAE: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 train_mim.py --network fcmae --encoder convnext_v2_tiny --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 256 --wd 0.05 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 15 --batch-size 256 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --layer-decay 0.9 --resume-epoch 10 --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: fine-tuning, second stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim-intermediate --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 384 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim-intermediate --opt adamw --lr 0.000025 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 64 --epochs 60 --size 448 --wd 1e-8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.9 --resume-epoch 10
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 320 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 128 --epochs 210 --size 320 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.9 --resume-epoch 10
```

At epoch 120 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 64 --epochs 210 --size 384 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.9 --resume-epoch 120 --load-states
```

At epoch 170 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 64 --epochs 210 --size 448 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.9 --resume-epoch 170 --load-states
```

#### FCMAE: ConvNeXt v2 Base

```sh
torchrun --nproc_per_node=2 train_mim.py --network fcmae --encoder convnext_v2_base --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 128 --wd 0.05 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 15 --batch-size 128 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --layer-decay 0.8 --resume-epoch 10 --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: fine-tuning, second stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim-intermediate --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 384 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim-intermediate --opt adamw --lr 0.000025 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 32 --epochs 40 --size 448 --wd 1e-8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.8 --resume-epoch 10
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 320 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

#### FCMAE: RegNet Y 8 GF

```sh
torchrun --nproc_per_node=2 train_mim.py --network fcmae --encoder regnet_y_8g --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 128 --wd 0.05 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --lr 0.1 --lr-scheduler cosine --warmup-epochs 15 --batch-size 128 --size 256 --epochs 110 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --compile --layer-decay 0.9 --resume-epoch 10 --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --lr 0.001 --lr-scheduler cosine --batch-size 128 --size 256 --epochs 30 --wd 0.0005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --compile --layer-decay 0.8 --resume-epoch 10
```

### MAE Hiera

#### MAE Hiera: Hiera Small

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_hiera --encoder hiera_small --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 256 --wd 0.05 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE Hiera: Hiera AbsWin Tiny

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_hiera --encoder hiera_abswin_tiny --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 256 --wd 0.05 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_tiny --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --wd 0.3 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_tiny --tag mim --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 310 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --layer-decay 0.65 --resume-epoch 10 --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_tiny --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --wd 0.3 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

#### MAE Hiera: Hiera AbsWin Small

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_hiera --encoder hiera_abswin_small --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 256 --wd 0.05 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE Hiera: Hiera AbsWin Base

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_hiera --encoder hiera_abswin_base --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 400 --batch-size 256 --wd 0.05 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE Hiera: Hiera AbsWin Base Plus

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_hiera --encoder hiera_abswin_base_plus --opt adamw --lr 0.0008 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --epochs 800 --batch-size 128 --wd 0.05 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### MAE ViT

#### MAE ViT: Simple ViT b16

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_vit --encoder simple_vit_b16 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 256 --wd 0.0001 --clip-grad-norm 1 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE ViT: ViT SAM b16

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_vit --encoder vit_sam_b16 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 128 --wd 0.0001 --clip-grad-norm 1 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE ViT: ViTReg4 b16

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_vit --encoder vitreg4_b16 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 256 --wd 0.05 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_b16 --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 512 --epochs 10 --size 256 --wd 0.3 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_b16 --tag mim --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 320 --wd 0.3 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_b16 --tag mim --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 15 --epochs 100 --size 320 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 10
```

At epoch 75 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_b16 --tag mim --opt adamw --lr 0.00004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --epochs 100 --size 384 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 75
```

At epoch 80 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_b16 --tag mim --opt adamw --lr 0.00002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 32 --epochs 100 --size 448 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 80
```

#### MAE ViT: ViTReg4 l16

```sh
torchrun --nproc_per_node=2 train_mim.py --network mae_vit --encoder vitreg4_l16 --opt adamw --lr 0.00015 --opt-betas 0.9 0.95 --lr-scheduler cosine --warmup-epochs 40 --batch-size 128 --wd 0.05 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_l16 --tag mim --lr 0.3 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 256 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_l16 --tag mim --lr 0.3 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --epochs 10 --size 320 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_l16 --tag mim --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 32 --warmup-epochs 20 --epochs 100 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 10
```

At epoch 75 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_l16 --tag mim --lr 0.01 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 16 --epochs 100 --size 384 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 75
```

At epoch 80 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network vitreg4_l16 --tag mim --lr 0.0075 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 16 --epochs 100 --size 448 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 80
```

### SimMIM

#### SimMIM: MaxViT Tiny

```sh
torchrun --nproc_per_node=2 train_mim.py --network simmim --encoder maxvit_t --opt adamw --lr 0.0001 --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_t --tag mim --opt adamw --lr 0.00005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --size 256 --epochs 10 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --clip-grad-norm 1 --amp --resume-epoch 0 --reset-head --freeze-body
```

#### SimMIM: NextViT Small

```sh
torchrun --nproc_per_node=2 train_mim.py --network simmim --encoder nextvit_s --opt adamw --lr 0.0001 --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: NextViT Base

```sh
torchrun --nproc_per_node=2 train_mim.py --network simmim --encoder nextvit_b --opt adamw --lr 0.0001 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: Swin Transformer v2 Small

```sh
torchrun --nproc_per_node=2 train_mim.py --network simmim --encoder swin_transformer_v2_s --opt adamw --lr 0.0001 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

After N epochs, switch to float16

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --tag mim --opt adamw --lr 0.00005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --size 256 --epochs 10 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --clip-grad-norm 5 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --tag mim --opt adamw --lr 0.0006 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 256 --warmup-epochs 20 --epochs 110 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --layer-decay 0.9 --resume-epoch 10
```

#### SimMIM: Swin Transformer v2 Base

```sh
torchrun --nproc_per_node=2 train_mim.py --network simmim --encoder swin_transformer_v2_b --opt adamw --lr 0.0001 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --wd 0.05 --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

After N epochs, switch to float16

#### SimMIM: Swin Transformer v2 w2 Base

```sh
torchrun --nproc_per_node=2 train_mim.py --network simmim --encoder swin_transformer_v2_w2_b --opt adamw --lr 0.0001 --lr-scheduler cosine --warmup-epochs 10 --batch-size 64 --wd 0.05 --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

After N epochs, switch to float16

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_w2_b --tag mim --opt adamw --lr 0.00005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --size 256 --epochs 10 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --clip-grad-norm 5 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```
