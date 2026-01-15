# Reference Pre-training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

Examples use repo-root script names (e.g., `train_mim.py`). If you installed Birder as a package, use the module form such as `python -m birder.scripts.train_mim`.

## Image Pre-training

- [CrossMAE](#crossmae)
- [FCMAE](#fcmae)
- [MAE Hiera](#mae-hiera)
- [MAE ViT](#mae-vit)
- [SimMIM](#simmim)

### CrossMAE

#### CrossMAE: ViT reg4 b14

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network crossmae --encoder vit_reg4_b14 --encoder-model-config drop_path_rate=0.0 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### CrossMAE: SoViT reg4 150m p14 AVG

BIO

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network crossmae_dec512d12 --tag bio --encoder vit_reg4_so150m_p14_avg --encoder-model-config drop_path_rate=0.0 --min-mask-size 2 --batch-size 384 --opt adamw --opt-fused --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --epochs 400 --warmup-epochs 20 --rgb-mode none --amp --amp-dtype bfloat16 --compile --wds --wds-info /mnt/data/ssl_bio_packed/_info.json
```

### FCMAE

#### FCMAE: ConvNeXt v2 Atto

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder convnext_v2_atto --encoder-model-config drop_path_rate=0.0 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --epochs 400 --warmup-epochs 40 --fast-matmul --compile --find-unused-parameters --data-path data/training
```

Full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_atto --tag mim --batch-size 512 --opt adamw --lr 0.0002 --wd 0.3 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 600 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.2 --ra-sampler --ra-reps 2 --fast-matmul --compile --resume-epoch 0
```

#### FCMAE: ConvNeXt v2 Nano

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder convnext_v2_nano --encoder-model-config drop_path_rate=0.0 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### FCMAE: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder convnext_v2_tiny --encoder-model-config drop_path_rate=0.0 --batch-size 256 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Intermediate training training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --batch-size 256 --opt adamw --lr 0.0002 --wd 0.05 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 100 --warmup-epochs 15 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: region-specific full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim-intermediate --batch-size 64 --opt adamw --lr 0.000025 --wd 1e-8 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 60 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --compile-opt --resume-epoch 0
```

Full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --batch-size 128 --opt adamw --lr 0.0002 --wd 0.05 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 210 --warmup-epochs 30 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --compile-opt --resume-epoch 0
```

At epoch 120 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --batch-size 64 --opt adamw --lr 0.0002 --wd 0.05 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 210 --warmup-epochs 30 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --compile-opt --resume-epoch 120 --load-states
```

At epoch 170 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag mim --batch-size 64 --opt adamw --lr 0.0002 --wd 0.05 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 210 --warmup-epochs 30 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --compile-opt --resume-epoch 170 --load-states
```

#### FCMAE: ConvNeXt v2 Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder convnext_v2_base --encoder-model-config drop_path_rate=0.0 --batch-size 128 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Intermediate training training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim --batch-size 128 --opt adamw --lr 0.0002 --wd 0.05 --layer-decay 0.8 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 100 --warmup-epochs 15 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: region-specific full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag mim-intermediate --batch-size 32 --opt adamw --lr 0.000025 --wd 1e-8 --layer-decay 0.8 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 40 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --compile-opt --resume-epoch 0
```

#### FCMAE: RegNet Y 8 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network fcmae --encoder regnet_y_8g --batch-size 128 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Intermediate training training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag mim --batch-size 128 --lr 0.1 --wd 0.00005 --layer-decay 0.9 --lr-scheduler cosine --epochs 110 --warmup-epochs 15 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile --compile-opt --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

### MAE Hiera

#### MAE Hiera: Hiera Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_small --encoder-model-config drop_path_rate=0.0 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --lr 0.0008 --wd 0.05 --lr-scheduler cosine --epochs 400 --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE Hiera: Hiera AbsWin Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_tiny --encoder-model-config drop_path_rate=0.0 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --lr 0.0008 --wd 0.05 --lr-scheduler cosine --epochs 400 --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Intermediate training training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_tiny --tag mim --batch-size 256 --opt adamw --lr 0.002 --lr-scale 1024 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 310 --warmup-epochs 15 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

#### MAE Hiera: Hiera AbsWin Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_small --encoder-model-config drop_path_rate=0.0 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --lr 0.0008 --wd 0.05 --lr-scheduler cosine --epochs 400 --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE Hiera: Hiera AbsWin Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_base --encoder-model-config drop_path_rate=0.2 --batch-size 512 --opt adamw --opt-betas 0.9 0.95 --lr 0.0008 --wd 0.05 --lr-scheduler cosine --epochs 400 --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Intermediate training training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim-intermediate --batch-size 192 --opt adamw --lr 0.002 --lr-scale 1024 --wd 0.05 --norm-wd 0 --transformer-embedding-decay 0 --layer-decay 0.7 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim-intermediate --batch-size 80 --opt adamw --lr 0.0003 --lr-scale 1024 --wd 0.05 --norm-wd 0 --transformer-embedding-decay 0 --layer-decay 0.7 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 30 --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_base --tag mim --batch-size 128 --opt adamw --lr 0.002 --wd 0.05 --norm-wd 0 --layer-decay 0.7 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --compile-opt --resume-epoch 0
```

#### MAE Hiera: Hiera AbsWin Base Plus

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_hiera --encoder hiera_abswin_base_plus --encoder-model-config drop_path_rate=0.2 --batch-size 256 --opt adamw --opt-betas 0.9 0.95 --lr 0.0008 --wd 0.05 --lr-scheduler cosine --epochs 400 --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### MAE ViT

#### MAE ViT: Simple ViT b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder simple_vit_b16 --batch-size 256 --opt adamw --opt-betas 0.9 0.95 --clip-grad-norm 1 --lr 0.00015 --wd 0.0001 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE ViT: ViT SAM b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder vit_sam_b16 --encoder-model-config drop_path_rate=0.0 --batch-size 128 --opt adamw --opt-betas 0.9 0.95 --clip-grad-norm 1 --lr 0.00015 --wd 0.0001 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE ViT: ViT reg4 b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder vit_reg4_b16 --encoder-model-config drop_path_rate=0.0 --batch-size 256 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Intermediate training training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim-intermediate --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 110 --warmup-epochs 15 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --save-frequency 1 --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim-intermediate --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.75 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 110 --warmup-epochs 15 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --save-frequency 1 --resume-epoch 80 --load-scheduler --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 110 --warmup-epochs 15 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 0
```

At epoch 80 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_b16 --tag mim --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.00004 --wd 0.05 --norm-wd 0 --layer-decay 0.65 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 110 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --compile-opt --resume-epoch 80
```

#### MAE ViT: ViT l16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder vit_l16 --encoder-model-config drop_path_rate=0.0 --batch-size 256 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### MAE ViT: ViT reg8 l16 AVG

Pixio like training (should be fine-tuned later with APS type attention pooling)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit_dec512d24_npl --encoder vit_reg8_l16_avg --min-mask-size 4 --batch-size 64 --opt adamw --opt-betas 0.9 0.95 --lr 0.00025 --wd 0.05 --lr-scheduler cosine --epochs 400 --warmup-epochs 40 --size 256 --rgb-mode none --amp --amp-dtype bfloat16 --compile --compile-opt --wds --wds-info data/ssl_packed/_info.json
```

#### MAE ViT: RoPE SoViT reg4 150m p14 AP

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network mae_vit --encoder rope_vit_reg4_so150m_p14_ap --encoder-model-config drop_path_rate=0.0 --batch-size 256 --opt adamw --opt-betas 0.9 0.95 --lr 0.00015 --wd 0.05 --lr-scheduler cosine --warmup-epochs 40 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

### SimMIM

#### SimMIM: MaxViT Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder maxvit_t --encoder-model-config drop_path_rate=0.0 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.05 --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: NextViT Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder nextvit_s --encoder-model-config drop_path_rate=0.0 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.05 --lr-scheduler cosine --epochs 100 --warmup-epochs 10 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: NextViT Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder nextvit_b --encoder-model-config drop_path_rate=0.0 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.05 --lr-scheduler cosine --warmup-epochs 10 --amp --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: RegNet Y 4 GF

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder regnet_y_4g --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --lr-scheduler cosine --warmup-epochs 10 --amp --compile --compile-opt --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: Swin Transformer v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder swin_transformer_v2_s --encoder-model-config drop_path_rate=0.0 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.05 --lr-scheduler cosine --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --tag mim --batch-size 64 --opt adamw --clip-grad-norm 5 --lr 0.0006 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 110 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --compile-opt --resume-epoch 0
```

#### SimMIM: Swin Transformer v2 Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder swin_transformer_v2_b --encoder-model-config drop_path_rate=0.0 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.05 --lr-scheduler cosine --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: Swin Transformer v2 w2 Base

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_mim --network simmim --encoder swin_transformer_v2_w2_b --encoder-model-config drop_path_rate=0.0 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.05 --lr-scheduler cosine --warmup-epochs 10 --amp --amp-dtype bfloat16 --compile --compile-opt --find-unused-parameters --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```
