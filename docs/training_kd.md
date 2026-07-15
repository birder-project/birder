# Reference Knowledge Distillation Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## Knowledge Distillation

### MobileNet v4

#### MobileNet v4 Medium with a ConvNeXt v2 Tiny teacher

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_kd --type soft --teacher convnext_v2_tiny --student mobilenet_v4_m --student-tag dist --batch-size 512 --opt adamw --clip-grad-norm 5 --lr 0.003 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 500 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --amp --compile
```

### DeiT

#### DeiT s16 with a RegNet Y 8 GF teacher

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_kd --type deit --teacher regnet_y_8g --teacher-tag intermediate --teacher-epoch 0 --student deit_s16 --student-tag dist --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --amp --compile
```

#### DeiT b16 with a RegNet Y 8 GF teacher

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_kd --type deit --teacher regnet_y_8g --teacher-tag intermediate --teacher-epoch 0 --student deit_b16 --student-tag dist --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### Tiny ViT

#### Tiny ViT 5M with a ViT l16 teacher

Intermediate training

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_kd --type soft --teacher vit_l16 --teacher-tag intermediate --teacher-epoch 0 --student tiny_vit_5m --student-tag dist --temperature 1 --batch-size 64 --opt adamw --clip-grad-norm 5 --grad-accum-steps 2 --lr 0.002 --wd 0.01 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```
