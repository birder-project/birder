# Reference Knowledge Distillation Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## Knowledge Distillation

### MobileNet v4

#### MobileNet v4 Medium with a ConvNeXt v2 Tiny teacher

```sh
torchrun --nproc_per_node=2 train_kd.py --type soft --teacher convnext_v2_tiny --student mobilenet_v4_m --student-tag dist --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 512 --size 256 --epochs 500 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### DeiT

#### DeiT s16 with a RegNet Y 8 GF teacher

```sh
torchrun --nproc_per_node=2 train_kd.py --type deit --teacher regnet_y_8g --teacher-tag intermediate --teacher-epoch 0 --student deit_s16 --student-tag dist --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### DeiT b16 with a RegNet Y 8 GF teacher

```sh
torchrun --nproc_per_node=2 train_kd.py --type deit --teacher regnet_y_8g --teacher-tag intermediate --teacher-epoch 0 --student deit_b16 --student-tag dist --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 5 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### Tiny ViT

#### Tiny ViT 5M with a ViT L16 teacher

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train_kd.py --type soft --temperature 1 --teacher vit_l16 --teacher-tag intermediate --teacher-epoch 0 --student tiny_vit_5m --student-tag dist --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 5 --epochs 90 --size 256 --wd 0.01 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --aug-level 4 --clip-grad-norm 5 --amp --compile --wds --wds-class-file data/intermediate/classes.txt --wds-info-file data/intermediate/_info.json
```
