# Reference Knowledge Distillation Procedure

## Knowledge Distillation

### MobileNet v4

#### MobileNet v4 Medium with a ConvNeXt v2 Tiny teacher

```sh
torchrun --nproc_per_node=2 train_kd.py --type soft --teacher convnext_v2_tiny --student mobilenet_v4_m --student-tag dist --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 512 --size 256 --epochs 500 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### DeiT

#### DeiT s16 with a RegNet 8 GF teacher

```sh
torchrun --nproc_per_node=2 train_kd.py --type deit --teacher regnet --teacher-param 8 --teacher-tag intermediate --teacher-epoch 0 --student deit_s16 --student-tag dist --opt adamw --lr 0.0005 --lr-scheduler cosine --batch-size 128 --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### DeiT b16 with a RegNet 8 GF teacher

```sh
torchrun --nproc_per_node=2 train_kd.py --type deit --teacher regnet --teacher-param 8 --teacher-tag intermediate --teacher-epoch 0 --student deit_b16 --student-tag dist --opt adamw --lr 0.0005 --lr-scheduler cosine --batch-size 64 --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### DeiT3

#### DeiT3 s16 with a RegNet 8 GF teacher

```sh
torchrun --nproc_per_node=2 train_kd.py --type deit --teacher regnet --teacher-param 8 --teacher-tag intermediate --teacher-epoch 0 --student deit3_s16 --student-tag dist --opt adamw --lr 0.0005 --lr-scheduler cosine --batch-size 128 --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```
