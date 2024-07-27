# Reference Knowledge Distillation Procedure

## Knowledge Distillation

### ConvNeXt v2 Tiny Teacher: MobileNet v4 Medium

```sh
torchrun --nproc_per_node=2 train_kd.py --type soft --teacher convnext_v2_tiny --student mobilenet_v4_m --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 512 --size 256 --epochs 500 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### ConvNeXt v2 Tiny Teacher: DeiT b16

```sh
torchrun --nproc_per_node=2 train_kd.py --type deit --teacher convnext_v2_tiny --student deit_b16 --opt adamw --lr 0.0005 --lr-scheduler cosine --warmup-epochs 20 --batch-size 128 --size 256 --epochs 300 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```
