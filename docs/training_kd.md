# Reference Knowledge Distillation Procedure

## Knowledge Distillation

### ConvNeXt v2 Tiny Teacher: RegNet 1.6 GF

```sh
torchrun --nproc_per_node=2 train_kd.py --teacher convnext_v2_tiny --teacher-epoch 0 --student regnet --student-param 1.6 --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --compile
```

### ConvNeXt v2 Tiny Teacher: MobileNet v3

```sh
torchrun --nproc_per_node=2 train_kd.py --teacher convnext_v2_tiny --teacher-epoch 0 --student mobilenet_v3 --student-param 1 --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 64 --size 288 --epochs 300 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --compile
```
