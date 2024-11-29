# Reference Training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## Table of Contents

* [General Training Guidelines](#general-training-guidelines)
* [Network Specific Training Procedures](#network-specific-training-procedures)
* [Common Dataset Training Scenarios](#common-dataset-training-scenarios)

## General Training Guidelines

It's best to avoid cutmix on compact models - <https://arxiv.org/abs/2404.11202v1>

Most networks train more effectively with growing resolution and augmentation as described at - <https://arxiv.org/abs/2104.00298>

Intermediate training on weakly supervised dataset:

* Run without EMA (Exponential Moving Average)
* Use higher weight decay
* Maintain the same learning rate and number of epochs
* Use the same augmentations
* Use the lowest resolution for the model (usually 256x256)

On fine-tuning phase

* Run only the last training phase (highest resolution)
* Limit to at most 30% of the total epochs
* Apply a small layer-decay (0.98 - 0.99)
* Consider using a slightly lower learning rate

## Network Specific Training Procedures

* [AlexNet](#alexnet)
* [CaiT](#cait)
* [CoaT](#coat)
* [ConvMixer](#convmixer)
* [ConvNeXt v1](#convnext-v1)
* [ConvNeXt v2](#convnext-v2)
* [CrossViT](#crossvit)
* [CSPNet](#cspnet)
* [CSWin Transformer](#cswin-transformer)
* [Darknet](#darknet)
* [DaViT](#davit)
* [DeiT](#deit)
* [DeiT3](#deit3)
* [DenseNet](#densenet)
* [DPN](#dpn)
* [EdgeNeXt](#edgenext)
* [EdgeViT](#edgevit)
* [EfficientFormer v1](#efficientformer-v1)
* [EfficientFormer v2](#efficientformer-v2)
* [EfficientNet Lite](#efficientnet-lite)
* [EfficientNet v1](#efficientnet-v1)
* [EfficientNet v2](#efficientnet-v2)
* [FasterNet](#fasternet)
* [FastViT](#fastvit)
* [FocalNet](#focalnet)
* [GhostNet v1](#ghostnet-v1)
* [GhostNet v2](#ghostnet-v2)
* [Hiera](#hiera)
* [HieraDet](#hieradet)
* [InceptionNeXt](#inceptionnext)
* [Inception-ResNet v2](#inception-resnet-v2)
* [Inception v3](#inception-v3)
* [Inception v4](#inception-v4)
* [LeViT](#levit)
* [MaxViT](#maxvit)
* [MetaFormer](#metaformer)
* [MnasNet](#mnasnet)
* [Mobilenet v1](#mobilenet-v1)
* [Mobilenet v2](#mobilenet-v2)
* [Mobilenet v3 Large](#mobilenet-v3-large)
* [Mobilenet v3 Small](#mobilenet-v3-small)
* [Mobilenet v4](#mobilenet-v4)
* [Mobilenet v4 Hybrid](#mobilenet-v4-hybrid)
* [MobileOne](#mobileone)
* [MobileViT v1](#mobilevit-v1)
* [MobileViT v2](#mobilevit-v2)
* [MogaNet](#moganet)
* [MViT v2](#mvit-v2)
* [Next-ViT](#next-vit)
* [NFNet](#nfnet)
* [PiT](#pit)
* [PVT v1](#pvt-v1)
* [PVT v2](#pvt-v2)
* [RDNet](#rdnet)
* [RegNet](#regnet)
* [RepGhost](#repghost)
* [RepVgg](#repvgg)
* [ResMLP](#resmlp)
* [ResNeSt](#resnest)
* [ResNet v2](#resnet-v2)
* [ResNeXt](#resnext)
* [SE ResNet v2](#se-resnet-v2)
* [SE ResNeXt](#se-resnext)
* [Sequencer2d](#sequencer2d)
* [ShuffleNet v1](#shufflenet-v1)
* [ShuffleNet v2](#shufflenet-v2)
* [Simple ViT](#simple-vit)
* [SqueezeNet](#squeezenet)
* [SqueezeNext](#squeezenext)
* [Swin Transformer v1](#swin-transformer-v1)
* [Swin Transformer v2](#swin-transformer-v2)
* [Tiny ViT](#tiny-vit)
* [UniFormer](#uniformer)
* [VGG](#vgg)
* [VGG Reduced](#vgg-reduced)
* [ViT](#vit)
* [ViT SAM](#vit-sam)
* [ViTReg](#vitreg)
* [Wide ResNet](#wide-resnet)
* [Xception](#xception)
* [XCiT](#xcit)

### AlexNet

```sh
torchrun --nproc_per_node=2 train.py --network alexnet --lr 0.01 --batch-size 128 --aug-level 2
```

### CaiT

#### CaiT: Small 24

```sh
torchrun --nproc_per_node=2 train.py --network cait_s24 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 5 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### CoaT

#### CoaT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network coat_tiny --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile
```

#### CoaT: Mini

```sh
torchrun --nproc_per_node=2 train.py --network coat_mini --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile
```

#### CoaT: Small

```sh
torchrun --nproc_per_node=2 train.py --network coat_small --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 32 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### CoaT: Lite Tiny

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_tiny --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --fast-matmul --compile
```

#### CoaT: Lite Mini

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_mini --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile
```

#### CoaT: Lite Small

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_small --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile
```

#### CoaT: Lite Medium

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_medium --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 32 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### ConvMixer

#### ConvMixer: 768 / 32

```sh
torchrun --nproc_per_node=2 train.py --network convmixer_768_32 --opt adamw --lr 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --epochs 300 --smoothing-alpha 0.1 --mixup-alpha 0.5 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ConvMixer: 1024 / 20

```sh
torchrun --nproc_per_node=2 train.py --network convmixer_1024_20 --opt adamw --lr 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --epochs 150 --smoothing-alpha 0.1 --mixup-alpha 0.5 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ConvMixer: 1536 / 20

```sh
torchrun --nproc_per_node=2 train.py --network convmixer_1536_20 --opt adamw --lr 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --epochs 150 --smoothing-alpha 0.1 --mixup-alpha 0.5 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### ConvNeXt v1

#### ConvNeXt v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1_tiny --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 128 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none
```

#### ConvNeXt v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1_small --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 320 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none
```

#### ConvNeXt v1: Base

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1_base --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none
```

#### ConvNeXt v1: Large

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1_large --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 448 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none
```

### ConvNeXt v2

#### ConvNeXt v2: Atto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_atto --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 600 --size 256 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### ConvNeXt v2: Femto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_femto --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 600 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.2 --mixup-alpha 0.3 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### ConvNeXt v2: Pico

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_pico --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 600 --size 288 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.2 --mixup-alpha 0.3 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### ConvNeXt v2: Nano

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_nano --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 600 --size 288 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.2 --mixup-alpha 0.5 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

#### ConvNeXt v2: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 128 --epochs 300 --size 320 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --stop-epoch 200
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 64 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 200 --load-states --stop-epoch 240
```

At epoch 240 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 64 --epochs 300 --size 448 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 240 --load-states
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag intermediate --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 10 --batch-size 256 --epochs 100 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag intermediate --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 448 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag intermediate --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 64 --epochs 80 --size 448 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.99 --resume-epoch 10
```

#### ConvNeXt v2: Base

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 16 --epochs 200 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag intermediate --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 64 --epochs 100 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: at epoch 80 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag intermediate --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 64 --epochs 100 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --amp --compile --resume-epoch 80 --load-states --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag intermediate --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --model-ema --amp --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag intermediate --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 32 --epochs 80 --size 448 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.99 --resume-epoch 10
```

#### ConvNeXt v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_large --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 16 --epochs 200 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### ConvNeXt v2: Huge

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_huge --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 4 --epochs 100 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### CrossViT

#### CrossViT: 9 Dagger

```sh
torchrun --nproc_per_node=2 train.py --network crossvit_9d --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 30 --epochs 300 --size 240 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --amp
```

#### CrossViT: 15 Dagger

```sh
torchrun --nproc_per_node=2 train.py --network crossvit_15d --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 30 --epochs 300 --size 336 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile
```

#### CrossViT: 18

```sh
torchrun --nproc_per_node=2 train.py --network crossvit_18 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 30 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --grad-accum-steps 16 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### CSPNet

#### CSPNet: CSP ResNet 50

```sh
torchrun --nproc_per_node=2 train.py --network csp_resnet_50 --lr 0.1 --lr-scheduler polynomial --lr-power 4 --batch-size 128 --epochs 90 --size 256 --wd 0.0005 --smoothing-alpha 0.1 --aug-level 3 --fast-matmul --compile
```

#### CSPNet: CSP ResNeXt 50

```sh
torchrun --nproc_per_node=2 train.py --network csp_resnext_50 --lr 0.1 --lr-scheduler polynomial --lr-power 4 --batch-size 128 --epochs 90 --size 256 --wd 0.0005 --smoothing-alpha 0.1 --aug-level 3 --fast-matmul --compile
```

#### CSPNet: CSP Darknet 53

```sh
torchrun --nproc_per_node=2 train.py --network csp_darknet_53 --lr 0.1 --lr-scheduler polynomial --lr-power 4 --batch-size 128 --epochs 90 --size 256 --wd 0.0005 --smoothing-alpha 0.1 --aug-level 3 --fast-matmul --compile
```

#### CSPNet: CSP SE ResNet 50

Same as non SE version

#### CSPNet: CSP SE ResNeXt 50

Same as non SE version

#### CSPNet: CSP SE Darknet 53

Same as non SE version

### CSWin Transformer

#### CSWin Transformer: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network cswin_transformer_t --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --size 256 --warmup-epochs 20 --epochs 300 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### CSWin Transformer: Small

```sh
torchrun --nproc_per_node=2 train.py --network cswin_transformer_s --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 256 --warmup-epochs 20 --epochs 300 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### CSWin Transformer: Base

```sh
torchrun --nproc_per_node=2 train.py --network cswin_transformer_b --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 32 --size 256 --warmup-epochs 20 --epochs 300 --wd 0.1 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### CSWin Transformer: Large

```sh
torchrun --nproc_per_node=2 train.py --network cswin_transformer_l --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 16 --size 256 --warmup-epochs 20 --epochs 300 --wd 0.1 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

On intermediate training use wd of 0.2 (large only)

### Darknet

#### Darknet: 53

```sh
torchrun --nproc_per_node=2 train.py --network darknet_53 --lr 0.1 --lr-scheduler cosine --batch-size 128 --epochs 90 --size 256 --wd 0.0005 --smoothing-alpha 0.1 --aug-level 3 --fast-matmul --compile
```

### DaViT

#### DaViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network davit_tiny --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### DeiT

#### DeiT: t16

```sh
torchrun --nproc_per_node=2 train.py --network deit_t16 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --fast-matmul --compile
```

#### DeiT: s16

```sh
torchrun --nproc_per_node=2 train.py --network deit_s16 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile
```

#### DeiT: b16

```sh
torchrun --nproc_per_node=2 train.py --network deit_b16 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 288 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --stop-epoch 200
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network deit_b16 --opt adamw --lr 0.00015 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --resume-epoch 200
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network deit_b16 --tag intermediate --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 200 --size 256 --wd 0.1 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --clip-grad-norm 1 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

### DeiT3

Same as DeiT

### DenseNet

#### DenseNet: 161

```sh
torchrun --nproc_per_node=2 train.py --network densenet_161 --lr-scheduler cosine --batch-size 64 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### DPN

#### DPN: 92

```sh
torchrun --nproc_per_node=2 train.py --network dpn_92 --lr 0.316 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.9 --batch-size 64 --epochs 90 --wd 0.0001 --smoothing-alpha 0.1 --aug-level 4
```

#### DPN: 131

```sh
torchrun --nproc_per_node=2 train.py --network dpn_131 --lr 0.4 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.9 --batch-size 32 --epochs 90 --wd 0.0001 --smoothing-alpha 0.1 --aug-level 4
```

### EdgeNeXt

### EdgeNeXt: Extra Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_xxs --opt adamw --lr 0.006 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EdgeNeXt: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_xs --opt adamw --lr 0.006 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile
```

### EdgeNeXt: Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_s --opt adamw --lr 0.006 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### EdgeViT

#### EdgeViT: Extra Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_xxs --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 200 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul
```

#### EdgeViT: Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_xs --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 200 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul
```

#### EdgeViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_s --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 128 --epochs 200 --size 288 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_s --tag intermediate --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

### EfficientFormer v1

#### EfficientFormer v1: L1

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v1_l1 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile
```

#### EfficientFormer v1: L3

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v1_l3 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 5 --epochs 300 --size 288 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### EfficientFormer v1: L7

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v1_l7 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 20 --epochs 300 --size 288 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### EfficientFormer v2

Must increase resolution gradually due to the nature of the down-sampling layers.

#### EfficientFormer v2: S1

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v2_s1 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 288 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### EfficientNet Lite

#### EfficientNet Lite: 0

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_lite0 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile --rgb-mode none
```

### EfficientNet v1

#### EfficientNet v1: B0

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b0 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 128 --epochs 300 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EfficientNet v1: B3

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b3 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 288 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EfficientNet v1: B4

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b4 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### EfficientNet v1: B5

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b5 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 384 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### EfficientNet v2

#### EfficientNet v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --compile --stop-epoch 150
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 384 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 150 --load-states --stop-epoch 200
```

At epoch 260 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 448 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 260 --load-states
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag intermediate --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 128 --epochs 100 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag intermediate --lr 0.1 --lr-scheduler cosine --batch-size 128 --epochs 10 --size 448 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag intermediate --lr 0.0025 --lr-scheduler cosine --batch-size 64 --epochs 100 --size 448 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 10
```

#### EfficientNet v2: Medium

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag intermediate --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 64 --epochs 100 --size 256 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag intermediate --lr 0.1 --lr-scheduler cosine --batch-size 128 --epochs 10 --size 448 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag intermediate --lr 0.0025 --lr-scheduler cosine --batch-size 32 --epochs 80 --size 448 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 10
```

#### EfficientNet v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_l --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 384 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### FasterNet

#### FasterNet: T0

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_t0 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.005 --grad-accum-steps 2 --smoothing-alpha 0.1 --aug-level 3 --fast-matmul --compile
```

#### FasterNet: T1

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_t1 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.01 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --fast-matmul --compile
```

#### FasterNet: T2

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_t2 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.02 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --compile
```

#### FasterNet: Small

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_s --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.03 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --compile
```

#### FasterNet: Medium

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_m --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 64 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --clip-grad-norm 1 --amp --compile
```

#### FasterNet: Large

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_l --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 32 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --clip-grad-norm 0.1 --amp --compile
```

### FastViT

#### FastViT: T8

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_t8 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### FastViT: T12

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_t12 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### FastViT: SA12

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_sa12 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 128 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### FastViT: SA24

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_sa24 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 128 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### FocalNet

#### FocalNet: Tiny SRF

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_t_srf --opt adamw --lr 1e-3 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 64 --epochs 300 --size 288 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --rgb-mode none
```

#### FocalNet: Small LRF

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_s_lrf --opt adamw --lr 1e-3 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 64 --epochs 300 --size 320 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --rgb-mode none --stop-epoch 150
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_s_lrf --opt adamw --lr 1e-3 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 32 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --rgb-mode none --resume-epoch 150 --load-states --stop-epoch 220
```

At epoch 220 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_s_lrf --opt adamw --lr 1e-3 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 16 --epochs 300 --size 448 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --rgb-mode none --resume-epoch 220 --load-states
```

### GhostNet v1

#### GhostNet v1: 0.5 (50)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v1 --net-param 0.5 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 512 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### GhostNet v1: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v1 --net-param 1 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### GhostNet v1: 1.3 (130)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v1 --net-param 1.3 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### GhostNet v2

#### GhostNet v2: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2 --net-param 1 --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --warmup-epochs 3 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### GhostNet v2: 1.3 (130)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2 --net-param 1.3 --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --warmup-epochs 3 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

#### GhostNet v2: 1.6 (160)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2 --net-param 1.6 --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --warmup-epochs 3 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

### Hiera

#### Hiera: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network hiera_tiny --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### Hiera: AbsWin Small

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_small --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### HieraDet

Same as Hiera

### InceptionNeXt

#### InceptionNeXt: Small

```sh
torchrun --nproc_per_node=2 train.py --network inception_next_s --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

#### InceptionNeXt: Base

```sh
torchrun --nproc_per_node=2 train.py --network inception_next_b --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 288 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### Inception-ResNet v2

```sh
torchrun --nproc_per_node=2 train.py --network inception_resnet_v2 --lr-scheduler cosine --batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### Inception v3

```sh
torchrun --nproc_per_node=2 train.py --network inception_v3 --lr-scheduler cosine --batch-size 128 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### Inception v4

```sh
torchrun --nproc_per_node=2 train.py --network inception_v4 --lr-scheduler cosine --batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### LeViT

#### LeViT: 128s

```sh
torchrun --nproc_per_node=2 train.py --network levit_128s --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --warmup-epochs 5 --epochs 400 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --model-ema-decay 0.9998 --clip-grad-norm 1 --fast-matmul --compile
```

#### LeViT: 128

```sh
torchrun --nproc_per_node=2 train.py --network levit_128s --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 256 --warmup-epochs 5 --epochs 400 --size 256 --wd 0.025 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --model-ema-decay 0.9998 --clip-grad-norm 1 --amp --compile
```

#### LeViT: 256

```sh
torchrun --nproc_per_node=2 train.py --network levit_256 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 128 --warmup-epochs 5 --epochs 400 --size 256 --wd 0.025 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --model-ema-decay 0.9998 --clip-grad-norm 1 --amp --compile
```

### MaxViT

#### MaxViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_t --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 256 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 3 --model-ema --clip-grad-norm 1 --amp --compile --rgb-mode none
```

#### MaxViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_s --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 256 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile --rgb-mode none --stop-epoch 150
```

At epoch 150 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_s --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 16 --size 384 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile --rgb-mode none --resume-epoch 140 --load-scheduler --stop-epoch 180
```

At epoch 180 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_s --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 16 --size 448 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile --rgb-mode none --resume-epoch 180 --load-scheduler
```

#### MaxViT: Base

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_b --opt adamw --lr 0.0014 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 32 --size 288 --warmup-epochs 32 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --compile --rgb-mode none
```

### MetaFormer

#### MetaFormer: PoolFormer v1 s24

```sh
torchrun --nproc_per_node=2 train.py --network poolformer_v1_s24 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --size 256 --warmup-epochs 5 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --amp --compile
```

#### MetaFormer: PoolFormer v2 s24

```sh
torchrun --nproc_per_node=2 train.py --network poolformer_v2_s24 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --size 256 --warmup-epochs 5 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --amp --compile
```

#### MetaFormer: ConvFormer s18

```sh
torchrun --nproc_per_node=2 train.py --network convformer_s18 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --size 256 --warmup-epochs 20 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --amp --compile
```

#### MetaFormer: CAFormer s18

```sh
torchrun --nproc_per_node=2 train.py --network caformer_s18 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --size 256 --warmup-epochs 20 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --amp --compile --stop-epoch 225
```

At epoch 225 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network caformer_s18 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --size 320 --warmup-epochs 20 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --amp --compile --resume-epoch 225 --load-states --stop-epoch 250
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network caformer_s18 --opt adamw --lr 0.004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 384 --warmup-epochs 20 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --amp --compile --resume-epoch 250 --load-states
```

### MnasNet

#### MnasNet: 0.5

```sh
torchrun --nproc_per_node=2 train.py --network mnasnet --net-param 0.5 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 200 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### MnasNet: 1

```sh
torchrun --nproc_per_node=2 train.py --network mnasnet --net-param 1 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 200 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### Mobilenet v1

#### Mobilenet v1: Original

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1 --net-param 1 --opt rmsprop --lr 0.045 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.94 --batch-size 256 --aug-level 2
```

#### Mobilenet v1: v4 procedure

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1 --net-param 1 --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 600 --wd 0.01 --smoothing-alpha 0.1 --aug-level 3 --ra-sampler --ra-reps 2 --clip-grad-norm 5
```

### Mobilenet v2

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v2 --net-param 2 --opt rmsprop --lr 0.045 --lr-scheduler step --lr-step-size 1 --lr-step-gamma 0.98 --batch-size 128 --size 256 --epochs 300 --wd 0.00004 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --fast-matmul --compile
```

### Mobilenet v3 Large

#### Mobilenet v3 Large: 1.5

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large --net-param 1.5 --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --compile --stop-epoch 320
```

At epoch 320 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large --net-param 1.5 --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 128 --size 384 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --compile --resume-epoch 320 --load-states
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large --net-param 1.5 --tag intermediate --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

### Mobilenet v3 Small

#### Mobilenet v3 Small: 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_small --net-param 1 --opt rmsprop --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 400 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --fast-matmul --compile --stop-epoch 320
```

### Mobilenet v4

#### Mobilenet v4: Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_s --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 512 --size 256 --epochs 800 --wd 0.01 --smoothing-alpha 0.1 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5
```

#### Mobilenet v4: Medium

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_m --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 500 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --fast-matmul --compile
```

#### Mobilenet v4: Large

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_l --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 20 --batch-size 256 --size 256 --epochs 500 --wd 0.2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### Mobilenet v4 Hybrid

#### Mobilenet v4 Hybrid: Medium

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_hybrid_m --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 20 --batch-size 256 --size 256 --epochs 500 --wd 0.15 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Mobilenet v4 Hybrid: Large

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_hybrid_l --opt adamw --lr 0.003 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 20 --batch-size 256 --size 256 --epochs 500 --wd 0.2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### MobileOne

#### MobileOne: s0

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s0 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### MobileOne: s1

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s1 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 256 --epochs 300 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### MobileOne: s2

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s2 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 128 --epochs 300 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### MobileViT v1

#### MobileViT v1: Extra Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1_xxs --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 300 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### MobileViT v1: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1_xs --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 5 --batch-size 256 --size 256 --epochs 300 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --ra-sampler --ra-reps 2 --amp --compile
```

#### MobileViT v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1_s --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

### MobileViT v2

#### MobileViT v2: 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1 --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 20 --batch-size 256 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### MobileViT v2: 1.5

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1.5 --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 20 --batch-size 128 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --stop-epoch 260
```

At epoch 260 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1.5 --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 20 --batch-size 64 --size 384 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 260 --load-states
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1.5 --tag intermediate --opt adamw --lr 0.0003 --lr-scheduler cosine --lr-cosine-min 2e-5 --batch-size 128 --size 256 --epochs 80 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1.5 --tag intermediate --lr 0.001 --lr-scheduler cosine --batch-size 64 --size 384 --epochs 50 --wd 0.00004 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed --resume-epoch 0
```

#### MobileViT v2: 2

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 2 --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 20 --batch-size 128 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### MogaNet

#### MogaNet: X-Tiny

```sh
torchrun --nproc_per_node=2 train.py --network moganet_xt --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.03 --smoothing-alpha 0.1 --mixup-alpha 0.1 --cutmix --aug-level 3 --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network moganet_t --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.04 --smoothing-alpha 0.1 --mixup-alpha 0.1 --cutmix --aug-level 3 --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: Small

```sh
torchrun --nproc_per_node=2 train.py --network moganet_s --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: Base

```sh
torchrun --nproc_per_node=2 train.py --network moganet_b --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: Large

```sh
torchrun --nproc_per_node=2 train.py --network moganet_l --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: X-Large

```sh
torchrun --nproc_per_node=2 train.py --network moganet_xl --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 64 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### MViT v2

#### MViT v2: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_t --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 32 --warmup-epochs 70 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### MViT v2: Base

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 32 --warmup-epochs 70 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b --tag intermediate --opt adamw --lr 0.0000675 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 32 --warmup-epochs 20 --epochs 90 --size 256 --wd 0.01 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b --tag intermediate --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --fast-matmul --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b --opt adamw --lr 0.00007 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 32 --warmup-epochs 70 --epochs 40 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --resume-epoch 10
```

#### MViT v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_l --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 32 --warmup-epochs 70 --epochs 300 --size 256 --wd 0.1 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### MViT v2: Base w/cls token

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b_cls --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 32 --warmup-epochs 70 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### Next-ViT

#### Next-ViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_s --opt adamw --lr 0.00175 --lr-scheduler cosine --warmup-epochs 20 --batch-size 64 --size 256 --epochs 300 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --stop-epoch 250
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_s --opt adamw --lr 0.00175 --lr-scheduler cosine --warmup-epochs 20 --batch-size 32 --size 384 --epochs 300 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --resume-epoch 250 --load-states
```

#### Next-ViT: Base

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_b --opt adamw --lr 0.00175 --lr-scheduler cosine --warmup-epochs 20 --batch-size 64 --size 256 --epochs 300 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile --stop-epoch 250
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_b --opt adamw --lr 0.00175 --lr-scheduler cosine --warmup-epochs 20 --batch-size 32 --size 384 --epochs 300 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 250 --load-states
```

When finished, run 30 more epochs at increased resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_b --opt adamw --lr 5e-6 --lr-scheduler cosine --batch-size 32 --size 448 --epochs 330 --wd 1e-8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 300
```

### NFNet

#### NFNet: F0

```sh
torchrun --nproc_per_node=2 train.py --network nfnet_f0 --nesterov --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 5 --batch-size 128 --epochs 360 --size 256 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile --stop-epoch 250
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network nfnet_f0 --nesterov --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 5 --batch-size 128 --epochs 360 --size 384 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile --resume-epoch 250 --load-states
```

### PiT

#### PiT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network pit_t --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --clip-grad-norm 1 --fast-matmul
```

#### PiT: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network pit_xs --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --fast-matmul
```

#### PiT: Small

```sh
torchrun --nproc_per_node=2 train.py --network pit_s --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --clip-grad-norm 1 --amp
```

#### PiT: Base

```sh
torchrun --nproc_per_node=2 train.py --network pit_b --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### PVT v1

#### PVT v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v1_t --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v1_s --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v1: Medium

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v1_m --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### PVT v1: Large

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v1_l --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### PVT v2

#### PVT v2: B0

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b0 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### PVT v2: B1

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b1 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v2: B2 Linear

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b2_li --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b2_li --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 20 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 200 --load-states
```

#### PVT v2: B2

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b2 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v2: B3

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b3 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### PVT v2: B4

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b4 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### PVT v2: B5

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b5 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### RDNet

#### RDNet: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network rdnet_t --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

#### RDNet: Small

```sh
torchrun --nproc_per_node=2 train.py --network rdnet_s --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 128 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

#### RDNet: Base

```sh
torchrun --nproc_per_node=2 train.py --network rdnet_b --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 64 --epochs 300 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

### RegNet

#### RegNet: X 200 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_x_200m --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### RegNet: X 400 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_x_400m --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --ra-sampler --ra-reps 2
```

#### RegNet: Y 200 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_200m --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### RegNet: Y 400 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_400m --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --ra-sampler --ra-reps 2
```

#### RegNet: Y 600 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_600m --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --ra-sampler --ra-reps 2
```

#### RegNet: Y 800 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_800m --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2
```

#### RegNet: Y 1600 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_1600m --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2
```

#### RegNet: Y 8 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile --stop-epoch 70
```

At epoch 70 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 64 --size 384 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 70 --load-states --stop-epoch 90
```

At epoch 90 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 32 --size 448 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 90 --load-states
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag intermediate --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: at epoch 80 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag intermediate --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 64 --size 384 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --compile --resume-epoch 80 --load-states --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag intermediate --lr 0.1 --lr-scheduler cosine --batch-size 128 --epochs 10 --size 384 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --freeze-body
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag intermediate --lr 0.04 --lr-scheduler cosine --batch-size 32 --size 448 --epochs 30 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 10
```

#### RegNet: Y 16 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_16g --lr 0.2 --lr-scheduler cosine --warmup-epochs 5 --batch-size 64 --size 288 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_16g --tag intermediate --lr 0.2 --lr-scheduler cosine --warmup-epochs 5 --batch-size 64 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

### RepGhost

#### RepGhost: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network repghost --net-param 1 --lr 0.6 --lr-scheduler cosine --warmup-epochs 5 --batch-size 256 --size 256 --epochs 300 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --model-ema --model-ema-steps 1 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

### RepVgg

#### RepVgg: B1

```sh
torchrun --nproc_per_node=2 train.py --network repvgg_b1 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 5 --batch-size 128 --epochs 200 --size 256 --wd 0.0001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2
```

### ResMLP

#### ResMLP: 24

```sh
torchrun --nproc_per_node=2 train.py --network resmlp_24 --opt adamw --lr 0.005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### ResNeSt

#### ResNeSt: 14

```sh
torchrun --nproc_per_node=2 train.py --network resnest_14 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 270 --batch-size 256 --size 256 --wd 0.0001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

#### ResNeSt: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnest_50 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 270 --batch-size 64 --size 256 --wd 0.0001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

#### ResNeSt: 101

```sh
torchrun --nproc_per_node=2 train.py --network resnest_101 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 270 --batch-size 32 --size 256 --wd 0.0001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

### ResNet v2

#### ResNet v2: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --batch-size 256 --epochs 90 --smoothing-alpha 0.1 --aug-level 3
```

#### ResNet v2: 50, ResNet strikes back procedure

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 128 --epochs 200 --size 256 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2
```

### ResNeXt

#### ResNeXt: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnext_50 --lr-scheduler cosine --batch-size 64 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### ResNeXt: 101

```sh
torchrun --nproc_per_node=2 train.py --network resnext_101 --lr 0.04 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 64 --epochs 200 --size 256 --wd 0.001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --stop-epoch 150
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network resnext_101 --lr 0.04 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 32 --epochs 200 --size 384 --wd 0.001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --resume-epoch 150 --load-states
```

### SE ResNet v2

Same as ResNet v2

### SE ResNeXt

Same as ResNeXt

### Sequencer2d

#### Sequencer2d: Small

```sh
torchrun --nproc_per_node=2 train.py --network sequencer2d_s --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 20 --batch-size 128 --epochs 300 --size 252 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --stop-epoch 250
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network sequencer2d_s --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 20 --batch-size 64 --epochs 300 --size 392 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 250 --load-states
```

#### Sequencer2d: Medium

```sh
torchrun --nproc_per_node=2 train.py --network sequencer2d_m --opt adamw --lr 0.0015 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 20 --batch-size 128 --epochs 300 --size 252 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### Sequencer2d: Large

```sh
torchrun --nproc_per_node=2 train.py --network sequencer2d_l --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 20 --batch-size 64 --epochs 300 --size 252 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### ShuffleNet v1

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v1 --net-param 4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### ShuffleNet v2

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v2 --net-param 2 --lr 0.5 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --epochs 300 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

### Simple ViT

#### Simple ViT: b32

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit_b32 --opt adamw --lr 0.0004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 200 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### Simple ViT: b16

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit_b16 --opt adamw --lr 0.0004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 20 --epochs 200 --size 288 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### Simple ViT: l32

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit_l32 --opt adamw --lr 0.00025 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 20 --epochs 200 --size 320 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### Simple ViT: l16

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit_l16 --lr 0.3 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 16 --warmup-epochs 15 --epochs 400 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### SqueezeNet

```sh
torchrun --nproc_per_node=2 train.py --network squeezenet --lr 0.04 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.95 --batch-size 512 --wd 0.0002 --smoothing-alpha 0.1 --aug-level 2
```

### SqueezeNext

```sh
torchrun --nproc_per_node=2 train.py --network squeezenext --net-param 2 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.8 --batch-size 128 --wd 0.0001 --warmup-epochs 5 --epochs 120 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --fast-matmul --compile
```

### Swin Transformer v1

#### Swin Transformer v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1_t --opt adamw --lr 0.0004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --size 256 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1_s --opt adamw --lr 0.0004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 256 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v1: Base

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1_b --opt adamw --lr 0.0004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 32 --size 320 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v1: Large

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1_l --opt adamw --lr 0.0004 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 16 --size 384 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### Swin Transformer v2

#### Swin Transformer v2: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_t --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 256 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --size 256 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v2: Base

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_b --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 32 --size 256 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_l --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 16 --size 320 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### Tiny ViT

#### Tiny ViT: 5M

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

Optional intermediate training (suggested in the paper to use KD for this step)

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag intermediate --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 5 --epochs 90 --size 256 --wd 0.01 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --clip-grad-norm 5 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag intermediate --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --fast-matmul --compile --resume-epoch 0 --reset-head --freeze-body --unfreeze-features
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag intermediate --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --freeze-bn --batch-size 256 --warmup-epochs 5 --epochs 40 --size 256 --wd 1e-7 --norm-wd 0 --smoothing-alpha 0.1 --aug-level 4 --clip-grad-norm 5 --amp --compile --layer-decay 0.8 --resume-epoch 10
```

Optional intermediate training: increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag intermediate --opt adamw --lr 0.00004 --lr-scheduler cosine --lr-cosine-min 1e-7 --freeze-bn --batch-size 128 --warmup-epochs 5 --epochs 30 --size 384 --wd 1e-7 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --aug-level 4 --clip-grad-norm 5 --amp --compile --layer-decay 0.8 --resume-epoch 0
```

#### Tiny ViT: 11M

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_11m --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Tiny ViT: 21M

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_21m --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --aug-level 4 --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### UniFormer

#### UniFormer: Small

```sh
torchrun --nproc_per_node=2 train.py --network uniformer_s --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 64 --epochs 300 --size 320 --wd 0.05 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### VGG

```sh
torchrun --nproc_per_node=2 train.py --network vgg_13 --lr 0.01 --batch-size 128 --aug-level 2
```

### VGG Reduced

```sh
torchrun --nproc_per_node=2 train.py --network vgg_reduced_19 --lr 0.1 --batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### ViT

#### ViT: b32

```sh
torchrun --nproc_per_node=2 train.py --network vit_b32 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 30 --epochs 300 --size 224 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ViT: b16

```sh
torchrun --nproc_per_node=2 train.py --network vit_b16 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --warmup-epochs 30 --epochs 300 --size 288 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ViT: l32

```sh
torchrun --nproc_per_node=2 train.py --network vit_l32 --opt adamw --lr 0.0007 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 30 --epochs 300 --size 320 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ViT: l16

```sh
torchrun --nproc_per_node=2 train.py --network vit_l16 --lr 0.3 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 16 --warmup-epochs 10 --epochs 400 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ViT: h14

```sh
torchrun --nproc_per_node=2 train.py --network vit_h14 --lr 0.3 --lr-scheduler cosine --lr-cosine-min 1e-6 --batch-size 8 --warmup-epochs 10 --epochs 400 --size 336 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### ViT SAM

#### ViT SAM: b16

```sh
torchrun --nproc_per_node=2 train.py --network vit_sam_b16 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 20 --epochs 300 --size 256 --wd 0.1 --norm-wd 0 --grad-accum-steps 4 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### ViTReg

Same as ViT

### Wide ResNet

#### Wide ResNet: 50

```sh
torchrun --nproc_per_node=2 train.py --network wide_resnet_50 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 128 --epochs 300 --size 256 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

### Xception

```sh
torchrun --nproc_per_node=2 train.py --network xception --lr-scheduler cosine --batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### XCiT

#### XCiT: nano p16

```sh
torchrun --nproc_per_node=2 train.py --network xcit_nano12_p16 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --warmup-epochs 30 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --fast-matmul --compile
```

#### XCiT: small p8

```sh
torchrun --nproc_per_node=2 train.py --network xcit_small12_p8 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 30 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### XCiT: medium p16

```sh
torchrun --nproc_per_node=2 train.py --network xcit_medium24_p16 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 30 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --grad-accum-steps 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

## Common Dataset Training Scenarios

### ImageNet

#### ResNet v2: 50 ImageNet 1K example

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --tag imagenet1k --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --batch-size 256 --epochs 90 --smoothing-alpha 0.1 --aa --rgb-mode imagenet --fast-matmul --compile --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

#### ResNet v2: 50 ImageNet 1K example, ResNet strikes back procedure

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --tag imagenet1k --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 128 --epochs 200 --size 256 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --rgb-mode imagenet --fast-matmul --compile --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

### ImageNet 21K

#### ResNet v2: 50 ImageNet 21K example

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --tag imagenet21k --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --batch-size 256 --epochs 90 --smoothing-alpha 0.1 --aa --rgb-mode imagenet --fast-matmul --compile --wds --wds-class-file public_datasets_metadata/imagenet-21k-classes.txt --wds-train-size 13087061 --data-path ~/Datasets/imagenet-w21-webp-wds/training --wds-val-size 64215 --val-path ~/Datasets/imagenet-w21-webp-wds/validation
```
