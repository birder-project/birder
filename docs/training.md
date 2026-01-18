# Reference Training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

Examples use repo-root script names (e.g., `train.py`). If you installed Birder as a package, use the module form such as `python -m birder.scripts.train`.

## Table of Contents

- [General Training Guidelines](#general-training-guidelines)
- [Network Specific Training Procedures](#network-specific-training-procedures)
- [Common Dataset Training Scenarios](#common-dataset-training-scenarios)

## General Training Guidelines

It's best to avoid cutmix on compact models - <https://arxiv.org/abs/2404.11202v1>

Most networks train more effectively with growing resolution and augmentation as described at - <https://arxiv.org/abs/2104.00298>

## Network Specific Training Procedures

- [AlexNet](#alexnet)
- [BiFormer](#biformer)
- [CaiT](#cait)
- [CAS-ViT](#cas-vit)
- [CoaT](#coat)
- [Conv2Former](#conv2former)
- [ConvMixer](#convmixer)
- [ConvNeXt v1](#convnext-v1)
- [ConvNeXt v2](#convnext-v2)
- [CrossFormer](#crossformer)
- [CrossViT](#crossvit)
- [CSPNet](#cspnet)
- [CSWin Transformer](#cswin-transformer)
- [Darknet](#darknet)
- [DaViT](#davit)
- [DeiT](#deit)
- [DeiT3](#deit3)
- [DenseNet](#densenet)
- [DPN](#dpn)
- [EdgeNeXt](#edgenext)
- [EdgeViT](#edgevit)
- [EfficientFormer v1](#efficientformer-v1)
- [EfficientFormer v2](#efficientformer-v2)
- [EfficientNet Lite](#efficientnet-lite)
- [EfficientNet v1](#efficientnet-v1)
- [EfficientNet v2](#efficientnet-v2)
- [EfficientViM](#efficientvim)
- [EfficientViT MIT](#efficientvit-mit)
- [EfficientViT MSFT](#efficientvit-msft)
- [FasterNet](#fasternet)
- [FastViT](#fastvit)
- [FlexiViT](#flexivit)
- [FocalNet](#focalnet)
- [GC ViT](#gc-vit)
- [GhostNet v1](#ghostnet-v1)
- [GhostNet v2](#ghostnet-v2)
- [GroupMixFormer](#groupmixformer)
- [HGNet v1](#hgnet-v1)
- [HGNet v2](#hgnet-v2)
- [Hiera](#hiera)
- [HieraDet](#hieradet)
- [HorNet](#hornet)
- [iFormer](#iformer)
- [InceptionNeXt](#inceptionnext)
- [Inception-ResNet v1](#inception-resnet-v1)
- [Inception-ResNet v2](#inception-resnet-v2)
- [Inception v3](#inception-v3)
- [Inception v4](#inception-v4)
- [LeViT](#levit)
- [LIT v1](#lit-v1)
- [LIT v2](#lit-v2)
- [MaxViT](#maxvit)
- [MetaFormer](#metaformer)
- [MnasNet](#mnasnet)
- [Mobilenet v1](#mobilenet-v1)
- [Mobilenet v2](#mobilenet-v2)
- [Mobilenet v3 Large](#mobilenet-v3-large)
- [Mobilenet v3 Small](#mobilenet-v3-small)
- [Mobilenet v4](#mobilenet-v4)
- [Mobilenet v4 Hybrid](#mobilenet-v4-hybrid)
- [MobileOne](#mobileone)
- [MobileViT v1](#mobilevit-v1)
- [MobileViT v2](#mobilevit-v2)
- [MogaNet](#moganet)
- [MViT v2](#mvit-v2)
- [Next-ViT](#next-vit)
- [NFNet](#nfnet)
- [PiT](#pit)
- [PVT v1](#pvt-v1)
- [PVT v2](#pvt-v2)
- [RDNet](#rdnet)
- [RegionViT](#regionvit)
- [RegNet](#regnet)
- [RepGhost](#repghost)
- [RepVgg](#repvgg)
- [RepViT](#repvit)
- [ResMLP](#resmlp)
- [ResNeSt](#resnest)
- [ResNet v1](#resnet-v1)
- [ResNet v2](#resnet-v2)
- [ResNeXt](#resnext)
- [RoPE DeiT3](#rope-deit3)
- [RoPE FlexiViT](#rope-flexivit)
- [RoPE ViT](#rope-vit)
- [SE ResNet v1](#se-resnet-v1)
- [SE ResNet v2](#se-resnet-v2)
- [SE ResNeXt](#se-resnext)
- [Sequencer2d](#sequencer2d)
- [ShuffleNet v1](#shufflenet-v1)
- [ShuffleNet v2](#shufflenet-v2)
- [Simple ViT](#simple-vit)
- [SqueezeNet](#squeezenet)
- [SqueezeNext](#squeezenext)
- [StarNet](#starnet)
- [SwiftFormer](#swiftformer)
- [Swin Transformer v1](#swin-transformer-v1)
- [Swin Transformer v2](#swin-transformer-v2)
- [Tiny ViT](#tiny-vit)
- [TransNeXt](#transnext)
- [UniFormer](#uniformer)
- [VAN](#van)
- [VGG](#vgg)
- [VGG Reduced](#vgg-reduced)
- [ViT](#vit)
- [ViT Parallel](#vit-parallel)
- [ViT SAM](#vit-sam)
- [VoVNet v1](#vovnet-v1)
- [VoVNet v2](#vovnet-v2)
- [Wide ResNet](#wide-resnet)
- [Xception](#xception)
- [XCiT](#xcit)

### AlexNet

```sh
torchrun --nproc_per_node=2 train.py --network alexnet --batch-size 128 --lr 0.01 --aug-level 2
```

### BiFormer

#### BiFormer: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network biformer_t --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### BiFormer: Small

```sh
torchrun --nproc_per_node=2 train.py --network biformer_s --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### BiFormer: Base

```sh
torchrun --nproc_per_node=2 train.py --network biformer_b --batch-size 32 --opt adamw --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### CaiT

#### CaiT: Small 24

```sh
torchrun --nproc_per_node=2 train.py --network cait_s24 --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### CAS-ViT

#### CAS-ViT: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network cas_vit_xs --batch-size 384 --opt adamw --lr 0.006 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### CAS-ViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network cas_vit_s --batch-size 256 --opt adamw --lr 0.006 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 7 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### CAS-ViT: Mini

```sh
torchrun --nproc_per_node=2 train.py --network cas_vit_m --batch-size 128 --opt adamw --lr 0.006 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### CAS-ViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network cas_vit_t --batch-size 64 --opt adamw --lr 0.006 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### CoaT

#### CoaT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network coat_tiny --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp --compile
```

#### CoaT: Mini

```sh
torchrun --nproc_per_node=2 train.py --network coat_mini --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp --compile
```

#### CoaT: Small

```sh
torchrun --nproc_per_node=2 train.py --network coat_small --batch-size 32 --opt adamw --clip-grad-norm 5 --grad-accum-steps 8 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### CoaT: Lite Tiny

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_tiny --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul --compile
```

#### CoaT: Lite Mini

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_mini --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp --compile
```

#### CoaT: Lite Small

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_small --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp --compile
```

#### CoaT: Lite Medium

```sh
torchrun --nproc_per_node=2 train.py --network coat_lite_medium --batch-size 32 --opt adamw --clip-grad-norm 5 --grad-accum-steps 8 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### Conv2Former

#### Conv2Former: Nano

```sh
torchrun --nproc_per_node=2 train.py --network conv2former_n --batch-size 512 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### Conv2Former: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network conv2former_t --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### Conv2Former: Small

```sh
torchrun --nproc_per_node=2 train.py --network conv2former_s --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network conv2former_s --tag intermediate --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network conv2former_s --tag intermediate --reset-head --freeze-body --unfreeze-features --batch-size 256 --opt adamw --lr 0.0002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 10 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network conv2former_s --tag intermediate --batch-size 128 --opt adamw --lr 0.0001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 40 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 10
```

#### Conv2Former: Base

```sh
torchrun --nproc_per_node=2 train.py --network conv2former_b --batch-size 64 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

### ConvMixer

#### ConvMixer: 768 / 32

```sh
torchrun --nproc_per_node=2 train.py --network convmixer_768_32 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.5 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### ConvMixer: 1024 / 20

```sh
torchrun --nproc_per_node=2 train.py --network convmixer_1024_20 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 150 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.5 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### ConvMixer: 1536 / 20

```sh
torchrun --nproc_per_node=2 train.py --network convmixer_1536_20 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 150 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.5 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### ConvNeXt v1

#### ConvNeXt v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1_tiny --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 10 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp
```

#### ConvNeXt v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1_small --batch-size 32 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 10 --model-ema --size 320 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp
```

#### ConvNeXt v1: Base

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1_base --batch-size 16 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 10 --model-ema --size 384 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp
```

#### ConvNeXt v1: Large

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1_large --batch-size 16 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 10 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp
```

### ConvNeXt v2

#### ConvNeXt v2: Atto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_atto --batch-size 256 --opt adamw --lr 0.0002 --wd 0.3 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 600 --size 256 --aug-level 8 --smoothing-alpha 0.2 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### ConvNeXt v2: Femto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_femto --batch-size 256 --opt adamw --lr 0.0002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 600 --size 256 --aug-level 8 --smoothing-alpha 0.2 --mixup-alpha 0.3 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### ConvNeXt v2: Pico

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_pico --batch-size 128 --opt adamw --lr 0.0002 --wd 0.3 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 600 --size 288 --aug-level 8 --smoothing-alpha 0.2 --mixup-alpha 0.3 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### ConvNeXt v2: Nano

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_nano --batch-size 128 --opt adamw --lr 0.0002 --wd 0.3 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 600 --size 288 --aug-level 8 --smoothing-alpha 0.2 --mixup-alpha 0.5 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### ConvNeXt v2: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --batch-size 128 --opt adamw --lr 0.0008 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --stop-epoch 200 --warmup-epochs 40 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --batch-size 64 --opt adamw --lr 0.0008 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --stop-epoch 240 --warmup-epochs 40 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 200 --load-states
```

At epoch 240 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --batch-size 64 --opt adamw --lr 0.0008 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 40 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 240 --load-states
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag intermediate --batch-size 256 --opt adamw --lr 0.0008 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 100 --warmup-epochs 10 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag raw384px --batch-size 128 --opt adamw --lr 0.000025 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 40 --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --compile-opt --resume-epoch 0 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag intermediate --reset-head --freeze-body --unfreeze-features --batch-size 128 --opt adamw --lr 0.0002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 10 --size 448 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_tiny --tag intermediate --batch-size 64 --opt adamw --lr 0.0001 --wd 0.05 --norm-wd 0 --layer-decay 0.99 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 80 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 10
```

#### ConvNeXt v2: Base

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --batch-size 16 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag intermediate --batch-size 64 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: at epoch 80 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag intermediate --batch-size 64 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 10 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 80 --load-states --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag intermediate --reset-head --freeze-body --unfreeze-features --batch-size 128 --opt adamw --lr 0.0002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 10 --model-ema --size 384 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_base --tag intermediate --batch-size 32 --opt adamw --grad-accum-steps 4 --lr 0.0001 --wd 0.05 --norm-wd 0 --layer-decay 0.99 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 80 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 10
```

#### ConvNeXt v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_large --batch-size 16 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### ConvNeXt v2: Huge

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2_huge --batch-size 4 --opt adamw --lr 0.0008 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 10 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### CrossFormer

#### CrossFormer: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network crossformer_t --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### CrossFormer: Small

```sh
torchrun --nproc_per_node=2 train.py --network crossformer_s --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### CrossFormer: Base

```sh
torchrun --nproc_per_node=2 train.py --network crossformer_b --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### CrossFormer: Large

```sh
torchrun --nproc_per_node=2 train.py --network crossformer_l --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### CrossViT

#### CrossViT: 9 Dagger

```sh
torchrun --nproc_per_node=2 train.py --network crossvit_9d --batch-size 256 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.004 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 30 --model-ema --size 240 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp
```

#### CrossViT: 15 Dagger

```sh
torchrun --nproc_per_node=2 train.py --network crossvit_15d --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.004 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 30 --model-ema --size 336 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### CrossViT: 18

```sh
torchrun --nproc_per_node=2 train.py --network crossvit_18 --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 16 --lr 0.004 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 30 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### CSPNet

#### CSPNet: CSP ResNet 50

```sh
torchrun --nproc_per_node=2 train.py --network csp_resnet_50 --batch-size 128 --lr 0.1 --wd 0.0005 --lr-scheduler polynomial --lr-power 4 --epochs 90 --size 256 --aug-level 6 --smoothing-alpha 0.1 --fast-matmul --compile
```

#### CSPNet: CSP ResNeXt 50

```sh
torchrun --nproc_per_node=2 train.py --network csp_resnext_50 --batch-size 128 --lr 0.1 --wd 0.0005 --lr-scheduler polynomial --lr-power 4 --epochs 90 --size 256 --aug-level 6 --smoothing-alpha 0.1 --fast-matmul --compile
```

#### CSPNet: CSP Darknet 53

```sh
torchrun --nproc_per_node=2 train.py --network csp_darknet_53 --batch-size 128 --lr 0.1 --wd 0.0005 --lr-scheduler polynomial --lr-power 4 --epochs 90 --size 256 --aug-level 6 --smoothing-alpha 0.1 --fast-matmul --compile
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
torchrun --nproc_per_node=2 train.py --network cswin_transformer_t --batch-size 128 --opt adamw --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### CSWin Transformer: Small

```sh
torchrun --nproc_per_node=2 train.py --network cswin_transformer_s --batch-size 128 --opt adamw --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network cswin_transformer_s --batch-size 48 --opt adamw --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 200 --load-states
```

#### CSWin Transformer: Base

```sh
torchrun --nproc_per_node=2 train.py --network cswin_transformer_b --batch-size 32 --opt adamw --grad-accum-steps 8 --lr 0.001 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### CSWin Transformer: Large

```sh
torchrun --nproc_per_node=2 train.py --network cswin_transformer_l --batch-size 16 --opt adamw --grad-accum-steps 16 --lr 0.001 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

On intermediate training use wd of 0.2 (large only)

### Darknet

#### Darknet: 53

```sh
torchrun --nproc_per_node=2 train.py --network darknet_53 --batch-size 128 --lr 0.1 --wd 0.0005 --lr-scheduler cosine --epochs 90 --size 256 --aug-level 6 --smoothing-alpha 0.1 --fast-matmul --compile
```

### DaViT

#### DaViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network davit_tiny --batch-size 256 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --stop-epoch 220 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

At epoch 220 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network davit_tiny --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 16 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 220 --load-states
```

#### DaViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network davit_small --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network davit_small --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

### DeiT

#### DeiT: t16

```sh
torchrun --nproc_per_node=2 train.py --network deit_t16 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul --compile
```

#### DeiT: s16

```sh
torchrun --nproc_per_node=2 train.py --network deit_s16 --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp --compile
```

#### DeiT: b16

```sh
torchrun --nproc_per_node=2 train.py --network deit_b16 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --stop-epoch 200 --warmup-epochs 20 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network deit_b16 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.00015 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 200
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network deit_b16 --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

### DeiT3

#### DeiT3: t16

```sh
torchrun --nproc_per_node=2 train.py --network deit3_t16 --bce-loss --bce-threshold 0.05 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 600 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### DeiT3: s16

```sh
torchrun --nproc_per_node=2 train.py --network deit3_s16 --bce-loss --bce-threshold 0.05 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 600 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Fine-tuning, increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network deit3_s16 --model-config drop_path_rate=0.0 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.00001 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 20 --warmup-epochs 5 --model-ema --size 384 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

#### DeiT3: m16

```sh
torchrun --nproc_per_node=2 train.py --network deit3_m16 --bce-loss --bce-threshold 0.05 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.003 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Fine-tuning, increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network deit3_m16 --model-config drop_path_rate=0.0 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.00001 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 20 --warmup-epochs 5 --model-ema --size 384 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network deit3_m16 --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.02 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 240 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --smoothing-alpha 0.1 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network deit3_m16 --tag intermediate --batch-size 512 --opt adamw --clip-grad-norm 5 --lr 1e-4 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 10 --size 256 --aug-level 6 --smoothing-alpha 0.1 --amp --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network deit3_m16 --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.02 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 50 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --smoothing-alpha 0.1 --cutmix --amp --compile --resume-epoch 0
```

#### DeiT3: b16

```sh
torchrun --nproc_per_node=2 train.py --network deit3_b16 --bce-loss --bce-threshold 0.05 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.003 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 5 --model-ema --size 224 --aug-level 2 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Fine-tuning, increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network deit3_b16 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.00001 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 20 --warmup-epochs 5 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

Fine-tuning, increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network deit3_b16 --batch-size 32 --opt adamw --clip-grad-norm 1 --lr 0.00001 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 20 --warmup-epochs 5 --model-ema --size 384 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network deit3_b16 --tag intermediate --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.02 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 240 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --smoothing-alpha 0.1 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network deit3_b16 --tag intermediate --batch-size 512 --opt adamw --clip-grad-norm 5 --lr 1e-4 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 10 --size 256 --aug-level 6 --smoothing-alpha 0.1 --amp --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network deit3_b16 --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.02 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 50 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --smoothing-alpha 0.1 --cutmix --amp --compile --resume-epoch 0
```

#### DeiT3: l16

```sh
torchrun --nproc_per_node=2 train.py --network deit3_l16 --bce-loss --bce-threshold 0.05 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.003 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 5 --model-ema --size 192 --aug-level 2 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Fine-tuning, increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network deit3_l16 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.00001 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 20 --warmup-epochs 5 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

Fine-tuning, increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network deit3_l16 --batch-size 32 --opt adamw --clip-grad-norm 1 --lr 0.00001 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 20 --warmup-epochs 5 --model-ema --size 384 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

### DenseNet

#### DenseNet: 161

```sh
torchrun --nproc_per_node=2 train.py --network densenet_161 --batch-size 64 --lr-scheduler cosine --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### DPN

#### DPN: 92

```sh
torchrun --nproc_per_node=2 train.py --network dpn_92 --batch-size 64 --lr 0.316 --wd 0.0001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.9 --epochs 90 --aug-level 8 --smoothing-alpha 0.1
```

#### DPN: 131

```sh
torchrun --nproc_per_node=2 train.py --network dpn_131 --batch-size 32 --lr 0.4 --wd 0.0001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.9 --epochs 90 --aug-level 8 --smoothing-alpha 0.1
```

### EdgeNeXt

### EdgeNeXt: Extra Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_xxs --batch-size 256 --opt adamw --lr 0.006 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EdgeNeXt: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_xs --batch-size 256 --opt adamw --lr 0.006 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

### EdgeNeXt: Small

```sh
torchrun --nproc_per_node=2 train.py --network edgenext_s --batch-size 256 --opt adamw --lr 0.006 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

### EdgeViT

#### EdgeViT: Extra Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_xxs --batch-size 256 --opt adamw --lr 5e-4 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 200 --warmup-epochs 5 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --fast-matmul
```

#### EdgeViT: Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_xs --batch-size 256 --opt adamw --lr 5e-4 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 200 --warmup-epochs 5 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --fast-matmul
```

#### EdgeViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_s --batch-size 128 --opt adamw --lr 5e-4 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 200 --warmup-epochs 5 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --amp
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network edgevit_s --tag intermediate --batch-size 256 --opt adamw --lr 5e-4 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 100 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

### EfficientFormer v1

#### EfficientFormer v1: L1

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v1_l1 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EfficientFormer v1: L3

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v1_l3 --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --amp --compile
```

#### EfficientFormer v1: L7

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v1_l7 --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### EfficientFormer v2

Must increase resolution gradually due to the nature of the down-sampling layers.

#### EfficientFormer v2: S1

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v2_s1 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### EfficientFormer v2: S2

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v2_s2 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### EfficientFormer v2: L

```sh
torchrun --nproc_per_node=2 train.py --network efficientformer_v2_l --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### EfficientNet Lite

#### EfficientNet Lite: 0

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_lite0 --batch-size 256 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --rgb-mode none --ra-sampler --ra-reps 2 --fast-matmul --compile
```

### EfficientNet v1

#### EfficientNet v1: B0

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b0 --batch-size 128 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EfficientNet v1: B3

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b3 --batch-size 64 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EfficientNet v1: B4

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b4 --batch-size 32 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### EfficientNet v1: B5

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1_b5 --batch-size 16 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 384 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### EfficientNet v2

#### EfficientNet v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --batch-size 256 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --stop-epoch 100 --warmup-epochs 10 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 100 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --batch-size 96 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --stop-epoch 150 --warmup-epochs 10 --model-ema --size 320 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 100 --load-states
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --batch-size 64 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --stop-epoch 200 --warmup-epochs 10 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 150 --load-states
```

At epoch 260 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --batch-size 64 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 260 --load-states
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag intermediate --batch-size 128 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag intermediate --reset-head --freeze-body --unfreeze-features --batch-size 128 --lr 0.1 --lr-scheduler cosine --epochs 10 --size 448 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag intermediate --batch-size 64 --lr 0.0025 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --epochs 100 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 10
```

#### EfficientNet v2: Medium

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --batch-size 32 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag intermediate --batch-size 64 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 100 --warmup-epochs 10 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag intermediate --reset-head --freeze-body --unfreeze-features --batch-size 128 --lr 0.1 --lr-scheduler cosine --epochs 10 --size 448 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_m --tag intermediate --batch-size 32 --lr 0.0025 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --epochs 80 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 10
```

#### EfficientNet v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_l --batch-size 16 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### EfficientViM

#### EfficientViM: M1

```sh
torchrun --nproc_per_node=2 train.py --network efficientvim_m1 --batch-size 512 --opt adamw --clip-grad-norm 0.02 --lr 0.002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 300 --warmup-epochs 20 --model-ema --model-ema-steps 1 --model-ema-decay 0.9995 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile
```

When finished, run 30 more epochs at increased resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientvim_m1 --batch-size 256 --opt adamw --clip-grad-norm 0.02 --lr 0.001 --wd 1e-8 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 330 --model-ema --model-ema-steps 1 --model-ema-decay 0.9995 --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile --resume-epoch 300
```

#### EfficientViM: M3

```sh
torchrun --nproc_per_node=2 train.py --network efficientvim_m3 --batch-size 512 --opt adamw --clip-grad-norm 0.02 --lr 0.002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 300 --warmup-epochs 20 --model-ema --model-ema-steps 1 --model-ema-decay 0.9995 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile
```

When finished, run 30 more epochs at increased resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientvim_m3 --batch-size 256 --opt adamw --clip-grad-norm 0.02 --lr 0.001 --wd 1e-8 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 330 --model-ema --model-ema-steps 1 --model-ema-decay 0.9995 --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 300
```

### EfficientViT MIT

#### EfficientViT MIT: B0

```sh
torchrun --nproc_per_node=2 train.py --network efficientvit_mit_b0 --batch-size 256 --opt adamw --clip-grad-norm 2 --lr 0.00025 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EfficientViT MIT: B3

```sh
torchrun --nproc_per_node=2 train.py --network efficientvit_mit_b3 --batch-size 128 --opt adamw --clip-grad-norm 2 --lr 0.00025 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### EfficientViT MIT: L1

```sh
torchrun --nproc_per_node=2 train.py --network efficientvit_mit_l1 --batch-size 128 --opt adamw --clip-grad-norm 2 --lr 0.00015 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile
```

### EfficientViT MSFT

#### EfficientViT MSFT: M0

```sh
torchrun --nproc_per_node=2 train.py --network efficientvit_msft_m0 --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

### FasterNet

#### FasterNet: T0

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_t0 --batch-size 256 --opt adamw --grad-accum-steps 2 --lr 0.004 --wd 0.005 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 6 --smoothing-alpha 0.1 --fast-matmul --compile
```

#### FasterNet: T1

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_t1 --batch-size 256 --opt adamw --grad-accum-steps 2 --lr 0.004 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --fast-matmul --compile
```

#### FasterNet: T2

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_t2 --batch-size 128 --opt adamw --grad-accum-steps 4 --lr 0.004 --wd 0.02 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile
```

#### FasterNet: Small

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_s --batch-size 128 --opt adamw --grad-accum-steps 4 --lr 0.004 --wd 0.03 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile
```

#### FasterNet: Medium

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_m --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### FasterNet: Large

```sh
torchrun --nproc_per_node=2 train.py --network fasternet_l --batch-size 32 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 8 --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

### FastViT

#### FastViT: T8

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_t8 --batch-size 256 --opt adamw --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### FastViT: T12

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_t12 --batch-size 256 --opt adamw --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### FastViT: SA12

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_sa12 --batch-size 128 --opt adamw --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### FastViT: SA24

```sh
torchrun --nproc_per_node=2 train.py --network fastvit_sa24 --batch-size 128 --opt adamw --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### FastViT: MobileClip v1 i0

```sh
torchrun --nproc_per_node=2 train.py --network mobileclip_v1_i0 --batch-size 256 --opt adamw --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### FastViT: MobileClip v1 i2

```sh
torchrun --nproc_per_node=2 train.py --network mobileclip_v1_i2 --batch-size 128 --opt adamw --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### FastViT: MobileClip v2 i3

```sh
torchrun --nproc_per_node=2 train.py --network mobileclip_v2_i3 --batch-size 64 --opt adamw --grad-accum-steps 8 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### FlexiViT

Same as [ViT](#vit)

### FocalNet

#### FocalNet: Tiny SRF

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_t_srf --batch-size 64 --opt adamw --clip-grad-norm 5 --lr 1e-3 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 288 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp --compile
```

#### FocalNet: Small LRF

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_s_lrf --batch-size 64 --opt adamw --clip-grad-norm 5 --lr 1e-3 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --stop-epoch 150 --warmup-epochs 20 --model-ema --size 320 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_s_lrf --batch-size 32 --opt adamw --clip-grad-norm 5 --lr 1e-3 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --stop-epoch 220 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 150 --load-states
```

At epoch 220 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_s_lrf --batch-size 16 --opt adamw --clip-grad-norm 5 --lr 1e-3 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 220 --load-states
```

#### FocalNet: Base LRF

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_b_lrf --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 1e-3 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --ra-sampler --ra-reps 2 --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_b_lrf --tag intermediate --model-config drop_path_rate=0.2 --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 1e-3 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 90 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network focalnet_b_lrf --tag intermediate --reset-head --freeze-body --batch-size 512 --opt adamw --clip-grad-norm 5 --lr 1e-4 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 10 --size 256 --aug-level 6 --smoothing-alpha 0.1 --rgb-mode none --amp --compile --resume-epoch 0
```

### GC ViT

#### GC ViT: XX-Tiny

```sh
torchrun --nproc_per_node=2 train.py --network gc_vit_xxt --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.005 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 310 --warmup-epochs 20 --cooldown-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul --compile
```

#### GC ViT: X-Tiny

```sh
torchrun --nproc_per_node=2 train.py --network gc_vit_xt --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.005 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 310 --warmup-epochs 20 --cooldown-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul --compile
```

#### GC ViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network gc_vit_t --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.005 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 310 --warmup-epochs 20 --cooldown-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul --compile
```

#### GC ViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network gc_vit_s --batch-size 32 --opt adamw --clip-grad-norm 5 --lr 0.005 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 310 --warmup-epochs 20 --cooldown-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul --compile
```

### GhostNet v1

#### GhostNet v1: 0.5 (50)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v1_0_5 --batch-size 512 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

#### GhostNet v1: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v1_1_0 --batch-size 256 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

#### GhostNet v1: 1.3 (130)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v1_1_3 --batch-size 256 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### GhostNet v2

#### GhostNet v2: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2_1_0 --batch-size 256 --opt rmsprop --lr 0.064 --wd 0.00001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --epochs 400 --warmup-epochs 3 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### GhostNet v2: 1.3 (130)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2_1_3 --batch-size 256 --opt rmsprop --lr 0.064 --wd 0.00001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --epochs 400 --warmup-epochs 3 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

#### GhostNet v2: 1.6 (160)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2_1_6 --batch-size 256 --opt rmsprop --lr 0.064 --wd 0.00001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --epochs 400 --warmup-epochs 3 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

### GroupMixFormer

#### GroupMixFormer: Mobile

```sh
torchrun --nproc_per_node=2 train.py --network groupmixformer_mobile --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### GroupMixFormer: Small

```sh
torchrun --nproc_per_node=2 train.py --network groupmixformer_s --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### GroupMixFormer: Base

```sh
torchrun --nproc_per_node=2 train.py --network groupmixformer_b --batch-size 64 --opt adamw --clip-grad-norm 5 --grad-accum-steps 2 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

### HGNet v1

#### HGNet v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network hgnet_v1_tiny --batch-size 128 --lr 0.5 --wd 0.00004 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 400 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --fast-matmul --compile
```

#### HGNet v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network hgnet_v1_small --batch-size 128 --lr 0.5 --wd 0.00004 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 400 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile
```

#### HGNet v1: Base

```sh
torchrun --nproc_per_node=2 train.py --network hgnet_v1_base --batch-size 128 --lr 0.5 --wd 0.00004 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 400 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.4 --amp --compile
```

### HGNet v2

#### HGNet v2: B0

```sh
torchrun --nproc_per_node=2 train.py --network hgnet_v2_b0 --batch-size 128 --lr 0.125 --wd 0.00004 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 400 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --fast-matmul --compile
```

#### HGNet v2: B1

```sh
torchrun --nproc_per_node=2 train.py --network hgnet_v2_b1 --batch-size 128 --lr 0.125 --wd 0.00004 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 400 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile
```

### Hiera

#### Hiera: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network hiera_tiny --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### Hiera: AbsWin Small

```sh
torchrun --nproc_per_node=2 train.py --network hiera_abswin_small --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### HieraDet

Same as Hiera

### HorNet

#### HorNet: Tiny 7x7

```sh
torchrun --nproc_per_node=2 train.py --network hornet_tiny_7x7 --batch-size 128 --opt adamw --clip-grad-norm 10 --grad-accum-steps 2 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### HorNet: Tiny GF

```sh
torchrun --nproc_per_node=2 train.py --network hornet_tiny_gf --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp
```

#### HorNet: Small 7x7

```sh
torchrun --nproc_per_node=2 train.py --network hornet_small_7x7 --batch-size 64 --opt adamw --clip-grad-norm 5 --grad-accum-steps 4 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### HorNet: Small GF

```sh
torchrun --nproc_per_node=2 train.py --network hornet_small_gf --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp
```

#### HorNet: Base GF

```sh
torchrun --nproc_per_node=2 train.py --network hornet_base_gf --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network hornet_base_gf --tag intermediate --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.002 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile-opt --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network hornet_base_gf --tag intermediate --reset-head --freeze-body --unfreeze-features --batch-size 128 --opt adamw --lr 0.0002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 10 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --resume-epoch 0
```

### iFormer

#### iFormer: Small

```sh
torchrun --nproc_per_node=2 train.py --network iformer_s --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### iFormer: Base

```sh
torchrun --nproc_per_node=2 train.py --network iformer_b --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### iFormer: Large

```sh
torchrun --nproc_per_node=2 train.py --network iformer_l --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### InceptionNeXt

#### InceptionNeXt: Atto

```sh
torchrun --nproc_per_node=2 train.py --network inception_next_a --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 450 --warmup-epochs 5 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --fast-matmul --compile
```

#### InceptionNeXt: Small

```sh
torchrun --nproc_per_node=2 train.py --network inception_next_s --batch-size 64 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### InceptionNeXt: Base

```sh
torchrun --nproc_per_node=2 train.py --network inception_next_b --batch-size 32 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 300 --warmup-epochs 20 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

### Inception-ResNet v1

```sh
torchrun --nproc_per_node=2 train.py --network inception_resnet_v1 --batch-size 128 --lr-scheduler cosine --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### Inception-ResNet v2

```sh
torchrun --nproc_per_node=2 train.py --network inception_resnet_v2 --batch-size 64 --lr-scheduler cosine --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### Inception v3

```sh
torchrun --nproc_per_node=2 train.py --network inception_v3 --batch-size 128 --lr-scheduler cosine --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### Inception v4

```sh
torchrun --nproc_per_node=2 train.py --network inception_v4 --batch-size 64 --lr-scheduler cosine --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### LeViT

#### LeViT: 128s

```sh
torchrun --nproc_per_node=2 train.py --network levit_128s --batch-size 512 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 400 --warmup-epochs 5 --model-ema --model-ema-decay 0.9998 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile
```

#### LeViT: 128

```sh
torchrun --nproc_per_node=2 train.py --network levit_128 --batch-size 512 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 400 --warmup-epochs 5 --model-ema --model-ema-decay 0.9998 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile
```

#### LeViT: 256

```sh
torchrun --nproc_per_node=2 train.py --network levit_256 --batch-size 256 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 400 --warmup-epochs 5 --model-ema --model-ema-decay 0.9998 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

### LIT v1

#### LIT v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network lit_v1_s --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.0005 --wd 0.05 --custom-layer-wd offset_conv=0.0 --custom-layer-lr-scale offset_conv=0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile
```

### LIT v2

#### LIT v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network lit_v2_s --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.0005 --wd 0.05 --custom-layer-wd offset_conv=0.0 --custom-layer-lr-scale offset_conv=0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile
```

### MaxViT

#### MaxViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_t --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.003 --wd 0.05 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile
```

#### MaxViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_s --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.003 --wd 0.05 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --stop-epoch 150 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile
```

At epoch 150 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_s --batch-size 16 --opt adamw --clip-grad-norm 1 --lr 0.003 --wd 0.05 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --stop-epoch 180 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile --resume-epoch 140 --load-scheduler
```

At epoch 180 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_s --batch-size 16 --opt adamw --clip-grad-norm 1 --lr 0.003 --wd 0.05 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile --resume-epoch 180 --load-scheduler
```

#### MaxViT: Base

```sh
torchrun --nproc_per_node=2 train.py --network maxvit_b --batch-size 32 --opt adamw --clip-grad-norm 1 --lr 0.0014 --wd 0.05 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 32 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --rgb-mode none --amp --compile
```

### MetaFormer

#### MetaFormer: PoolFormer v1 s24

```sh
torchrun --nproc_per_node=2 train.py --network poolformer_v1_s24 --batch-size 128 --opt adamw --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### MetaFormer: PoolFormer v2 s24

```sh
torchrun --nproc_per_node=2 train.py --network poolformer_v2_s24 --batch-size 128 --opt adamw --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### MetaFormer: ConvFormer s18

```sh
torchrun --nproc_per_node=2 train.py --network convformer_s18 --batch-size 128 --opt adamw --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### MetaFormer: CAFormer s18

```sh
torchrun --nproc_per_node=2 train.py --network caformer_s18 --batch-size 256 --opt adamw --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --stop-epoch 225 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

At epoch 225 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network caformer_s18 --batch-size 128 --opt adamw --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --stop-epoch 250 --warmup-epochs 20 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 225 --load-states
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network caformer_s18 --batch-size 64 --opt adamw --lr 0.004 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 250 --load-states
```

### MnasNet

#### MnasNet: 0.5

```sh
torchrun --nproc_per_node=2 train.py --network mnasnet_0_5 --batch-size 256 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 200 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

#### MnasNet: 1

```sh
torchrun --nproc_per_node=2 train.py --network mnasnet_1_0 --batch-size 256 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 200 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### Mobilenet v1

#### Mobilenet v1: Original

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1_1_0 --batch-size 256 --opt rmsprop --lr 0.045 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.94 --aug-level 2
```

#### Mobilenet v1: v4 procedure

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1_1_0 --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.002 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 600 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2
```

### Mobilenet v2

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v2_2_0 --batch-size 128 --opt rmsprop --lr 0.045 --wd 0.00004 --lr-scheduler step --lr-step-size 1 --lr-step-gamma 0.98 --epochs 300 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --fast-matmul --compile
```

### Mobilenet v3

#### Mobilenet v3: Small 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_small_1_0 --batch-size 256 --opt rmsprop --lr 0.064 --wd 0.00001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --epochs 400 --stop-epoch 320 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --fast-matmul --compile
```

#### Mobilenet v3: Large 1.5

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large_1_5 --batch-size 256 --opt rmsprop --lr 0.064 --wd 0.00001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --epochs 400 --stop-epoch 320 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile
```

At epoch 320 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large_1_5 --batch-size 128 --opt rmsprop --lr 0.064 --wd 0.00001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --epochs 400 --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile --resume-epoch 320 --load-states
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3_large_1_5 --tag intermediate --batch-size 256 --opt rmsprop --lr 0.064 --wd 0.00001 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --epochs 400 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

### Mobilenet v4

#### Mobilenet v4: Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_s --batch-size 512 --opt adamw --clip-grad-norm 5 --lr 0.002 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 800 --warmup-epochs 5 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2
```

#### Mobilenet v4: Medium

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_m --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.003 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 500 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul --compile
```

#### Mobilenet v4: Large

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_l --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.00225 --wd 0.2 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 500 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

### Mobilenet v4 Hybrid

#### Mobilenet v4 Hybrid: Medium

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_hybrid_m --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.002 --wd 0.15 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 500 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp --compile
```

#### Mobilenet v4 Hybrid: Large

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v4_hybrid_l --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.0025 --wd 0.2 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 500 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --amp-dtype bfloat16 --compile
```

### MobileOne

#### MobileOne: s0

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s0 --batch-size 256 --lr 0.1 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### MobileOne: s1

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s1 --batch-size 256 --lr 0.1 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### MobileOne: s2

```sh
torchrun --nproc_per_node=2 train.py --network mobileone_s2 --batch-size 128 --lr 0.1 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 5e-6 --epochs 300 --warmup-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### MobileViT v1

#### MobileViT v1: Extra Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1_xxs --batch-size 256 --opt adamw --lr 0.002 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 2e-5 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### MobileViT v1: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1_xs --batch-size 256 --opt adamw --lr 0.002 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 2e-5 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

#### MobileViT v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1_s --batch-size 128 --opt adamw --lr 0.002 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 2e-5 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

### MobileViT v2

#### MobileViT v2: 1

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2_1_0 --batch-size 256 --opt adamw --lr 0.002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 2e-5 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

#### MobileViT v2: 1.5

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2_1_5 --batch-size 128 --opt adamw --lr 0.002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 2e-5 --epochs 300 --stop-epoch 260 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 260 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2_1_5 --batch-size 64 --opt adamw --lr 0.002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 2e-5 --epochs 300 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 260 --load-states
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2_1_5 --tag intermediate --batch-size 128 --opt adamw --lr 0.0003 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 2e-5 --epochs 80 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

#### MobileViT v2: 2

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2_2_0 --batch-size 128 --opt adamw --lr 0.002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 2e-5 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### MogaNet

#### MogaNet: X-Tiny

```sh
torchrun --nproc_per_node=2 train.py --network moganet_xt --batch-size 128 --opt adamw --lr 0.001 --wd 0.03 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.1 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network moganet_t --batch-size 128 --opt adamw --lr 0.001 --wd 0.04 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.1 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: Small

```sh
torchrun --nproc_per_node=2 train.py --network moganet_s --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: Base

```sh
torchrun --nproc_per_node=2 train.py --network moganet_b --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: Large

```sh
torchrun --nproc_per_node=2 train.py --network moganet_l --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### MogaNet: X-Large

```sh
torchrun --nproc_per_node=2 train.py --network moganet_xl --batch-size 64 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-5 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### MViT v2

#### MViT v2: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_t --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --stop-epoch 200 --warmup-epochs 70 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_t --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 70 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile --resume-epoch 200 --load-scheduler
```

#### MViT v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_s --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 70 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile
```

#### MViT v2: Base

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 70 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b --tag intermediate --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0000675 --wd 0.01 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 90 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b --tag intermediate --reset-head --freeze-body --unfreeze-features --batch-size 256 --opt adamw --lr 0.001 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 10 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.00007 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 40 --warmup-epochs 70 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile --resume-epoch 10
```

#### MViT v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_l --batch-size 32 --opt adamw --clip-grad-norm 1 --lr 0.002 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 70 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile
```

#### MViT v2: Base w/cls token

```sh
torchrun --nproc_per_node=2 train.py --network mvit_v2_b_cls --batch-size 32 --opt adamw --clip-grad-norm 1 --lr 0.002 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 70 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile
```

### Next-ViT

#### Next-ViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_s --batch-size 128 --opt adamw --lr 0.00175 --wd 0.1 --lr-scheduler cosine --epochs 300 --stop-epoch 250 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_s --batch-size 32 --opt adamw --lr 0.00175 --wd 0.1 --lr-scheduler cosine --epochs 300 --warmup-epochs 20 --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --amp --compile --resume-epoch 250 --load-states
```

#### Next-ViT: Base

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_b --batch-size 64 --opt adamw --lr 0.00175 --wd 0.1 --lr-scheduler cosine --epochs 300 --stop-epoch 250 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_b --batch-size 32 --opt adamw --lr 0.00175 --wd 0.1 --lr-scheduler cosine --epochs 300 --warmup-epochs 20 --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 250 --load-states
```

When finished, run 30 more epochs at increased resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit_b --batch-size 32 --opt adamw --lr 5e-6 --wd 1e-8 --lr-scheduler cosine --epochs 330 --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 300
```

### NFNet

#### NFNet: F0

```sh
torchrun --nproc_per_node=2 train.py --network nfnet_f0 --batch-size 128 --nesterov --lr 0.1 --wd 0.00002 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 360 --stop-epoch 250 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network nfnet_f0 --batch-size 128 --nesterov --lr 0.1 --wd 0.00002 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 360 --warmup-epochs 5 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --amp-dtype bfloat16 --compile --resume-epoch 250 --load-states
```

### PiT

#### PiT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network pit_t --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul
```

#### PiT: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network pit_xs --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul
```

#### PiT: Small

```sh
torchrun --nproc_per_node=2 train.py --network pit_s --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --amp
```

#### PiT: Base

```sh
torchrun --nproc_per_node=2 train.py --network pit_b --batch-size 128 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### PVT v1

#### PVT v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v1_t --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v1_s --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v1: Medium

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v1_m --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v1: Large

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v1_l --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### PVT v2

#### PVT v2: B0

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b0 --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### PVT v2: B1

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b1 --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v2: B2 Linear

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b2_li --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b2_li --batch-size 64 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 200 --load-states
```

#### PVT v2: B2

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b2 --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v2: B3

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b3 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v2: B4

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b4 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### PVT v2: B5

```sh
torchrun --nproc_per_node=2 train.py --network pvt_v2_b5 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### RDNet

#### RDNet: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network rdnet_t --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### RDNet: Small

```sh
torchrun --nproc_per_node=2 train.py --network rdnet_s --batch-size 128 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### RDNet: Base

```sh
torchrun --nproc_per_node=2 train.py --network rdnet_b --batch-size 64 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### RegionViT

#### RegionViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network regionvit_t --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 50 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### RegionViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network regionvit_s --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 50 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 3 --amp --compile
```

#### RegionViT: Medium

```sh
torchrun --nproc_per_node=2 train.py --network regionvit_m --batch-size 128 --opt adamw --grad-accum-steps 2 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 50 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 3 --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network regionvit_m --tag intermediate --batch-size 128 --opt adamw --grad-accum-steps 2 --lr 0.001 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 120 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regionvit_m --tag intermediate --batch-size 512 --opt adamw --lr 1e-4 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 10 --size 256 --aug-level 6 --smoothing-alpha 0.1 --amp --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network regionvit_m --tag intermediate --batch-size 128 --opt adamw --grad-accum-steps 2 --lr 0.001 --wd 1e-8 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 30 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 3 --amp --compile --resume-epoch 0
```

#### RegionViT: Base

```sh
torchrun --nproc_per_node=2 train.py --network regionvit_b --batch-size 64 --opt adamw --grad-accum-steps 4 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 50 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 3 --amp --compile
```

### RegNet

#### RegNet: X 200 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_x_200m --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

#### RegNet: X 400 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_x_400m --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2
```

#### RegNet: Y 200 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_200m --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

#### RegNet: Y 400 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_400m --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2
```

#### RegNet: Y 600 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_600m --batch-size 128 --lr 0.8 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2
```

#### RegNet: Y 800 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_800m --batch-size 128 --lr 0.8 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2
```

#### RegNet: Y 1.6 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_1_6g --batch-size 256 --lr 0.8 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

#### RegNet: Y 4 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_4g --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### RegNet: Y 8 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --stop-epoch 70 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 70 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --batch-size 64 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --stop-epoch 90 --warmup-epochs 5 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 70 --load-states
```

At epoch 90 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --batch-size 32 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 90 --load-states
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag intermediate --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: at epoch 80 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag intermediate --batch-size 64 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile --resume-epoch 80 --load-states --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag intermediate --reset-head --freeze-body --batch-size 128 --lr 0.1 --lr-scheduler cosine --epochs 10 --size 384 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_8g --tag intermediate --batch-size 32 --lr 0.04 --wd 0.00005 --lr-scheduler cosine --epochs 30 --model-ema --size 448 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 10
```

#### RegNet: Y 16 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_16g --batch-size 64 --lr 0.2 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network regnet_y_16g --tag intermediate --batch-size 64 --lr 0.2 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

#### RegNet: Z 500 MF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_z_500m --batch-size 128 --lr 0.8 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2
```

#### RegNet: Z 4 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet_z_4g --batch-size 128 --lr 0.4 --wd 0.00005 --lr-scheduler cosine --epochs 100 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### RepGhost

#### RepGhost: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network repghost_1_0 --batch-size 256 --lr 0.6 --wd 0.00001 --lr-scheduler cosine --epochs 300 --warmup-epochs 5 --model-ema --model-ema-steps 1 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

### RepVgg

#### RepVgg: B1

```sh
torchrun --nproc_per_node=2 train.py --network repvgg_b1 --batch-size 128 --lr 0.1 --wd 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2
```

### RepViT

#### RepViT: M0.9

```sh
torchrun --nproc_per_node=2 train.py --network repvit_m0_9 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --fast-matmul --compile
```

### ResMLP

#### ResMLP: 24

```sh
torchrun --nproc_per_node=2 train.py --network resmlp_24 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.005 --wd 0.2 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### ResNeSt

#### ResNeSt: 14

```sh
torchrun --nproc_per_node=2 train.py --network resnest_14 --batch-size 256 --lr 0.1 --wd 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 270 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

#### ResNeSt: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnest_50 --batch-size 64 --lr 0.1 --wd 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 270 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2 --amp --compile
```

#### ResNeSt: 101

```sh
torchrun --nproc_per_node=2 train.py --network resnest_101 --batch-size 32 --lr 0.1 --wd 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 270 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### ResNet v1

#### ResNet v1: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v1_50 --batch-size 256 --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --epochs 90 --aug-level 6 --smoothing-alpha 0.1
```

### ResNet v2

#### ResNet v2: 50

Same as ResNet v1

#### ResNet v2: 50, ResNet strikes back procedure (A2)

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --bce-loss --bce-threshold 0.2 --batch-size 256 --opt lamb --grad-accum-steps 4 --lr 0.005 --wd 0.02 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --mixup-alpha 0.1 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### ResNeXt

#### ResNeXt: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnext_50 --batch-size 64 --lr-scheduler cosine --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

#### ResNeXt: 101

```sh
torchrun --nproc_per_node=2 train.py --network resnext_101 --batch-size 64 --lr 0.04 --wd 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --stop-epoch 150 --warmup-epochs 10 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network resnext_101 --batch-size 32 --lr 0.04 --wd 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 10 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --resume-epoch 150 --load-states
```

### RoPE DeiT3

Same as [DeiT3](#deit3)

### RoPE FlexiViT

Same as [FlexiViT](#flexivit)

### RoPE ViT

Same as [ViT](#vit)

### SE ResNet v1

Same as ResNet v1

### SE ResNet v2

Same as ResNet v2

### SE ResNeXt

Same as ResNeXt

### Sequencer2d

#### Sequencer2d: Small

```sh
torchrun --nproc_per_node=2 train.py --network sequencer2d_s --batch-size 128 --opt adamw --lr 0.002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --stop-epoch 250 --warmup-epochs 20 --model-ema --size 252 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network sequencer2d_s --batch-size 64 --opt adamw --lr 0.002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --model-ema --size 392 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 250 --load-states
```

#### Sequencer2d: Medium

```sh
torchrun --nproc_per_node=2 train.py --network sequencer2d_m --batch-size 128 --opt adamw --lr 0.0015 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --model-ema --size 252 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### Sequencer2d: Large

```sh
torchrun --nproc_per_node=2 train.py --network sequencer2d_l --batch-size 64 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 20 --model-ema --size 252 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### ShuffleNet v1

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v1_4 --batch-size 256 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### ShuffleNet v2

```sh
torchrun --nproc_per_node=2 train.py --network shufflenet_v2_2_0 --batch-size 128 --lr 0.5 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --epochs 300 --warmup-epochs 5 --model-ema --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --ra-sampler --ra-reps 2
```

### Simple ViT

#### Simple ViT: b32

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit_b32 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0004 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### Simple ViT: b16

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit_b16 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.0004 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### Simple ViT: l32

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit_l32 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.00025 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 320 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### Simple ViT: l16

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit_l16 --batch-size 16 --clip-grad-norm 1 --lr 0.3 --wd 0.00002 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 400 --warmup-epochs 15 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### SMT

### SMT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network smt_t --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile
```

### SMT: Small

```sh
torchrun --nproc_per_node=2 train.py --network smt_s --batch-size 128 --opt adamw --clip-grad-norm 5 --grad-accum-steps 2 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network smt_s --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 5 --grad-accum-steps 2 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 90 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile --save-frequency 1 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network smt_s --tag intermediate --reset-head --freeze-body --batch-size 512 --opt adamw --lr 0.001 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 10 --size 256 --aug-level 2 --smoothing-alpha 0.1 --amp --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network smt_s --tag intermediate --model-config drop_path_rate=0.1 --batch-size 128 --opt adamw --clip-grad-norm 5 --grad-accum-steps 2 --lr 0.00002 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 40 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --amp --compile --resume-epoch 10
```

### SMT: Base

```sh
torchrun --nproc_per_node=2 train.py --network smt_b --batch-size 64 --opt adamw --clip-grad-norm 5 --grad-accum-steps 4 --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --amp --compile
```

### SqueezeNet

```sh
torchrun --nproc_per_node=2 train.py --network squeezenet --batch-size 512 --lr 0.04 --wd 0.0002 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.95 --aug-level 2 --smoothing-alpha 0.1
```

### SqueezeNext

```sh
torchrun --nproc_per_node=2 train.py --network squeezenext_2_0 --batch-size 128 --lr 0.4 --wd 0.0001 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.8 --epochs 120 --warmup-epochs 5 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --fast-matmul --compile
```

### StarNet

#### StarNet: ESM05

```sh
torchrun --nproc_per_node=2 train.py --network starnet_esm05 --batch-size 512 --opt adamw --lr 0.003 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul
```

#### StarNet: ESM10

```sh
torchrun --nproc_per_node=2 train.py --network starnet_esm10 --batch-size 512 --opt adamw --lr 0.003 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul
```

#### StarNet: S1

```sh
torchrun --nproc_per_node=2 train.py --network starnet_s1 --batch-size 512 --opt adamw --lr 0.003 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile
```

#### StarNet: S4

```sh
torchrun --nproc_per_node=2 train.py --network starnet_s4 --batch-size 512 --opt adamw --lr 0.003 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile
```

### SwiftFormer

#### SwiftFormer: Extra Small

```sh
torchrun --nproc_per_node=2 train.py --network swiftformer_xs --batch-size 256 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 2 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### SwiftFormer: Small

```sh
torchrun --nproc_per_node=2 train.py --network swiftformer_s --batch-size 256 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 2 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --amp --compile
```

#### SwiftFormer: L1

```sh
torchrun --nproc_per_node=2 train.py --network swiftformer_l1 --batch-size 128 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 4 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### SwiftFormer: L3

```sh
torchrun --nproc_per_node=2 train.py --network swiftformer_l3 --batch-size 64 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 8 --lr 0.001 --wd 0.025 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### Swin Transformer v1

#### Swin Transformer v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1_t --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.0004 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --amp --compile
```

#### Swin Transformer v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1_s --batch-size 64 --opt adamw --clip-grad-norm 5 --lr 0.0004 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --amp --compile
```

#### Swin Transformer v1: Base

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1_b --batch-size 32 --opt adamw --clip-grad-norm 5 --lr 0.0004 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --amp --compile
```

#### Swin Transformer v1: Large

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1_l --batch-size 16 --opt adamw --clip-grad-norm 5 --lr 0.0004 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 384 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --ra-sampler --ra-reps 2 --amp --compile
```

### Swin Transformer v2

#### Swin Transformer v2: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_t --batch-size 64 --opt adamw --clip-grad-norm 5 --lr 0.0007 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### Swin Transformer v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --batch-size 64 --opt adamw --clip-grad-norm 5 --lr 0.0007 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --tag intermediate --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.0007 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 224 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --drop-last --amp --compile --save-frequency 1 --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_s --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 5 --grad-accum-steps 2 --lr 0.0007 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --drop-last --amp --compile --save-frequency 1 --resume-epoch 120 --load-states --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

#### Swin Transformer v2: Base

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_b --batch-size 32 --opt adamw --clip-grad-norm 5 --lr 0.0007 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### Swin Transformer v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_l --batch-size 16 --opt adamw --clip-grad-norm 5 --lr 0.0007 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 200 --warmup-epochs 20 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### Tiny ViT

#### Tiny ViT: 5M

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --amp --compile
```

Intermediate training training (suggested in the paper to use KD for this step)

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag intermediate --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.002 --wd 0.01 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Intermediate training training: linear probing

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag intermediate --reset-head --freeze-body --unfreeze-features --batch-size 256 --opt adamw --lr 0.001 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 10 --size 256 --aug-level 2 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --fast-matmul --compile --resume-epoch 0
```

Intermediate training training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag intermediate --batch-size 256 --opt adamw --clip-grad-norm 5 --lr 0.0005 --wd 1e-7 --norm-wd 0 --layer-decay 0.8 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 40 --warmup-epochs 5 --freeze-bn --size 256 --aug-level 8 --smoothing-alpha 0.1 --amp --compile --resume-epoch 10
```

Intermediate training training: increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_5m --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 5 --grad-accum-steps 2 --lr 0.00004 --wd 1e-7 --norm-wd 0 --layer-decay 0.8 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 30 --warmup-epochs 5 --freeze-bn --size 384 --aug-level 8 --smoothing-alpha 0.1 --amp --compile --resume-epoch 0
```

#### Tiny ViT: 11M

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_11m --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --amp --compile
```

#### Tiny ViT: 21M

```sh
torchrun --nproc_per_node=2 train.py --network tiny_vit_21m --batch-size 128 --opt adamw --clip-grad-norm 5 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --ra-sampler --ra-reps 2 --amp --compile
```

### TransNeXt

#### TransNeXt: Micro

```sh
torchrun --nproc_per_node=2 train.py --network transnext_micro --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### TransNeXt: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network transnext_tiny --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### TransNeXt: Small

```sh
torchrun --nproc_per_node=2 train.py --network transnext_small --batch-size 32 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

### UniFormer

#### UniFormer: Small

```sh
torchrun --nproc_per_node=2 train.py --network uniformer_s --batch-size 64 --opt adamw --grad-accum-steps 4 --lr 0.001 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 320 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### VAN

#### VAN: B0

```sh
torchrun --nproc_per_node=2 train.py --network van_b0 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### VAN: B1

```sh
torchrun --nproc_per_node=2 train.py --network van_b1 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### VAN: B2

```sh
torchrun --nproc_per_node=2 train.py --network van_b2 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### VGG

```sh
torchrun --nproc_per_node=2 train.py --network vgg_13 --batch-size 128 --lr 0.01 --aug-level 2
```

### VGG Reduced

```sh
torchrun --nproc_per_node=2 train.py --network vgg_reduced_19 --batch-size 64 --lr 0.1 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### ViT

#### ViT: b32

```sh
torchrun --nproc_per_node=2 train.py --network vit_b32 --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0007 --wd 0.3 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 30 --model-ema --size 224 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### ViT: b16

```sh
torchrun --nproc_per_node=2 train.py --network vit_b16 --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0007 --wd 0.3 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 30 --model-ema --size 288 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### ViT: l32

```sh
torchrun --nproc_per_node=2 train.py --network vit_l32 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.0007 --wd 0.3 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 30 --model-ema --size 320 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### ViT: l16

```sh
torchrun --nproc_per_node=2 train.py --network vit_l16 --batch-size 64 --opt adamw --opt-betas 0.9 0.95 --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0001 --wd 0.3 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### ViT: h14

```sh
torchrun --nproc_per_node=2 train.py --network vit_h14 --batch-size 32 --opt adamw --opt-betas 0.9 0.95 --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.0001 --wd 0.3 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 200 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile
```

#### ViT: SoViT 150m p14 AP

```sh
torchrun --nproc_per_node=2 train.py --network vit_so150m_p14_ap --batch-size 256 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### ViT Parallel

#### ViT Parallel: s16 18x2 LS

```sh
torchrun --nproc_per_node=2 train.py --network vit_parallel_s16_18x2_ls --bce-loss --bce-threshold 0.05 --batch-size 192 --opt adamw --clip-grad-norm 1 --lr 0.003 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 800 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

Fine-tuning, increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit_parallel_s16_18x2_ls --model-config drop_path_rate=0.0 --batch-size 64 --opt adamw --clip-grad-norm 1 --lr 0.00001 --wd 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 20 --warmup-epochs 5 --model-ema --size 384 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --amp --compile --resume-epoch 0
```

Intermediate training training

```sh
torchrun --nproc_per_node=2 train.py --network vit_parallel_s16_18x2_ls --tag intermediate --batch-size 128 --opt adamw --clip-grad-norm 1 --lr 0.0005 --wd 0.02 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 240 --warmup-epochs 5 --model-ema --size 256 --aug-level 2 --smoothing-alpha 0.1 --cutmix --amp --compile --wds --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

### ViT SAM

#### ViT SAM: b16

```sh
torchrun --nproc_per_node=2 train.py --network vit_sam_b16 --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 4 --lr 0.0005 --wd 0.1 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### VoVNet v1

#### VoVNet v1: 39

```sh
torchrun --nproc_per_node=2 train.py --network vovnet_v1_39 --batch-size 128 --lr-scheduler cosine --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --fast-matmul --compile
```

### VoVNet v2

#### VoVNet v2: 19

```sh
torchrun --nproc_per_node=2 train.py --network vovnet_v2_19 --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### VoVNet v2: 39

```sh
torchrun --nproc_per_node=2 train.py --network vovnet_v2_39 --batch-size 256 --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 20 --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

### Wide ResNet

#### Wide ResNet: 50

```sh
torchrun --nproc_per_node=2 train.py --network wide_resnet_50 --batch-size 128 --lr 0.1 --wd 0.01 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --model-ema --size 256 --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2
```

### Xception

```sh
torchrun --nproc_per_node=2 train.py --network xception --batch-size 64 --lr-scheduler cosine --aug-level 6 --smoothing-alpha 0.1 --mixup-alpha 0.2
```

### XCiT

#### XCiT: nano-12 p16

```sh
torchrun --nproc_per_node=2 train.py --network xcit_nano12_p16 --batch-size 256 --opt adamw --clip-grad-norm 1 --grad-accum-steps 2 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 30 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --fast-matmul --compile
```

#### XCiT: small-12 p8

```sh
torchrun --nproc_per_node=2 train.py --network xcit_small12_p8 --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 30 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

#### XCiT: medium-24 p16

```sh
torchrun --nproc_per_node=2 train.py --network xcit_medium24_p16 --batch-size 64 --opt adamw --clip-grad-norm 1 --grad-accum-steps 8 --lr 0.0005 --wd 0.05 --norm-wd 0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 400 --warmup-epochs 30 --model-ema --size 256 --aug-level 8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --ra-sampler --ra-reps 2 --amp --compile
```

## Common Dataset Training Scenarios

### ImageNet

#### ResNet v2: 50 ImageNet 1K example

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --tag imagenet1k --batch-size 256 --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --epochs 90 --aug-type aa --smoothing-alpha 0.1 --rgb-mode imagenet --fast-matmul --compile --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

#### ResNet v2: 50 ImageNet 1K example, ResNet strikes back procedure (A2)

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --tag imagenet1k --bce-loss --bce-threshold 0.2 --batch-size 256 --opt lamb --grad-accum-steps 4 --lr 0.005 --wd 0.02 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 5 --size 256 --aug-level 8 --mixup-alpha 0.1 --cutmix --rgb-mode imagenet --ra-sampler --ra-reps 2 --amp --compile --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

### ImageNet 21K

#### ResNet v2: 50 ImageNet 21K example

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --tag imagenet21k --batch-size 256 --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --epochs 90 --aug-type aa --smoothing-alpha 0.1 --rgb-mode imagenet --fast-matmul --compile --wds --wds-info ~/Datasets/imagenet-w21-webp-wds/_info.json --wds-class-file public_datasets_metadata/imagenet-21k-classes.txt --wds-training-split train
```
