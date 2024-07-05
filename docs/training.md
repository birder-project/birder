# Reference Training Procedure

Training script and procedures adapted from PyTorch vision reference
<https://github.com/pytorch/vision/tree/main/references/classification>

Set `OMP_NUM_THREADS`.

## Image Classification

Best to avoid cutmix on compact models - <https://arxiv.org/abs/2404.11202v1>

Optional intermediate training on weakly supervised dataset notes:

* Without EMA
* Higher weight decay
* Same learning rate
* Same number of epochs
* Same augmentations
* The lowest resolution for the model

On fine-tuning phase

* Run only the last training phase (highest resolution)
* At most 30% of the total epochs
* A small layer-decay (0.98 - 0.99)
* Consider slightly lower learning rate

### Reset Head Examples

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2 --net-param 0 --lr 0.1 --lr-scheduler cosine --batch-size 64 --epochs 10 --size 448 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --resume-epoch 0 --reset-head
```

Followed by an adapted learning rate and epoch configuration from the original procedure

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2 --net-param 0 --lr 0.002 --lr-scheduler cosine --batch-size 32 --epochs 80 --size 448 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --resume-epoch 10
```

Example with adamw optimizer

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --epochs 10 --size 448 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head
```

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 32 --epochs 80 --size 448 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --resume-epoch 10
```

Swin Transformer v2

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2 --net-param 1 --opt adamw --lr 0.00005 --lr-scheduler cosine --batch-size 64 --size 384 --lr-cosine-min 1e-7 --epochs 10 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --clip-grad-norm 5 --amp --resume-epoch 0 --reset-head
```

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2 --net-param 1 --opt adamw --lr 0.00001 --lr-scheduler cosine --batch-size 16 --size 384 --lr-cosine-min 1e-7 --epochs 80 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --resume-epoch 10
```

----

* [AlexNet](#alexnet)
* [CaiT](#cait)
* [ConvNeXt v1](#convnext-v1)
* [ConvNeXt v2](#convnext-v2)
* [DeiT](#deit)
* [DeiT3](#deit3)
* [DenseNet](#densenet)
* [EdgeViT](#edgevit)
* [EfficientNet v1](#efficientnet-v1)
* [EfficientNet v2](#efficientnet-v2)
* [InceptionNeXt](#inceptionnext)
* [Inception-ResNet v2](#inception-resnet-v2)
* [Inception v3](#inception-v3)
* [Inception v4](#inception-v4)
* [MaxViT](#maxvit)
* [MnasNet](#mnasnet)
* [Mobilenet v1](#mobilenet-v1)
* [Mobilenet v2](#mobilenet-v2)
* [Mobilenet v3](#mobilenet-v3)
* [MobileViT v1](#mobilevit-v1)
* [MobileViT v2](#mobilevit-v2)
* [Next-ViT](#next-vit)
* [RegNet](#regnet)
* [ResNeSt](#resnest)
* [ResNet v2](#resnet-v2)
* [ResNeXt](#resnext)
* [SE ResNet v2](#se-resnet-v2)
* [SE ResNeXt](#se-resnext)
* [ShuffleNet v1](#shufflenet-v1)
* [ShuffleNet v2](#shufflenet-v2)
* [Simple ViT](#simple-vit)
* [SqueezeNet](#squeezenet)
* [SqueezeNext](#squeezenext)
* [Swin Transformer v1](#swin-transformer-v1)
* [Swin Transformer v2](#swin-transformer-v2)
* [Swin Transformer v2 w2](#swin-transformer-v2-w2)
* [VGG](#vgg)
* [VGG Reduced](#vgg-reduced)
* [ViT](#vit)
* [ViTReg](#vitreg)
* [Wide ResNet](#wide-resnet)
* [Xception](#xception)
* [XCiT](#xcit)

----

### AlexNet

```sh
torchrun --nproc_per_node=2 train.py --network alexnet --lr 0.01 --batch-size 128 --aug-level 2
```

### CaiT

#### CaiT: Small 24

```sh
torchrun --nproc_per_node=2 train.py --network cait --net-param 3 --opt adamw --lr 0.001 --lr-scheduler cosine --batch-size 64 --lr-cosine-min 1e-7 --warmup-epochs 5 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### ConvNeXt v1

#### ConvNeXt v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1 --net-param 0 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 128 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none
```

#### ConvNeXt v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1 --net-param 1 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 320 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none
```

#### ConvNeXt v1: Base

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1 --net-param 2 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none
```

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1 --net-param 2 --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 16 --epochs 40 --size 512 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none --resume-epoch 0
```

#### ConvNeXt v1: Large

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v1 --net-param 3 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 448 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --rgb-mode none
```

### ConvNeXt v2

#### ConvNeXt v2: Atto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 0 --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 300 --size 256 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp
```

#### ConvNeXt v2: Femto

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 1 --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.2 --mixup-alpha 0.3 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp
```

#### ConvNeXt v2: Pico

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 2 --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 300 --size 288 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.2 --mixup-alpha 0.3 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp
```

#### ConvNeXt v2: Nano

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 3 --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 300 --size 288 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.2 --mixup-alpha 0.5 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp
```

#### ConvNeXt v2: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 64 --epochs 300 --size 320 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 64 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 200 --load-states
```

At epoch 240 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --opt adamw --lr 0.0008 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 40 --batch-size 32 --epochs 300 --size 448 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 240 --load-states
```

#### ConvNeXt v2: Base

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 5 --opt adamw --lr 0.00625 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 16 --epochs 200 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### ConvNeXt v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 6 --opt adamw --lr 0.00625 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 20 --batch-size 16 --epochs 200 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### ConvNeXt v2: Huge

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 7 --opt adamw --lr 0.00125 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 4 --epochs 100 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### DeiT

#### DeiT: b16

```sh
torchrun --nproc_per_node=2 train.py --network deit --net-param 2 --opt adamw --lr 0.0005 --lr-scheduler cosine --batch-size 128 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 300 --size 288 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

At epoch 200 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network deit --net-param 2 --opt adamw --lr 0.00015 --lr-scheduler cosine --batch-size 64 --lr-cosine-min 1e-7 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --resume-epoch 200
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network deit --net-param 2 --tag intermediate --opt adamw --lr 0.0005 --lr-scheduler cosine --batch-size 128 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --size 256 --wd 0.1 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --clip-grad-norm 1 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

### DeiT3

Same as DeiT

### DenseNet

#### DenseNet: 161

```sh
torchrun --nproc_per_node=2 train.py --network densenet --net-param 161 --lr-scheduler cosine --batch-size 64 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### EdgeViT

#### EdgeViT: Extra small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit --net-param 1 --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 200 --size 224 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

#### EdgeViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network edgevit --net-param 2 --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 128 --epochs 200 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network edgevit --net-param 2 --tag intermediate --opt adamw --lr 5e-4 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 5 --batch-size 256 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

### EfficientNet v1

#### EfficientNet v1: B3

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1 --net-param 3 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 288 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

#### EfficientNet v1: B4

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1 --net-param 4 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2
```

#### EfficientNet v1: B5

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v1 --net-param 5 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 384 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

### EfficientNet v2

#### EfficientNet v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2 --net-param 0 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2 --net-param 0 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 384 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 150 --load-states
```

At epoch 260 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2 --net-param 0 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 448 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 260 --load-states
```

#### EfficientNet v2: Medium

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2 --net-param 1 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### EfficientNet v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2 --net-param 2 --lr 0.5 --lr-scheduler cosine --lr-cosine-min 1e-6 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 384 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

#### FocalNet: Tiny SRF

```sh
torchrun --nproc_per_node=2 train.py --network focalnet --net-param 0 --opt adamw --lr 1e-4 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 288 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --rgb-mode none
```

#### FocalNet: Small LRF

```sh
torchrun --nproc_per_node=2 train.py --network focalnet --net-param 3 --opt adamw --lr 1e-4 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 320 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --rgb-mode none
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network focalnet --net-param 3 --opt adamw --lr 1e-4 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 384 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --rgb-mode none --resume-epoch 150 --load-states
```

At epoch 220 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network focalnet --net-param 3 --opt adamw --lr 1e-4 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 16 --epochs 300 --size 448 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile --rgb-mode none --resume-epoch 220 --load-states
```

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

#### GhostNet v2: 1 (100)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2 --net-param 1 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### GhostNet v2: 1.3 (130)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2 --net-param 1.3 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 128 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### GhostNet v2: 1.6 (160)

```sh
torchrun --nproc_per_node=2 train.py --network ghostnet_v2 --net-param 1.6 --lr 0.4 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.9 --batch-size 128 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### InceptionNeXt

#### InceptionNeXt: Small

```sh
torchrun --nproc_per_node=2 train.py --network inception_next --net-param 1 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 10 --batch-size 64 --epochs 300 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

#### InceptionNeXt: Base

```sh
torchrun --nproc_per_node=2 train.py --network inception_next --net-param 2 --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-5 --warmup-epochs 10 --batch-size 32 --epochs 300 --size 288 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile
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

### MaxViT

#### MaxViT: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network maxvit --net-param 0 --opt adamw --lr 0.003 --lr-scheduler cosine --batch-size 64 --size 256 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 3 --model-ema --clip-grad-norm 1 --amp --rgb-mode none
```

#### MaxViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network maxvit --net-param 1 --opt adamw --lr 0.003 --lr-scheduler cosine --batch-size 64 --size 256 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --rgb-mode none
```

At epoch 120 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network maxvit --net-param 1 --opt adamw --lr 0.003 --lr-scheduler cosine --batch-size 32 --size 320 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --rgb-mode none --resume-epoch 120 --load-scheduler
```

At epoch 140 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network maxvit --net-param 1 --opt adamw --lr 0.003 --lr-scheduler cosine --batch-size 16 --size 384 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --rgb-mode none --resume-epoch 140 --load-scheduler
```

At epoch 180 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network maxvit --net-param 1 --opt adamw --lr 0.003 --lr-scheduler cosine --batch-size 16 --size 448 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --rgb-mode none --resume-epoch 180 --load-scheduler
```

#### MaxViT: Base

```sh
torchrun --nproc_per_node=2 train.py --network maxvit --net-param 2 --opt adamw --lr 0.0014 --lr-scheduler cosine --batch-size 32 --size 288 --lr-cosine-min 1e-7 --warmup-epochs 32 --epochs 200 --wd 0.05 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --clip-grad-norm 1 --amp --rgb-mode none
```

### MnasNet

```sh
torchrun --nproc_per_node=2 train.py --network mnasnet --net-param 0.5 --lr 0.5 --nesterov --lr-scheduler cosine --lr-cosine-min 5e-6 --warmup-epochs 5 --batch-size 256 --epochs 200 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### Mobilenet v1

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v1 --net-param 1 --lr-scheduler cosine --batch-size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### Mobilenet v2

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v2 --net-param 2 --lr 0.045 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --epochs 200 --wd 0.00004 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### Mobilenet v3

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3 --net-param 1 --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 256 --size 256 --epochs 300 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4
```

At epoch 210 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network mobilenet_v3 --net-param 1 --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 128 --size 384 --epochs 300 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --resume-epoch 210 --load-states
```

### MobileViT v1

#### MobileViT v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v1 --net-param 2 --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 5 --batch-size 128 --size 256 --epochs 300 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --ra-sampler --ra-reps 2 --amp
```

### MobileViT v2

#### MobileViT v2: 1.5

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1.5 --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 20 --batch-size 128 --size 256 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 260 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1.5 --opt adamw --lr 0.002 --lr-scheduler cosine --lr-cosine-min 2e-5 --warmup-epochs 20 --batch-size 64 --size 384 --epochs 300 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 260 --load-states
```

Optional intermediate training

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1.5 --tag intermediate --opt adamw --lr 0.0003 --lr-scheduler cosine --lr-cosine-min 2e-5 --batch-size 128 --size 256 --epochs 80 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: fine-tuning

```sh
torchrun --nproc_per_node=2 train.py --network mobilevit_v2 --net-param 1.5 --tag intermediate --lr 0.001 --lr-scheduler cosine --batch-size 64 --size 384 --epochs 50 --wd 0.00004 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --ra-sampler --ra-reps 2 --amp --compile --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed --resume-epoch 0
```

### Next-ViT

#### Next-ViT: Small

```sh
torchrun --nproc_per_node=2 train.py --network nextvit --net-param 0 --opt adamw --lr 0.00175 --lr-scheduler cosine --warmup-epochs 20 --batch-size 64 --size 256 --epochs 300 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit --net-param 0 --opt adamw --lr 0.00175 --lr-scheduler cosine --warmup-epochs 20 --batch-size 32 --size 384 --epochs 300 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --amp --resume-epoch 250 --load-states
```

#### Next-ViT: Base

```sh
torchrun --nproc_per_node=2 train.py --network nextvit --net-param 1 --opt adamw --lr 0.00175 --lr-scheduler cosine --warmup-epochs 20 --batch-size 64 --size 256 --epochs 300 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 250 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit --net-param 1 --opt adamw --lr 0.00175 --lr-scheduler cosine --warmup-epochs 20 --batch-size 32 --size 384 --epochs 300 --wd 0.1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 250 --load-states
```

When finished, run 30 more epochs at increased resolution

```sh
torchrun --nproc_per_node=2 train.py --network nextvit --net-param 1 --opt adamw --lr 5e-6 --lr-scheduler cosine --batch-size 32 --size 448 --epochs 330 --wd 1e-8 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 300
```

### RegNet

#### RegNet: 1.6 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet --net-param 1.6 --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### RegNet: 8 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet --net-param 8 --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

At epoch 70 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network regnet --net-param 8 --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 32 --size 384 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 70 --load-states
```

At epoch 95 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network regnet --net-param 8 --lr 0.4 --lr-scheduler cosine --warmup-epochs 5 --batch-size 16 --size 448 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --resume-epoch 95 --load-states
```

#### RegNet: 16 GF

```sh
torchrun --nproc_per_node=2 train.py --network regnet --net-param 16 --lr 0.2 --lr-scheduler cosine --warmup-epochs 5 --batch-size 32 --size 288 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

### ResNeSt

#### ResNeSt: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnest --net-param 50 --lr-scheduler cosine --batch-size 32 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### ResNeSt: 101

```sh
torchrun --nproc_per_node=2 train.py --network resnest --net-param 101 --lr-scheduler cosine --batch-size 32 --size 288 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 4 --ra-sampler --ra-reps 2 --amp --compile
```

### ResNet v2

#### ResNet v2: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2 --net-param 50 --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --batch-size 256 --epochs 90 --smoothing-alpha 0.1 --aug-level 3
```

#### ResNet v2: 50, ResNet strikes back procedure

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2 --net-param 50 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 128 --epochs 200 --size 256 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

### ResNeXt

#### ResNeXt: 50

```sh
torchrun --nproc_per_node=2 train.py --network resnext --net-param 50 --lr-scheduler cosine --batch-size 64 --size 256 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

#### ResNeXt: 101

```sh
torchrun --nproc_per_node=2 train.py --network resnext --net-param 101 --lr 0.04 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 64 --epochs 200 --size 256 --wd 0.001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp
```

At epoch 150 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network resnext --net-param 101 --lr 0.04 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 10 --batch-size 32 --epochs 200 --size 384 --wd 0.001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --resume-epoch 150 --load-states
```

### SE ResNet v2

Same as ResNet v2

### SE ResNeXt

Same as ResNeXt

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
torchrun --nproc_per_node=2 train.py --network simple_vit --net-param 0 --opt adamw --lr 0.0004 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --size 256 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### Simple ViT: b16

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit --net-param 1 --opt adamw --lr 0.0004 --lr-scheduler cosine --batch-size 64 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --size 288 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### Simple ViT: l32

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit --net-param 2 --opt adamw --lr 0.00025 --lr-scheduler cosine --batch-size 64 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --size 320 --wd 0.0001 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### Simple ViT: l16

```sh
torchrun --nproc_per_node=2 train.py --network simple_vit --net-param 3 --lr 0.3 --lr-scheduler cosine --batch-size 16 --lr-cosine-min 1e-6 --warmup-epochs 15 --epochs 400 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### SqueezeNet

```sh
torchrun --nproc_per_node=2 train.py --network squeezenet --lr 0.01 --batch-size 512 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 2
```

### SqueezeNext

```sh
torchrun --nproc_per_node=2 train.py --network squeezenext --net-param 2 --lr 0.1 --lr-scheduler step --lr-step-size 20 --lr-step-gamma 0.75 --batch-size 128 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### Swin Transformer v1

#### Swin Transformer v1: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1 --net-param 0 --opt adamw --lr 0.0004 --lr-scheduler cosine --batch-size 128 --size 256 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v1: Small

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1 --net-param 1 --opt adamw --lr 0.0004 --lr-scheduler cosine --batch-size 64 --size 256 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v1: Base

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1 --net-param 2 --opt adamw --lr 0.0004 --lr-scheduler cosine --batch-size 32 --size 320 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v1: Large

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v1 --net-param 2 --opt adamw --lr 0.0004 --lr-scheduler cosine --batch-size 16 --size 384 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### Swin Transformer v2

#### Swin Transformer v2: Tiny

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2 --net-param 0 --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 64 --size 256 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v2: Small

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2 --net-param 1 --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 64 --size 256 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v2: Base

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2 --net-param 2 --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 32 --size 256 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

#### Swin Transformer v2: Large

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2 --net-param 3 --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 16 --size 320 --lr-cosine-min 1e-7 --warmup-epochs 20 --epochs 200 --wd 0.05 --norm-wd 0 --bias-weight-decay 0 --transformer-embedding-decay 0 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 5 --amp --compile
```

### Swin Transformer v2 w2

Same as Swin Transformer v2

### VGG

```sh
torchrun --nproc_per_node=2 train.py --network vgg --net-param 13 --lr 0.01 --batch-size 128 --aug-level 2
```

### VGG Reduced

```sh
torchrun --nproc_per_node=2 train.py --network vgg_reduced --net-param 19 --lr 0.1 --batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### ViT

#### ViT: b32

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 0 --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --warmup-epochs 30 --epochs 300 --size 224 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ViT: b16

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 1 --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 128 --lr-cosine-min 1e-7 --warmup-epochs 30 --epochs 300 --size 288 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ViT: l32

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 2 --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 64 --lr-cosine-min 1e-7 --warmup-epochs 30 --epochs 300 --size 320 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ViT: l16

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 3 --lr 0.3 --lr-scheduler cosine --batch-size 16 --lr-cosine-min 1e-6 --warmup-epochs 10 --epochs 400 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

#### ViT: h14

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 4 --lr 0.3 --lr-scheduler cosine --batch-size 8 --lr-cosine-min 1e-6 --warmup-epochs 10 --epochs 400 --size 336 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

### ViTReg

Same as ViT

### Wide ResNet

#### Wide ResNet: 50

```sh
torchrun --nproc_per_node=2 train.py --network wide_resnet --net-param 50 --lr 0.1 --lr-scheduler cosine --lr-cosine-min 1e-7 --warmup-epochs 5 --batch-size 128 --epochs 300 --size 256 --wd 0.01 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 3 --model-ema --ra-sampler --ra-reps 2
```

### Xception

```sh
torchrun --nproc_per_node=2 train.py --network xception --lr-scheduler cosine --batch-size 64 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3
```

### XCiT

### XCiT: small p8

```sh
torchrun --nproc_per_node=2 train.py --network xcit --net-param 5 --opt adamw --lr 0.0005 --lr-scheduler cosine --batch-size 64 --lr-cosine-min 1e-7 --warmup-epochs 30 --epochs 400 --size 256 --wd 0.05 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile
```

----

## ImageNet

### ResNet v2: 50 ImageNet 1K Example

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2 --net-param 50 --tag imagenet1k --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --batch-size 256 --epochs 90 --smoothing-alpha 0.1 --aug-level 3 --rgb-mode imagenet --wds --wds-class-file public_datasets_metadata/imagenet-1k-wds/classes.txt --wds-train-size 1281167 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

## ImageNet 21K

### ResNet v2: 50 ImageNet 21K Example

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2 --net-param 50 --tag imagenet21k --lr-scheduler step --lr-step-size 30 --lr-step-gamma 0.1 --batch-size 256 --epochs 90 --smoothing-alpha 0.1 --aug-level 3 --rgb-mode imagenet --wds --wds-class-file public_datasets_metadata/imagenet-w21-webp-wds/classes.txt --wds-train-size 13022846 --data-path ~/Datasets/imagenet-w21-webp-wds/training --wds-val-size 128430 --val-path ~/Datasets/imagenet-w21-webp-wds/validation
```

----

## Image Pre-training

### FCMAE

#### FCMAE: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 train_pretrain.py --network fcmae --encoder convnext_v2 --encoder-param 4 --opt adamw --lr 0.00015 --lr-scheduler cosine --lr-cosine-min 0.0 --warmup-epochs 40 --batch-size 256 --epochs 800 --wd 0.05 --amp --compile --save-frequency 1 --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 15 --batch-size 256 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --layer-decay 0.9 --resume-epoch 10 --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: fine-tuning, second stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --tag pretrained-intermediate --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 384 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head
```

Optional intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --tag pretrained-intermediate --opt adamw --lr 0.000025 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 64 --epochs 60 --size 448 --wd 1e-8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.9 --resume-epoch 10
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 320 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 128 --epochs 210 --size 320 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.96 --resume-epoch 10
```

At epoch 120 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 64 --epochs 210 --size 384 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.96 --resume-epoch 120 --load-states
```

At epoch 170 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 4 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 30 --batch-size 64 --epochs 210 --size 448 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.96 --resume-epoch 170 --load-states
```

#### FCMAE: ConvNeXt v2 Base

```sh
torchrun --nproc_per_node=2 train_pretrain.py --network fcmae --encoder convnext_v2 --encoder-param 5 --opt adamw --lr 0.00015 --lr-scheduler cosine --lr-cosine-min 0.0 --warmup-epochs 40 --batch-size 128 --epochs 800 --wd 0.05 --amp --compile --save-frequency 1 --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 5 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 256 --epochs 10 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 5 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-8 --warmup-epochs 15 --batch-size 128 --epochs 100 --size 256 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --amp --compile --layer-decay 0.8 --resume-epoch 10 --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Optional intermediate training: fine-tuning, second stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 5 --tag pretrained-intermediate --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 384 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head
```

Optional intermediate training: full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 5 --tag pretrained-intermediate --opt adamw --lr 0.000025 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 32 --epochs 40 --size 448 --wd 1e-8 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --amp --compile --layer-decay 0.8 --resume-epoch 10
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network convnext_v2 --net-param 5 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 128 --epochs 10 --size 320 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head
```

### MAE ViT

#### MAE ViT: ViT b16

```sh
torchrun --nproc_per_node=2 train_pretrain.py --network mae_vit --encoder vit --encoder-param 1 --opt adamw --lr 0.00015 --lr-scheduler cosine --lr-cosine-min 0.0 --warmup-epochs 40 --batch-size 256 --epochs 800 --wd 0.05 --amp --compile --save-frequency 1 --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 1 --tag pretrained --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 512 --lr-cosine-min 1e-7 --epochs 10 --size 256 --wd 0.3 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 1 --tag pretrained --opt adamw --lr 0.0007 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-7 --epochs 10 --size 320 --wd 0.3 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --resume-epoch 0 --reset-head
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 1 --tag pretrained --opt adamw --lr 0.0002 --lr-scheduler cosine --batch-size 128 --lr-cosine-min 1e-7 --warmup-epochs 15 --epochs 100 --size 320 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 10
```

At epoch 75 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 1 --tag pretrained --opt adamw --lr 0.00004 --lr-scheduler cosine --batch-size 64 --lr-cosine-min 1e-7 --epochs 100 --size 384 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 75
```

At epoch 80 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 1 --tag pretrained --opt adamw --lr 0.00002 --lr-scheduler cosine --batch-size 32 --lr-cosine-min 1e-7 --epochs 100 --size 448 --wd 0.3 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.8 --resume-epoch 80
```

#### MAE ViT: ViT l16

```sh
torchrun --nproc_per_node=2 train_pretrain.py --network mae_vit --encoder vit --encoder-param 3 --opt adamw --lr 0.00015 --lr-scheduler cosine --lr-cosine-min 0.0 --warmup-epochs 40 --batch-size 128 --epochs 800 --wd 0.05 --amp --compile --save-frequency 1 --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Optional intermediate training: first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 3 --tag pretrained --lr 0.3 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-6 --epochs 10 --size 256 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head --wds --wds-class-file data/training_packed/classes.txt --data-path data/training_packed --val-path data/validation_packed
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 3 --tag pretrained --lr 0.3 --lr-scheduler cosine --batch-size 256 --lr-cosine-min 1e-6 --epochs 10 --size 320 --wd 0.00002 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 2 --amp --compile --resume-epoch 0 --reset-head
```

Next, full fine-tuning with layer-wise learning rate decay

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 3 --tag pretrained --lr 0.1 --lr-scheduler cosine --batch-size 32 --lr-cosine-min 1e-6 --warmup-epochs 20 --epochs 100 --size 320 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.9 --resume-epoch 10
```

At epoch 75 increase resolution

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 3 --tag pretrained --lr 0.01 --lr-scheduler cosine --batch-size 16 --lr-cosine-min 1e-6 --epochs 100 --size 384 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.9 --resume-epoch 75
```

At epoch 80 increase resolution again

```sh
torchrun --nproc_per_node=2 train.py --network vit --net-param 3 --tag pretrained --lr 0.0075 --lr-scheduler cosine --batch-size 16 --lr-cosine-min 1e-6 --epochs 100 --size 448 --wd 0.00002 --norm-wd 0 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --model-ema --ra-sampler --ra-reps 2 --clip-grad-norm 1 --amp --compile --layer-decay 0.9 --resume-epoch 80
```

### SimMIM

#### SimMIM: Swin Transformer v2 Small

```sh
torchrun --nproc_per_node=2 train_pretrain.py --network simmim --encoder swin_transformer_v2 --encoder-param 1 --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 0.0 --warmup-epochs 10 --batch-size 128 --epochs 800 --wd 0.05 --clip-grad-norm 5 --amp --compile --save-frequency 1 --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2 --net-param 1 --tag pretrained --opt adamw --lr 0.00005 --lr-scheduler cosine --batch-size 256 --size 256 --lr-cosine-min 1e-7 --epochs 10 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --clip-grad-norm 5 --amp --resume-epoch 0 --reset-head
```

#### SimMIM: Swin Transformer v2 Base

```sh
torchrun --nproc_per_node=2 train_pretrain.py --network simmim --encoder swin_transformer_v2 --encoder-param 2 --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 0.0 --warmup-epochs 10 --batch-size 128 --epochs 800 --wd 0.05 --clip-grad-norm 5 --amp --compile --save-frequency 1 --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

#### SimMIM: Swin Transformer v2 w2 Base

```sh
torchrun --nproc_per_node=2 train_pretrain.py --network simmim --encoder swin_transformer_v2_w2 --encoder-param 2 --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 0.0 --warmup-epochs 10 --batch-size 64 --epochs 800 --wd 0.05 --clip-grad-norm 5 --amp --compile --save-frequency 1 --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network swin_transformer_v2_w2 --net-param 2 --tag pretrained --opt adamw --lr 0.00005 --lr-scheduler cosine --batch-size 128 --size 256 --lr-cosine-min 1e-7 --epochs 10 --wd 0.05 --smoothing-alpha 0.1 --mixup-alpha 0.8 --cutmix --aug-level 2 --clip-grad-norm 5 --amp --resume-epoch 0 --reset-head
```

## Knowledge Distillation

### ConvNeXt v2 Tiny Teacher: RegNet 1.6 GF

```sh
torchrun --nproc_per_node=2 train_kd.py --teacher convnext_v2 --teacher-param 4 --teacher-epoch 0 --student regnet --student-param 1.6 --lr 0.8 --lr-scheduler cosine --warmup-epochs 5 --batch-size 128 --size 256 --epochs 100 --wd 0.00005 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --compile
```

### ConvNeXt v2 Tiny Teacher: MobileNet v3

```sh
torchrun --nproc_per_node=2 train_kd.py --teacher convnext_v2 --teacher-param 4 --teacher-epoch 0 --student mobilenet_v3 --student-param 1 --lr 0.064 --lr-scheduler step --lr-step-size 2 --lr-step-gamma 0.973 --batch-size 64 --size 288 --epochs 300 --wd 0.00001 --smoothing-alpha 0.1 --mixup-alpha 0.2 --cutmix --aug-level 4 --compile
```