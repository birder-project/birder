# Reference Detection Training Procedure

## Object Detection

- [Deformable DETR](#deformable-detr)
- [Deformable DETR BoxRef](#deformable-detr-boxref)
- [DETR](#detr)
- [EfficientDet](#efficientdet)
- [Faster R-CNN](#faster-r-cnn)
- [FCOS](#fcos)
- [RetinaNet](#retinanet)
- [SSD](#ssd)
- [SSDLite](#ssdlite)
- [ViTDet](#vitdet)

### Deformable DETR

#### Deformable DETR: RegNet Y 4 GF

Optional intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr --tag coco --backbone regnet_y_4g --backbone-tag imagenet21k --backbone-pretrained --opt adamw --lr 0.0002 --backbone-lr 0.00002 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --freeze-backbone-bn --batch-size 4 --epochs 50 --wd 0.0001 --grad-accum-steps 2 --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

### Deformable DETR BoxRef

#### Deformable DETR BoxRef: RDNet Small

Optional intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr_boxref --tag coco --backbone rdnet_s --backbone-tag vicreg --backbone-pretrained --opt adamw --lr 0.0002 --backbone-lr 0.00002 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --batch-size 4 --epochs 50 --wd 0.0001 --grad-accum-steps 8 --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

Optional intermediate multiscale training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr_boxref --tag coco --backbone rdnet_s --backbone-tag vicreg --backbone-pretrained --opt adamw --lr 0.0002 --backbone-lr 0.00002 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --batch-size 2 --epochs 50 --wd 0.0001 --grad-accum-steps 16 --multiscale --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

Optional intermediate multiscale training: linear probing

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr_boxref --tag coco --backbone rdnet_s --backbone-tag vicreg --opt adamw --lr 0.0001 --batch-size 2 --epochs 10 --wd 0.0001 --grad-accum-steps 16 --multiscale --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --resume-epoch 0 --reset-head --freeze-body
```

### DETR

#### DETR: CSP ResNet 50

Optional intermediate multiscale training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network detr --tag coco --backbone csp_resnet_50 --backbone-tag imagenet1k --backbone-pretrained --opt adamw --lr 0.0001 --backbone-lr 0.00001 --lr-scheduler step --lr-step-size 200 --lr-step-gamma 0.1 --freeze-backbone-bn --batch-size 2 --epochs 300 --wd 0.0001 --grad-accum-steps 16 --aug-type detr --rgb-mode imagenet --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

#### DETR: Tiny ViT 11M

Optional intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network detr --tag coco --backbone tiny_vit_11m --opt adamw --lr 0.0001 --backbone-lr 0.00001 --lr-scheduler step --lr-step-size 200 --lr-step-gamma 0.1 --batch-size 64 --epochs 300 --wd 0.0001 --aug-level 3 --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

#### DETR: ViT reg4 m16 rms AVG

```sh
torchrun --nproc_per_node=2 train_detection.py --network detr --tag coco --backbone vit_reg4_m16_rms_avg --backbone-tag i-jepa-imagenet21k --backbone-pretrained --opt adamw --lr 0.0001 --backbone-lr 0.00001 --lr-scheduler step --lr-step-size 200 --lr-step-gamma 0.1 --batch-size 1 --grad-accum-steps 32 --epochs 300 --wd 0.0001 --aug-level 5 --multiscale --max-size 1152 --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

### EfficientDet

#### EfficientDet D0: EfficientNet v1 B0

```sh
torchrun --nproc_per_node=2 train_detection.py --network efficientdet_d0 --backbone efficientnet_v1_b0 --lr 0.08 --lr-scheduler cosine --warmup-epochs 10 --batch-size 32 --epochs 300 --wd 0.00004 --model-ema --clip-grad-norm 10 --amp --amp-dtype bfloat16 --compile
```

#### EfficientDet D3: EfficientNet v1 B3

Optional intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network efficientdet_d3 --tag coco --backbone efficientnet_v1_b3 --lr 0.08 --lr-scheduler cosine --warmup-epochs 10 --batch-size 24 --epochs 300 --wd 0.00004 --sync-bn --model-ema --clip-grad-norm 10 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

### Faster R-CNN

#### Faster R-CNN: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone efficientnet_v2_s --backbone-epoch 0 --lr 0.02 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --freeze-backbone-stages 2 --freeze-backbone-bn --batch-size 16 --epochs 26 --wd 0.0001 --fast-matmul --compile-backbone
```

#### Faster R-CNN: Hiera AbsWin Base

Optional intermediate training (COCO) - warmup

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --tag coco --backbone hieradet_base --backbone-tag mim --backbone-pretrained --freeze-backbone --lr 0.01 --batch-size 32 --epochs 2 --size 768 --wd 0.0001 --sync-bn --amp --amp-dtype bfloat16 --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

Optional intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --tag coco --backbone hieradet_base --backbone-tag mim --backbone-pretrained --lr 0.02 --backbone-lr 0.01 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --batch-size 16 --epochs 26 --size 768 --wd 0.0001 --grad-accum-steps 2 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

### FCOS

A bit unstable with AMP

#### FCOS: Tiny ViT 5M

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone tiny_vit_5m --backbone-epoch 0 --lr 0.01 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --freeze-backbone-bn --batch-size 16 --epochs 32 --wd 0.0001 --fast-matmul
```

#### FCOS: EfficientNet v2 Small

Optional warmup

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone efficientnet_v2_s --backbone-pretrained --freeze-backbone --lr 0.01 --freeze-backbone-bn --batch-size 64 --epochs 2 --wd 0.0001 --fast-matmul
```

Optional warmup: actual training

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone efficientnet_v2_s --backbone-pretrained --lr 0.01 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --freeze-backbone-bn --batch-size 16 --epochs 32 --wd 0.0001 --fast-matmul --resume-epoch 0
```

### RetinaNet

#### RetinaNet: ConvNeXt v2 Tiny

Optional intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network retinanet --tag coco --backbone convnext_v2_tiny --backbone-tag vicreg --opt adamw --lr 0.0001 --epochs 26 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --batch-size 2 --wd 0.05 --norm-wd 0 --sync-bn --aug-type multiscale --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

#### RetinaNet: CSP ResNet 50

```sh
torchrun --nproc_per_node=2 train_detection.py --network retinanet --backbone csp_resnet_50 --backbone-epoch 0 --opt adamw --lr 0.0001 --epochs 26 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --freeze-backbone-stages 1 --freeze-backbone-bn --batch-size 32 --wd 0.05 --norm-wd 0 --amp --compile --compile-opt
```

### SSD

#### SSD: MobileNet v4 Medium

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssd --backbone mobilenet_v4_m --backbone-epoch 0 --freeze-backbone-stages 4 --lr 0.015 --lr-scheduler cosine --batch-size 64 --epochs 300 --wd 0.00002 --aug-type ssd --fast-matmul
```

### SSDLite

#### SSDLite: MobileNet v2 1

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssdlite --backbone mobilenet_v2 --backbone-param 1 --backbone-epoch 0 --lr 0.15 --lr-scheduler cosine --batch-size 32 --epochs 600 --wd 0.00004 --aug-type ssdlite --fast-matmul
```

#### SSDLite: MobileNet v4 Hybrid Medium

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssdlite --backbone mobilenet_v4_hybrid_m --backbone-epoch 0 --opt adamw --lr 0.002 --backbone-lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 32 --warmup-epochs 20 --epochs 600 --wd 0.0001 --aug-type ssdlite --fast-matmul --compile-opt
```

Optional intermediate training (COCO)

#### SSDLite: RoPEi ViT reg1 s16 pn c1

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssdlite --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-pretrained --opt adamw --lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 64 --warmup-epochs 20 --epochs 600 --wd 0.0001 --rgb-mode none --freeze-backbone --fast-matmul --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

### ViTDet

#### ViTDet: ViT Det m16 rms

```sh
torchrun --nproc_per_node=2 train_detection.py --network vitdet --backbone vit_det_m16_rms --backbone-tag i-jepa-imagenet21k --backbone-pretrained --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 16 --warmup-epochs 2 --epochs 100 --size 672 --wd 0.1 --norm-wd 0 --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile-backbone --layer-decay 0.9
```

Optional intermediate training (Objects365-2020)

```sh
torchrun --nproc_per_node=2 train_detection.py --network vitdet --tag objects365 --backbone vit_det_m16_rms --backbone-tag i-jepa --backbone-pretrained --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 16 --warmup-epochs 2 --epochs 20 --size 672 --wd 0.1 --norm-wd 0 --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile-backbone --layer-decay 0.9 --data-path ~/Datasets/Objects365-2020/train --val-path ~/Datasets/Objects365-2020/val --coco-json-path ~/Datasets/Objects365-2020/train/zhiyuan_objv2_train.json --coco-val-json-path ~/Datasets/Objects365-2020/val/zhiyuan_objv2_val.json --ignore-file public_datasets_metadata/objects365_ignore.txt
```

Optional intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 train_detection.py --network vitdet --tag coco --backbone vit_det_m16_rms --backbone-tag i-jepa-imagenet21k --backbone-pretrained --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 16 --warmup-epochs 2 --epochs 100 --size 672 --wd 0.1 --norm-wd 0 --clip-grad-norm 1 --amp --amp-dtype bfloat16 --compile-backbone --layer-decay 0.9 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### ViTDet: ViT SAM b16

```sh
torchrun --nproc_per_node=2 train_detection.py --network vitdet --backbone vit_sam_b16 --backbone-epoch 0 --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 8 --warmup-epochs 2 --epochs 100 --size 672 --wd 0.1 --norm-wd 0 --clip-grad-norm 1 --amp --compile-backbone --compile-opt --layer-decay 0.7
```

## Common Dataset Training Scenarios

### COCO

#### Faster R-CNN: CSP ResNet 50 Backbone COCO example

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --tag coco --backbone csp_resnet_50 --backbone-tag imagenet1k --backbone-pretrained --lr 0.02 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --freeze-backbone-stages 2 --freeze-backbone-bn --batch-size 16 --epochs 26 --wd 0.0001 --rgb-mode imagenet --fast-matmul --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

#### Faster R-CNN: ResNet v1 50 Backbone COCO example

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --tag coco --backbone resnet_v1_50 --lr 0.1 --lr-scheduler multistep --lr-steps 352 384 --lr-step-gamma 0.1 --batch-size 16 --epochs 400 --wd 0.0004 --aug-type lsj --rgb-mode imagenet --amp --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

#### SSD: MobileNet v4 Medium Backbone COCO example

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssd --tag coco --backbone mobilenet_v4_m --backbone-epoch 0 --lr 0.015 --lr-scheduler cosine --batch-size 64 --epochs 300 --wd 0.00002 --aug-type ssd --fast-matmul --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

## Auto Labeler Training

### Deformable DETR BoxRef: ConvNeXt v2 Tiny

Step 1: Training

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr_boxref --tag coco --backbone convnext_v2_tiny --backbone-tag imagenet21k --backbone-pretrained --opt adamw --lr 0.0002 --backbone-lr 0.00002 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --batch-size 2 --epochs 50 --wd 0.0001 --grad-accum-steps 16 --multiscale --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

Step 2: Reset to binary head

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr_boxref --tag binary --backbone convnext_v2_tiny --backbone-tag imagenet21k --opt adamw --lr 0.0001 --batch-size 2 --epochs 10 --wd 0.0001 --grad-accum-steps 16 --multiscale --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --resume-epoch 0 --reset-head --freeze-body --binary-mode
```

Step 3: Fine-tuning

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr_boxref --tag binary --backbone convnext_v2_tiny --backbone-tag imagenet21k --opt adamw --lr 0.00005 --lr-scheduler step --lr-step-size 10 --lr-step-gamma 0.2 --batch-size 1 --epochs 20 --wd 0.0001 --grad-accum-steps 16 --multiscale --clip-grad-norm 0.1 --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --resume-epoch 0 --freeze-backbone --binary-mode
```

### FCOS: EfficientNet v2 Medium Backbone

Step 1: Warmup

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone efficientnet_v2_m --backbone-tag il-all --backbone-pretrained --freeze-backbone --lr 0.01 --freeze-backbone-bn --batch-size 32 --epochs 2 --wd 0.0001 --fast-matmul --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

Step 2: Training

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone efficientnet_v2_m --backbone-lr 0.001 --lr 0.01 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --freeze-backbone-bn --batch-size 8 --epochs 32 --wd 0.0001 --fast-matmul --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

Step 3: Reset to binary head

Step 4: Fine-tuning
