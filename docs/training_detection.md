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

#### Deformable DETR: RegNet Y 8 GF

Optional warmup

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr --backbone regnet_y_8g --backbone-epoch 0 --freeze-backbone --opt adamw --lr 0.0001 --freeze-backbone-bn --batch-size 4 --epochs 2 --wd 0.0001 --clip-grad-norm 1 --fast-matmul --compile-opt
```

Optional warmup: actual training

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr --backbone regnet_y_8g --backbone-epoch 0 --opt adamw --lr 0.0002 --backbone-lr 0.00002 --lr-scheduler cosine --freeze-backbone-bn --batch-size 4 --epochs 50 --wd 0.0001 --clip-grad-norm 1 --fast-matmul --compile-backbone --compile-opt --resume-epoch 0
```

### Deformable DETR BoxRef

#### Deformable DETR BoxRef: RegNet Y 8 GF

```sh
torchrun --nproc_per_node=2 train_detection.py --network deformable_detr_boxref --backbone regnet_y_8g --backbone-epoch 0 --opt adamw --lr 0.0002 --backbone-lr 0.00002 --lr-scheduler cosine --freeze-backbone-bn --batch-size 4 --epochs 50 --wd 0.0001 --clip-grad-norm 1 --fast-matmul --compile-opt
```

### DETR

#### DETR: RegNet Y 8 GF

Optional warmup

```sh
torchrun --nproc_per_node=2 train_detection.py --network detr --backbone regnet_y_8g --backbone-epoch 0 --freeze-backbone --opt adamw --lr 0.0001 --freeze-backbone-bn --batch-size 8 --epochs 2 --wd 0.0001 --clip-grad-norm 0.1 --fast-matmul --compile-opt
```

Optional warmup: actual training

```sh
torchrun --nproc_per_node=2 train_detection.py --network detr --backbone regnet_y_8g --backbone-epoch 0 --opt adamw --lr 0.0001 --backbone-lr 0.00001 --lr-scheduler cosine --freeze-backbone-bn --batch-size 8 --epochs 300 --wd 0.0001 --clip-grad-norm 0.1 --fast-matmul --compile-opt --resume-epoch 0
```

### EfficientDet

#### EfficientDet D0: EfficientNet v1 B0

```sh
torchrun --nproc_per_node=2 train_detection.py --network efficientdet_d0 --backbone efficientnet_v1_b0 --lr 0.08 --lr-scheduler cosine --sync-bn --warmup-epochs 5 --batch-size 16 --epochs 300 --wd 0.00004 --model-ema --clip-grad-norm 10 --amp --compile-backbone --compile-custom bifpn
```

#### EfficientDet D4: RegNet Y 8 GF

```sh
torchrun --nproc_per_node=2 train_detection.py --network efficientdet_d4 --backbone regnet_y_8g --backbone-epoch 0 --freeze-backbone --lr 0.08 --lr-scheduler cosine --warmup-epochs 2 --freeze-backbone-bn --batch-size 8 --epochs 300 --wd 0.00004 --clip-grad-norm 10 --amp --compile-backbone --compile-custom bifpn
```

### Faster R-CNN

#### Faster R-CNN: EfficientNet v2 Small Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone efficientnet_v2_s --backbone-epoch 0 --lr 0.02 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --freeze-backbone-stages 2 --freeze-backbone-bn --batch-size 16 --epochs 26 --wd 0.0001 --fast-matmul --compile-custom backbone_with_fpn
```

### FCOS

#### FCOS: Tiny ViT 5M Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone tiny_vit_5m --backbone-epoch 0 --lr 0.01 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --freeze-backbone-bn --batch-size 16 --epochs 32 --wd 0.0001 --amp
```

#### FCOS: EfficientNet v2 Small Backbone

Optional warmup

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone efficientnet_v2_s --backbone-pretrained --freeze-backbone --lr 0.01 --freeze-backbone-bn --batch-size 64 --epochs 2 --wd 0.0001 --fast-matmul
```

Optional warmup: actual training

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone efficientnet_v2_s --backbone-pretrained --lr 0.01 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --freeze-backbone-bn --batch-size 16 --epochs 32 --wd 0.0001 --amp --resume-epoch 0
```

### RetinaNet

#### RetinaNet: CSP ResNet 50 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network retinanet --backbone csp_resnet_50 --backbone-epoch 0 --opt adamw --lr 0.0001 --epochs 26 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --freeze-backbone-stages 1 --freeze-backbone-bn --batch-size 32 --wd 0.05 --norm-wd 0 --amp --compile-backbone --compile-opt
```

### SSD

#### SSD: MobileNet v4 Medium Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssd --backbone mobilenet_v4_m --backbone-epoch 0 --freeze-backbone-stages 4 --lr 0.015 --lr-scheduler cosine --batch-size 64 --epochs 300 --wd 0.00002 --fast-matmul
```

### SSDLite

#### SSDLite: MobileNet v2 1 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssdlite --backbone mobilenet_v2 --backbone-param 1 --backbone-epoch 0 --lr 0.15 --lr-scheduler cosine --batch-size 32 --epochs 600 --wd 0.00004 --fast-matmul
```

#### SSDLite: MobileNet v4 Hybrid Medium Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssdlite --backbone mobilenet_v4_hybrid_m --backbone-epoch 0 --opt adamw --lr 0.002 --backbone-lr 0.001 --lr-scheduler cosine --lr-cosine-min 1e-8 --batch-size 32 --warmup-epochs 20 --epochs 600 --wd 0.0001 --fast-matmul --compile-opt
```

### ViTDet

#### ViTDet: ViT SAM b16 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network vitdet --backbone vit_sam_b16 --backbone-epoch 0 --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 8 --warmup-epochs 2 --epochs 100 --wd 0.1 --norm-wd 0 --clip-grad-norm 1 --amp --compile --compile-opt --layer-decay 0.7
```

## Common Dataset Training Scenarios

### COCO

#### Faster R-CNN: CSP ResNet 50 Backbone COCO example

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --tag coco --backbone csp_resnet_50 --backbone-pretrained --backbone-tag imagenet1k --lr 0.02 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --freeze-backbone-stages 2 --freeze-backbone-bn --batch-size 16 --epochs 26 --wd 0.0001 --fast-matmul --compile-custom backbone_with_fpn --save-frequency 1 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

#### Faster R-CNN: ResNet v1 50 Backbone COCO example

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --tag coco --backbone resnet_v1_50 --lr 0.1 --lr-scheduler multistep --lr-steps 352 384 --lr-step-gamma 0.1 --batch-size 16 --epochs 400 --wd 0.0004 --amp --compile-custom backbone_with_fpn --save-frequency 1 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

#### RetinaNet: ResNet v1 50 Backbone COCO example

```sh
torchrun --nproc_per_node=2 train_detection.py --network retinanet --tag coco --backbone resnet_v1_50 --backbone-pretrained --backbone-tag imagenet1k --opt adamw --lr 0.0001 --epochs 26 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --freeze-backbone-stages 1 --freeze-backbone-bn --batch-size 32 --wd 0.05 --norm-wd 0 --amp --compile-backbone --compile-opt --save-frequency 1 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

#### SSD: MobileNet v4 Medium Backbone COCO example

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssd --tag coco --backbone mobilenet_v4_m --backbone-epoch 0 --lr 0.015 --lr-scheduler cosine --batch-size 64 --epochs 300 --wd 0.00002 --fast-matmul --compile-backbone --save-frequency 1 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

## Auto Labeler Training

### FCOS: EfficientNet v2 Medium Backbone

Step 1: Warmup

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone efficientnet_v2_m --backbone-pretrained --backbone-tag il-all --freeze-backbone --lr 0.01 --freeze-backbone-bn --batch-size 32 --epochs 2 --wd 0.0001 --fast-matmul --save-frequency 1 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

Step 2: Training

```sh
torchrun --nproc_per_node=2 train_detection.py --network fcos --backbone efficientnet_v2_m --backbone-lr 0.001 --lr 0.01 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --freeze-backbone-bn --batch-size 8 --epochs 32 --wd 0.0001 --fast-matmul --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```

Step 3: Reset to binary head

Step 4: Fine-tuning
