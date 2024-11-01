# Reference Detection Training Procedure

Training script and procedures adapted from PyTorch vision reference
<https://github.com/pytorch/vision/blob/main/references/detection>

## Object Detection

### DETR

#### DETR: EfficientNet v1 B0

```sh
torchrun --nproc_per_node=2 train_detection.py --network detr --backbone efficientnet_v1_b0 --backbone-epoch 0 --freeze-backbone --opt adamw --lr 0.0001 --lr-scheduler cosine --freeze-backbone-bn --batch-size 32 --epochs 300 --wd 0.0001 --clip-grad-norm 0.1 --fast-matmul
```

### EfficientDet

#### EfficientDet D0: EfficientNet v1 B0

```sh
torchrun --nproc_per_node=2 train_detection.py --network efficientdet_d0 --backbone efficientnet_v1_b0 --backbone-epoch 0 --freeze-backbone --lr 0.08 --lr-scheduler cosine --warmup-epochs 2 --freeze-backbone-bn --batch-size 32 --epochs 300 --wd 0.00004 --clip-grad-norm 10 --amp --compile
```

#### EfficientDet D4: RegNet Y 8 GF

```sh
torchrun --nproc_per_node=2 train_detection.py --network efficientdet_d4 --backbone regnet_y_8g --backbone-epoch 0 --freeze-backbone --lr 0.08 --lr-scheduler cosine --warmup-epochs 2 --freeze-backbone-bn --batch-size 8 --epochs 300 --wd 0.00004 --clip-grad-norm 10 --amp --compile
```

### Faster R-CNN

#### Faster R-CNN: MobileNet v3 Large Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone mobilenet_v3_large --backbone-param 1 --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --freeze-backbone-bn --batch-size 16 --epochs 100
```

#### Faster R-CNN: EfficientNet v2 Small Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone efficientnet_v2_s --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --freeze-backbone-bn --batch-size 16 --epochs 100 --fast-matmul --compile
```

#### Faster R-CNN: ResNeXt 101 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone resnext_101 --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --freeze-backbone-bn --batch-size 16 --epochs 100 --amp --compile
```

### RetinaNet

#### RetinaNet: ResNeXt 101 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network retinanet --backbone resnext_101 --backbone-epoch 0 --freeze-backbone --lr 0.01 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100 --amp --compile
```

### SSD

#### SSD: MobileNet v4 Medium Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssd --backbone mobilenet_v4_m --backbone-param 1 --backbone-epoch 0 --freeze-backbone-stages 4 --lr 0.015 --lr-scheduler cosine --batch-size 128 --epochs 300 --wd 0.00002
```

### ViTDet

#### ViTDet: ViT SAM b16 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network vitdet --backbone vit_sam_b16 --backbone-epoch 0 --opt adamw --lr 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --batch-size 8 --warmup-epochs 2 --epochs 100 --wd 0.1 --norm-wd 0 --clip-grad-norm 1 --amp --compile --layer-decay 0.7
```

## Common Dataset Training Scenarios

### COCO

#### SSD: MobileNet v4 Medium Backbone COCO example

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssd --tag coco --backbone mobilenet_v4_m --backbone-param 1 --backbone-epoch 0 --freeze-backbone-stages 4 --lr 0.015 --lr-scheduler cosine --batch-size 128 --epochs 300 --wd 0.00002 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json --class-file public_datasets_metadata/coco-classes.txt
```
