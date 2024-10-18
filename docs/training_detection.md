# Reference Detection Training Procedure

Training script and procedures adapted from PyTorch vision reference
<https://github.com/pytorch/vision/blob/main/references/detection>

## Object Detection

### EfficientDet

#### EfficientDet: EfficientNet v1 B0

```sh
torchrun --nproc_per_node=2 train_detection.py --network efficientdet_d0 --backbone efficientnet_v1_b0 --backbone-epoch 0 --freeze-backbone --lr 0.08 --lr-scheduler cosine --warmup-epochs 2 --batch-size 8 --epochs 300 --wd 0.00004 --clip-grad-norm 10 --amp
```

### Faster R-CNN

### Faster R-CNN: MobileNet v3 Large Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone mobilenet_v3_large --backbone-param 1 --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100
```

### Faster R-CNN: EfficientNet v2 Small Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone efficientnet_v2_s --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100
```

### Faster R-CNN: ResNeXt 101 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone resnext_101 --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100
```

### RetinaNet

### RetinaNet: ResNeXt 101 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network retinanet --backbone resnext_101 --backbone-epoch 0 --freeze-backbone --lr 0.01 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100
```

### SSD

### SSD: MobileNet v3 Large Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssd --backbone mobilenet_v3_large --backbone-param 1 --backbone-epoch 0 --freeze-backbone-stages 4 --lr 0.015 --lr-scheduler cosine --batch-size 128 --epochs 300 --wd 0.00002
```
