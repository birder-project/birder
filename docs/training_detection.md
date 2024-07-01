# Reference Detection Training Procedure

Training script and procedures adapted from PyTorch vision reference
<https://github.com/pytorch/vision/blob/main/references/detection>

Set `OMP_NUM_THREADS`.

## Object Detection

### Faster R-CNN

### Faster R-CNN: MobileNet v3 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone mobilenet_v3 --backbone-param 1 --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100
```

### Faster R-CNN: EfficientNet v2 Small Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone efficientnet_v2 --backbone-param 0 --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100
```

### Faster R-CNN: ResNeXt 101 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network faster_rcnn --backbone resnext --backbone-param 101 --backbone-epoch 0 --freeze-backbone --lr 0.02 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100
```

### RetinaNet

### RetinaNet: ResNeXt 101 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network retinanet --backbone resnext --backbone-param 101 --backbone-epoch 0 --freeze-backbone --lr 0.01 --lr-scheduler step --lr-step-size 5 --lr-step-gamma 0.93 --batch-size 16 --epochs 100
```

### SSD

### SSD: MobileNet v3 Backbone

```sh
torchrun --nproc_per_node=2 train_detection.py --network ssd --backbone mobilenet_v3 --backbone-param 1 --backbone-epoch 0 --freeze-backbone-stages 4 --lr 0.015 --lr-scheduler cosine --batch-size 128 --epochs 300 --wd 0.00002
```
