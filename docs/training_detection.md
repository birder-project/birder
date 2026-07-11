# Reference Detection Training Procedure

## Object Detection

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

- [Deformable DETR](#deformable-detr)
- [Deformable DETR BoxRef](#deformable-detr-boxref)
- [DETR](#detr)
- [EfficientDet](#efficientdet)
- [Faster R-CNN](#faster-r-cnn)
- [FCOS](#fcos)
- [LW-DETR](#lw-detr)
- [Plain DETR](#plain-detr)
- [RetinaNet](#retinanet)
- [RF-DETR](#rf-detr)
- [RT-DETR v1](#rt-detr-v1)
- [RT-DETR v2](#rt-detr-v2)
- [SSD](#ssd)
- [SSDLite](#ssdlite)
- [ViTDet](#vitdet)
- [YOLO v2](#yolo-v2)
- [YOLO v3](#yolo-v3)
- [YOLO v4](#yolo-v4)
- [YOLO v4 Tiny](#yolo-v4-tiny)

### Deformable DETR

#### Deformable DETR: RegNet Y 4 GF

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network deformable_detr --tag coco --backbone regnet_y_4g --backbone-tag imagenet21k --backbone-pretrained --batch-size 4 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 2 --lr 0.0002 --backbone-lr 0.00002 --wd 0.0001 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --epochs 50 --freeze-backbone-bn --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### Deformable DETR BoxRef

#### Deformable DETR BoxRef: RDNet Small

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network deformable_detr_boxref --tag coco --backbone rdnet_s --backbone-tag vicreg --backbone-pretrained --batch-size 4 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 8 --lr 0.0002 --backbone-lr 0.00002 --wd 0.0001 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --epochs 50 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training multiscale training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network deformable_detr_boxref --tag coco --backbone rdnet_s --backbone-tag vicreg --backbone-pretrained --batch-size 2 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 16 --lr 0.0002 --backbone-lr 0.00002 --wd 0.0001 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --epochs 50 --multiscale --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training multiscale training: head only

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network deformable_detr_boxref --tag coco --backbone rdnet_s --backbone-tag vicreg --reset-head --freeze-body --batch-size 2 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 16 --lr 0.0001 --wd 0.0001 --epochs 10 --multiscale --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --resume-epoch 0
```

### DETR

#### DETR: CSP ResNet 50

Intermediate training multiscale training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network detr --tag coco --backbone csp_resnet_50 --backbone-tag imagenet1k --backbone-pretrained --batch-size 2 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 16 --lr 0.0001 --backbone-lr 0.00001 --wd 0.0001 --lr-scheduler step --lr-step-size 200 --lr-step-gamma 0.1 --epochs 300 --freeze-backbone-bn --aug-type detr --rgb-mode imagenet --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### DETR: Tiny ViT 11M

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network detr --tag coco --backbone tiny_vit_11m --batch-size 64 --opt adamw --clip-grad-norm 0.1 --lr 0.0001 --backbone-lr 0.00001 --wd 0.0001 --lr-scheduler step --lr-step-size 200 --lr-step-gamma 0.1 --epochs 300 --aug-level 3 --amp --amp-dtype bfloat16 --compile --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### DETR: ViT reg4 m16 rms AVG

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network detr --tag coco --backbone vit_reg4_m16_rms_avg --backbone-tag i-jepa-imagenet21k --backbone-pretrained --batch-size 1 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 32 --lr 0.0001 --backbone-lr 0.00001 --wd 0.0001 --lr-scheduler step --lr-step-size 200 --lr-step-gamma 0.1 --epochs 300 --max-size 1152 --multiscale --aug-level 5 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### DETR: RoPEi ViT reg1 s16 pn c1 (PE-Spatial)

Intermediate training (COCO), warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network detr --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-pretrained --freeze-backbone --batch-size 16 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 8 --lr 0.0001 --wd 0.0001 --lr-scheduler step --lr-step-size 200 --lr-step-gamma 0.1 --epochs 300 --size 640 --aug-level 5 --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### EfficientDet

#### EfficientDet D0: EfficientNet v1 B0

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network efficientdet_d0 --backbone efficientnet_v1_b0 --batch-size 32 --clip-grad-norm 10 --lr 0.08 --wd 0.00004 --lr-scheduler cosine --epochs 300 --warmup-epochs 10 --model-ema --amp --amp-dtype bfloat16 --compile
```

#### EfficientDet D3: EfficientNet v1 B3

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network efficientdet_d3 --tag coco --backbone efficientnet_v1_b3 --batch-size 24 --clip-grad-norm 10 --lr 0.08 --wd 0.00004 --lr-scheduler cosine --epochs 300 --warmup-epochs 10 --model-ema --sync-bn --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### EfficientDet D4: RegNet Y 4 GF

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network efficientdet_d4 --tag coco --backbone regnet_y_4g --batch-size 24 --clip-grad-norm 10 --lr 0.08 --wd 0.00004 --lr-scheduler cosine --epochs 300 --warmup-epochs 10 --model-ema --size 640 --batch-multiscale --multiscale-min-size 512 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### Faster R-CNN

#### Faster R-CNN: EfficientNet v2 Small

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network faster_rcnn --backbone efficientnet_v2_s --backbone-epoch 0 --freeze-backbone-stages 2 --batch-size 16 --lr 0.02 --wd 0.0001 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --epochs 26 --freeze-backbone-bn --fast-matmul --compile-backbone
```

#### Faster R-CNN: Hiera AbsWin Base

Intermediate training (COCO) - warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network faster_rcnn --tag coco --backbone hieradet_base --backbone-tag mim --backbone-pretrained --freeze-backbone --batch-size 32 --lr 0.01 --wd 0.0001 --epochs 2 --sync-bn --size 768 --amp --amp-dtype bfloat16 --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network faster_rcnn --tag coco --backbone hieradet_base --backbone-tag mim --backbone-pretrained --batch-size 16 --grad-accum-steps 2 --lr 0.02 --backbone-lr 0.01 --wd 0.0001 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --epochs 26 --size 768 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### FCOS

Can be unstable with AMP

#### FCOS: Tiny ViT 5M

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network fcos --backbone tiny_vit_5m --backbone-epoch 0 --batch-size 16 --lr 0.01 --wd 0.0001 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --epochs 32 --freeze-backbone-bn --fast-matmul
```

#### FCOS: EfficientNet v2 Small

Optional warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network fcos --backbone efficientnet_v2_s --backbone-pretrained --freeze-backbone --batch-size 64 --lr 0.01 --wd 0.0001 --epochs 2 --freeze-backbone-bn --fast-matmul
```

Optional warmup: actual training

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network fcos --backbone efficientnet_v2_s --backbone-pretrained --batch-size 16 --lr 0.01 --wd 0.0001 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --epochs 32 --freeze-backbone-bn --fast-matmul --resume-epoch 0
```

### LW-DETR

#### LW-DETR: ConvNeXt v2 Tiny

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr --tag coco --backbone convnext_v2_tiny --backbone-tag vicreg --backbone-pretrained --batch-size 2 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 8 --lr 0.0001 --backbone-lr 0.00015 --wd 0.0001 --norm-wd 0 --backbone-layer-decay 0.75 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 60 --max-size 1152 --aug-type detr --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### LW-DETR: RoPEi ViT reg1 s16 pn c1 (PE-Spatial)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[5,8,11]}' --backbone-pretrained --batch-size 16 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --backbone-lr 0.00015 --wd 0.0001 --norm-wd 0 --backbone-layer-decay 0.75 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 60 --size 640 --batch-multiscale --multiscale-min-size 384 --aug-level 4 --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### LW-DETR 2-Stage: RoPEi ViT reg1 s16 pn c1 (PE-Spatial)

Intermediate training (Objects365-2020) - Dynamic, warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr_2stg --tag objects365 --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[5,8,11]}' --backbone-pretrained --freeze-backbone --batch-size 32 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --epochs 10 --steps-per-epoch 10000 --size 640 --batch-multiscale --multiscale-min-size 416 --multiscale-max-size 672 --aug-level 4 --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/Objects365-2020/train --val-path ~/Datasets/Objects365-2020/val --coco-json-path ~/Datasets/Objects365-2020/train/zhiyuan_objv2_train.json --coco-val-json-path ~/Datasets/Objects365-2020/val/zhiyuan_objv2_val.json --ignore-file public_datasets_metadata/objects365_ignore.txt
```

Intermediate training (Objects365-2020)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr_2stg --tag objects365 --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[5,8,11]}' --batch-size 16 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --backbone-lr 0.00015 --wd 0.0001 --norm-wd 0 --backbone-layer-decay 0.75 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 60 --model-ema --size 640 --batch-multiscale --multiscale-min-size 416 --multiscale-max-size 672 --aug-level 6 --rgb-mode centered --mosaic-prob 0.5 --mosaic-stop-epoch 36 --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --data-path ~/Datasets/Objects365-2020/train --val-path ~/Datasets/Objects365-2020/val --coco-json-path ~/Datasets/Objects365-2020/train/zhiyuan_objv2_train.json --coco-val-json-path ~/Datasets/Objects365-2020/val/zhiyuan_objv2_val.json --ignore-file public_datasets_metadata/objects365_ignore.txt
```

Fine-tune (COCO) - first stage

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr_2stg --tag objects365-coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[5,8,11]}' --reset-head --freeze-body --batch-size 32 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --epochs 10 --size 640 --batch-multiscale --multiscale-min-size 384 --multiscale-max-size 672 --aug-level 4 --rgb-mode centered --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Fine-tune (COCO) - second stage

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr_2stg --tag objects365-coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[5,8,11]}' --batch-size 16 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.00005 --wd 0.0001 --norm-wd 0 --backbone-layer-decay 0.75 --lr-scheduler cosine --epochs 100 --model-ema --model-ema-warmup 5 --size 640 --batch-multiscale --multiscale-min-size 416 --multiscale-max-size 672 --aug-level 5 --rgb-mode centered --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO) - Dynamic, warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr_2stg --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[5,8,11]}' --backbone-pretrained --freeze-backbone --batch-size 32 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --epochs 10 --size 640 --batch-multiscale --multiscale-min-size 384 --multiscale-max-size 672 --aug-level 4 --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr_2stg --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[5,8,11]}' --batch-size 24 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --backbone-lr 0.00015 --wd 0.0001 --norm-wd 0 --backbone-layer-decay 0.75 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 60 --model-ema --size 640 --batch-multiscale --multiscale-min-size 512 --multiscale-max-size 672 --aug-level 4 --rgb-mode centered --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO) - Native aspect ratio

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr_2stg --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[5,8,11]}' --batch-size 2 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --wd 0.0001 --norm-wd 0 --backbone-layer-decay 0.8 --lr-scheduler step --lr-step-size 20 --lr-step-gamma 0.1 --epochs 30 --model-ema --max-size 1152 --multiscale --aug-level 4 --rgb-mode centered --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### LW-DETR Large: HieraDet Small (SAM 2.1)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network lw_detr_l --tag coco --backbone hieradet_small --backbone-tag sam2_1 --backbone-pretrained --batch-size 2 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 8 --lr 0.0001 --backbone-lr 0.00015 --wd 0.0001 --norm-wd 0 --backbone-layer-decay 0.7 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 60 --max-size 1152 --aug-type detr --rgb-mode imagenet --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### Plain DETR

#### Plain DETR: RoPEi ViT reg1 s16 pn c1 (PE-Spatial)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network plain_detr --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-pretrained --batch-size 2 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 8 --lr 0.0002 --backbone-lr 0.00002 --wd 0.05 --norm-wd 0 --backbone-layer-decay 0.95 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --epochs 50 --max-size 1152 --aug-type detr --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### RetinaNet

#### RetinaNet: ConvNeXt v2 Tiny

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network retinanet --tag coco --backbone convnext_v2_tiny --backbone-tag vicreg --backbone-pretrained --batch-size 2 --opt adamw --opt-fused --lr 0.0001 --wd 0.05 --norm-wd 0 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --epochs 26 --sync-bn --aug-type multiscale --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### RetinaNet: CSP ResNet 50

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network retinanet --backbone csp_resnet_50 --backbone-epoch 0 --freeze-backbone-stages 1 --batch-size 32 --opt adamw --lr 0.0001 --wd 0.05 --norm-wd 0 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --epochs 26 --freeze-backbone-bn --amp --compile --compile-opt
```

### RF-DETR

#### RF-DETR Small: RoPEi ViT reg1 t16 pn c1 (PE-Spatial)

Intermediate training (Objects365-2020) - Dynamic, warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rf_detr_small --tag objects365 --backbone rope_i_vit_reg1_t16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[2,5,8,11]}' --backbone-pretrained --freeze-backbone --batch-size 32 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --wd 0.0001 --norm-wd 0 --epochs 10 --size 576 --batch-multiscale --multiscale-min-size 384 --multiscale-max-size 672 --aug-type fixed_detr --rgb-mode centered --amp --amp-dtype bfloat16 --compile --wds --wds-info ~/Datasets/Objects365-2020-wds/_info.json --wds-class-file public_datasets_metadata/objects365-classes.txt
```

#### RF-DETR: RoPEi ViT reg1 s16 pn c1 (PE-Spatial)

Intermediate training (COCO) - Dynamic, warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rf_detr --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[2,5,8,11]}' --backbone-pretrained --freeze-backbone --batch-size 32 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --wd 0.0001 --norm-wd 0 --epochs 10 --size 576 --batch-multiscale --multiscale-min-size 384 --multiscale-max-size 672 --aug-type fixed_detr --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rf_detr --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-model-config '{"out_indices":[2,5,8,11]}' --batch-size 24 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --backbone-lr 0.00015 --wd 0.0001 --norm-wd 0 --backbone-layer-decay 0.8 --epochs 100 --model-ema --model-ema-decay 0.993 --size 576 --batch-multiscale --multiscale-min-size 384 --multiscale-max-size 672 --aug-type fixed_detr --rgb-mode centered --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### RT-DETR v1

#### RT-DETR v1: HieraDet Small

Intermediate training (COCO) - Original

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rt_detr_v1 --tag coco --backbone hieradet_small --backbone-tag dino-v2 --backbone-pretrained --batch-size 16 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 2 --lr 0.0001 --backbone-lr 0.00001 --wd 0.0001 --norm-wd 0 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 72 --warmup-epochs 4 --model-ema --model-ema-decay 0.9999 --model-ema-warmup 60 --aug-level 5 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO) - Dynamic

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rt_detr_v1 --tag coco --backbone hieradet_small --backbone-tag dino-v2 --backbone-pretrained --batch-size 8 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 4 --lr 0.0001 --backbone-lr 0.00001 --wd 0.0001 --norm-wd 0 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 72 --warmup-epochs 4 --model-ema --model-ema-decay 0.9999 --model-ema-warmup 60 --max-size 1152 --multiscale --aug-level 5 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### RT-DETR v2

#### RT-DETR v2: ConvNeXt v1 Tiny

Intermediate training (COCO) - Original

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rt_detr_v2 --tag coco --backbone convnext_v1_tiny --backbone-tag dino-v2 --backbone-pretrained --batch-size 24 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --backbone-lr 0.00001 --wd 0.0001 --norm-wd 0 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 72 --warmup-epochs 4 --model-ema --model-ema-decay 0.9999 --model-ema-warmup 60 --size 640 --aug-level 5 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO) - Dynamic

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rt_detr_v2 --tag coco --backbone convnext_v1_tiny --backbone-tag dino-v2 --backbone-pretrained --batch-size 24 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 2 --lr 0.0001 --backbone-lr 0.00001 --wd 0.0001 --norm-wd 0 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 72 --warmup-epochs 4 --model-ema --model-ema-decay 0.9999 --model-ema-warmup 60 --size 640 --batch-multiscale --multiscale-min-size 384 --aug-level 5 --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### RT-DETR v2: HieraDet Small (SAM 2.1)

Intermediate training (COCO) - Dynamic, warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rt_detr_v2 --tag coco --backbone hieradet_small --backbone-tag sam2_1 --backbone-pretrained --freeze-backbone --batch-size 64 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --epochs 10 --size 640 --batch-multiscale --multiscale-min-size 384 --aug-level 5 --rgb-mode imagenet --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO) - Dynamic

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rt_detr_v2 --tag coco --backbone hieradet_small --backbone-tag sam2_1 --backbone-pretrained --batch-size 24 --opt adamw --opt-fused --clip-grad-norm 0.1 --grad-accum-steps 2 --lr 0.0001 --backbone-lr 0.00001 --wd 0.0001 --norm-wd 0 --lr-scheduler step --lr-step-size 50 --lr-step-gamma 0.1 --epochs 72 --warmup-epochs 4 --model-ema --model-ema-decay 0.9999 --model-ema-warmup 60 --size 640 --batch-multiscale --multiscale-min-size 384 --aug-level 5 --rgb-mode imagenet --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

IL-Common family - Warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network rt_detr_v2 --tag il-common-family --backbone hieradet_small --backbone-tag sam2_1 --backbone-pretrained --freeze-backbone --batch-size 64 --opt adamw --opt-fused --clip-grad-norm 0.1 --lr 0.0001 --wd 0.0001 --norm-wd 0 --lr-scheduler cosine --epochs 100 --size 640 --batch-multiscale --multiscale-min-size 384 --aug-level 6 --rgb-mode imagenet --mosaic-prob 0.5 --amp --amp-dtype bfloat16 --compile --coco-json-path data/detection_data/training_annotations_il-common_classes_coco.json --coco-val-json-path data/detection_data/validation_annotations_il-common_classes_coco.json --label-mapping data/il-common_family-mapping.json
```

### SSD

#### SSD: MobileNet v4 Medium

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network ssd --backbone mobilenet_v4_m --backbone-epoch 0 --freeze-backbone-stages 4 --batch-size 64 --lr 0.015 --wd 0.00002 --lr-scheduler cosine --epochs 300 --aug-type ssd --fast-matmul
```

### SSDLite

#### SSDLite: MobileNet v2 1

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network ssdlite --backbone mobilenet_v2_1_0 --backbone-epoch 0 --batch-size 32 --lr 0.15 --wd 0.00004 --lr-scheduler cosine --epochs 600 --aug-type ssdlite --fast-matmul
```

#### SSDLite: MobileNet v4 Hybrid Medium

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network ssdlite --backbone mobilenet_v4_hybrid_m --backbone-epoch 0 --batch-size 32 --opt adamw --lr 0.002 --backbone-lr 0.001 --wd 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-8 --epochs 600 --warmup-epochs 20 --aug-type ssdlite --fast-matmul --compile-opt
```

#### SSDLite: RoPEi ViT reg1 s16 pn c1 (PE-Spatial)

Intermediate training (COCO), warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network ssdlite --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-pretrained --freeze-backbone --batch-size 64 --opt adamw --lr 0.001 --wd 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 20 --rgb-mode centered --fast-matmul --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network ssdlite --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 --backbone-tag pe-spatial --backbone-pretrained --batch-size 64 --opt adamw --lr 0.0005 --backbone-lr 1e-5 --wd 0.0001 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 300 --warmup-epochs 10 --aug-type ssdlite --rgb-mode centered --fast-matmul --compile-backbone --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### ViTDet

#### ViTDet: ViT Det m16 rms

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network vitdet --backbone vit_det_m16_rms --backbone-tag i-jepa-imagenet21k --backbone-pretrained --batch-size 16 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.1 --norm-wd 0 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 2 --size 672 --amp --amp-dtype bfloat16 --compile-backbone
```

Intermediate training (Objects365-2020)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network vitdet --tag objects365 --backbone vit_det_m16_rms --backbone-tag i-jepa --backbone-pretrained --batch-size 16 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.1 --norm-wd 0 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 20 --warmup-epochs 2 --size 672 --amp --amp-dtype bfloat16 --compile-backbone --data-path ~/Datasets/Objects365-2020/train --val-path ~/Datasets/Objects365-2020/val --coco-json-path ~/Datasets/Objects365-2020/train/zhiyuan_objv2_train.json --coco-val-json-path ~/Datasets/Objects365-2020/val/zhiyuan_objv2_val.json --ignore-file public_datasets_metadata/objects365_ignore.txt
```

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network vitdet --tag coco --backbone vit_det_m16_rms --backbone-tag i-jepa-imagenet21k --backbone-pretrained --batch-size 16 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.1 --norm-wd 0 --layer-decay 0.9 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 2 --size 672 --amp --amp-dtype bfloat16 --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### ViTDet: ViT SAM b16

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network vitdet --backbone vit_sam_b16 --backbone-epoch 0 --batch-size 8 --opt adamw --clip-grad-norm 1 --lr 0.0001 --wd 0.1 --norm-wd 0 --layer-decay 0.7 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 100 --warmup-epochs 2 --size 672 --amp --compile-backbone --compile-opt
```

### YOLO v2

#### YOLO v2: Darknet 17

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network yolo_v2 --tag coco --backbone darknet_17 --batch-size 64 --lr 0.001 --wd 0.0005 --lr-scheduler multistep --lr-steps 60 90 --lr-step-gamma 0.1 --epochs 160 --warmup-epochs 3 --size 416 --aug-type ssdlite --fast-matmul --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### YOLO v3

#### YOLO v3: Darknet 53

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network yolo_v3 --tag coco --backbone darknet_53 --batch-size 64 --lr 0.001 --wd 0.0005 --lr-scheduler multistep --lr-steps 200 250 --lr-step-gamma 0.1 --epochs 300 --warmup-epochs 5 --size 416 --aug-type yolo --amp --amp-dtype bfloat16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### YOLO v4

#### YOLO v4: CSP ResNet 50

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network yolo_v4 --tag coco --backbone csp_resnet_50 --backbone-tag imagenet1k --backbone-model-config drop_block=0.1 --backbone-epoch 0 --batch-size 32 --grad-accum-steps 2 --lr 0.001 --wd 0.0005 --lr-scheduler multistep --lr-steps 300 350 --lr-step-gamma 0.1 --epochs 400 --warmup-epochs 5 --size 608 --batch-multiscale --multiscale-min-size 384 --aug-type yolo --rgb-mode imagenet --mosaic-prob 0.5 --mosaic-stop-epoch 360 --amp --amp-dtype float16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### YOLO v4: CSP Darknet 53

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network yolo_v4 --tag coco --backbone csp_darknet_53 --backbone-model-config drop_block=0.1 --batch-size 32 --grad-accum-steps 2 --lr 0.001 --wd 0.0005 --lr-scheduler multistep --lr-steps 300 350 --lr-step-gamma 0.1 --epochs 400 --warmup-epochs 5 --size 608 --batch-multiscale --multiscale-min-size 384 --aug-level 5 --mosaic-prob 0.5 --mosaic-stop-epoch 360 --amp --amp-dtype float16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

### YOLO v4 Tiny

#### YOLO v4 Tiny: VoVNet v2 19

Intermediate training (COCO)

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network yolo_v4_tiny --tag coco --backbone vovnet_v2_19 --batch-size 128 --lr 0.001 --wd 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-6 --epochs 600 --warmup-epochs 5 --size 416 --batch-multiscale --multiscale-min-size 320 --aug-type yolo --mosaic-prob 0.5 --mosaic-stop-epoch 540 --amp --amp-dtype float16 --compile --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

## Common Dataset Training Scenarios

### COCO

#### Faster R-CNN: CSP ResNet 50 Backbone COCO example

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network faster_rcnn --tag coco --backbone csp_resnet_50 --backbone-tag imagenet1k --backbone-pretrained --freeze-backbone-stages 2 --batch-size 16 --lr 0.02 --wd 0.0001 --lr-scheduler multistep --lr-steps 16 22 --lr-step-gamma 0.1 --epochs 26 --freeze-backbone-bn --rgb-mode imagenet --fast-matmul --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### Faster R-CNN: ResNet v1 50 Backbone COCO example

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network faster_rcnn --tag coco --backbone resnet_v1_50 --batch-size 16 --lr 0.1 --wd 0.0004 --lr-scheduler multistep --lr-steps 352 384 --lr-step-gamma 0.1 --epochs 400 --aug-type lsj --rgb-mode imagenet --amp --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

#### SSD: MobileNet v4 Medium Backbone COCO example

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network ssd --tag coco --backbone mobilenet_v4_m --backbone-epoch 0 --batch-size 64 --lr 0.015 --wd 0.00002 --lr-scheduler cosine --epochs 300 --aug-type ssd --fast-matmul --compile-backbone --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

## Auto Labeler Training

### Deformable DETR BoxRef: ConvNeXt v2 Tiny

Step 1: Training

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network deformable_detr_boxref --tag coco --backbone convnext_v2_tiny --backbone-tag imagenet21k --backbone-pretrained --batch-size 2 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 16 --lr 0.0002 --backbone-lr 0.00002 --wd 0.0001 --lr-scheduler step --lr-step-size 40 --lr-step-gamma 0.1 --epochs 50 --multiscale --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Step 2: Reset to binary head

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network deformable_detr_boxref --tag binary --backbone convnext_v2_tiny --backbone-tag imagenet21k --reset-head --binary-mode --freeze-body --batch-size 2 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 16 --lr 0.0001 --wd 0.0001 --epochs 10 --multiscale --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --resume-epoch 0
```

Step 3: Fine-tuning

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network deformable_detr_boxref --tag binary --backbone convnext_v2_tiny --backbone-tag imagenet21k --binary-mode --freeze-backbone --batch-size 1 --opt adamw --clip-grad-norm 0.1 --grad-accum-steps 16 --lr 0.00005 --wd 0.0001 --lr-scheduler step --lr-step-size 10 --lr-step-gamma 0.2 --epochs 20 --multiscale --amp --amp-dtype bfloat16 --compile-backbone --compile-opt --resume-epoch 0
```

### FCOS: EfficientNet v2 Medium Backbone

Step 1: Warmup

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network fcos --backbone efficientnet_v2_m --backbone-tag il-all --backbone-pretrained --freeze-backbone --batch-size 32 --lr 0.01 --wd 0.0001 --epochs 2 --freeze-backbone-bn --fast-matmul --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Step 2: Training

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_detection --network fcos --backbone efficientnet_v2_m --batch-size 8 --lr 0.01 --backbone-lr 0.001 --wd 0.0001 --lr-scheduler step --lr-step-size 15 --lr-step-gamma 0.1 --epochs 32 --freeze-backbone-bn --fast-matmul --resume-epoch 0 --data-path ~/Datasets/cocodataset/train2017 --val-path ~/Datasets/cocodataset/val2017 --coco-json-path ~/Datasets/cocodataset/annotations/instances_train2017.json --coco-val-json-path ~/Datasets/cocodataset/annotations/instances_val2017.json
```

Step 3: Reset to binary head

Step 4: Fine-tuning
