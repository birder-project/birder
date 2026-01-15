# Reference Classification Evaluation Training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## Datasets

- [Birder](#birder)
- [BIOSCAN-5M](#bioscan-5m)
- [iNaturalist 2021](#inaturalist-2021)
- [ImageNet 1K](#imagenet-1k)
- [ImageNet 12K](#imagenet-12k)
- [ImageNet 21K](#imagenet-21k)
- [Places 365](#places-365)

### Birder

Intermediate training: first stage - linear probing (quick)

```sh
torchrun --nproc_per_node=2 train.py --network hieradet_small --tag dino-v2-intermediate --reset-head --freeze-body --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode birder --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Fine-tuning, first stage - linear probing, region (quick)

```sh
torchrun --nproc_per_node=2 train.py --network hieradet_small --tag dino-v2-eu-common --reset-head --freeze-body --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode birder --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --data-path data/training_eu-common_packed --val-path data/validation_eu-common_packed
```

Fine-tuning, first stage - linear probing, region (full)

```sh
torchrun --nproc_per_node=2 train.py --network hieradet_small --tag dino-v2-eu-common --reset-head --freeze-body --batch-size 512 --lr 0.1 --wd 0.0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 10 --size 224 --aug-level 1 --rgb-mode birder --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --data-path data/training_eu-common_packed --val-path data/validation_eu-common_packed
```

#### Birder - Attentive Probing (AVG -> APS)

Intermediate training training: first stage - attentive probing (quick)

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_so150m_p14_aps --tag mim-intermediate --reset-head --freeze-body --unfreeze-features --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode none --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --non-strict-weights --wds-info data/intermediate_packed/_info.json --wds-class-file data/intermediate_packed/classes.txt
```

Fine-tuning, first stage - attentive probing, region (quick)

```sh
torchrun --nproc_per_node=2 train.py --network vit_reg4_so150m_p14_aps --tag mim-eu-common --reset-head --freeze-body --unfreeze-features --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode none --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --non-strict-weights --data-path data/training_eu-common_packed --val-path data/validation_eu-common_packed
```

### BIOSCAN-5M

Fine-tuning, first stage - linear probing (quick)

```sh
torchrun --nproc_per_node=2 train.py --network vit_b16_ls --tag franca-bioscan5m --reset-head --freeze-body --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode none --amp --amp-dtype bfloat16 --compile --resume-epoch 0 --data-path ~/Datasets/BIOSCAN-5M/family/training --val-path ~/Datasets/BIOSCAN-5M/family/validation
```

Fine-tuning (family), first stage - linear probing (full)

```sh
torchrun --nproc_per_node=2 train.py --network vit_b16_ls --tag franca-bioscan5m --reset-head --freeze-body --batch-size 512 --lr 0.1 --wd 0.0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 10 --size 224 --aug-level 1 --rgb-mode none --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/BIOSCAN-5M/family/training --val-path ~/Datasets/BIOSCAN-5M/family/validation
```

### iNaturalist 2021

Fine-tuning, first stage - linear probing (quick)

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag mmcr-inat21 --reset-head --freeze-body --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode birder --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

Fine-tuning (family), first stage - linear probing (full)

```sh
torchrun --nproc_per_node=2 train.py --network efficientnet_v2_s --tag mmcr-inat21 --reset-head --freeze-body --batch-size 512 --lr 0.1 --wd 0.0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 10 --size 224 --aug-level 1 --rgb-mode birder --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

#### iNaturalist 2021 - Attentive Probing (AVG -> AP)

Fine-tuning, first stage - attentive probing (quick)

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-inat21 --reset-head --freeze-body --unfreeze-features --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode none --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --non-strict-weights --data-path ~/Datasets/inat2021/train --val-path ~/Datasets/inat2021/val
```

### ImageNet 1K

Fine-tuning, first stage - linear probing (quick)

```sh
torchrun --nproc_per_node=2 train.py --network rdnet_t --tag ibot-imagenet1k --reset-head --freeze-body --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --rgb-mode imagenet --smoothing-alpha 0.1 --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

Fine-tuning, first stage - linear probing (full)

```sh
torchrun --nproc_per_node=2 train.py --network rdnet_t --tag ibot-imagenet1k --reset-head --freeze-body --batch-size 512 --lr 0.1 --wd 0.0 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 90 --warmup-epochs 10 --size 224 --aug-level 1 --rgb-mode imagenet --fast-matmul --compile --save-frequency 1 --resume-epoch 0 --wds --wds-class-file public_datasets_metadata/imagenet-1k-classes.txt --wds-train-size 1281167 --wds-val-size 50000 --data-path ~/Datasets/imagenet-1k-wds/training --val-path ~/Datasets/imagenet-1k-wds/validation
```

### ImageNet 12K

Fine-tuning, first stage - linear probing (quick)

```sh
torchrun --nproc_per_node=2 train.py --network xcit_small12_p16 --tag dino-v1-imagenet12k --reset-head --freeze-body --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --rgb-mode imagenet --smoothing-alpha 0.1 --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --wds --wds-info ~/Datasets/imagenet-12k-wds/_info.json --wds-class-file public_datasets_metadata/imagenet-12k-classes.txt --wds-training-split train
```

### ImageNet 21K

#### ImageNet 21K - Attentive Probing (AVG -> AP)

Fine-tuning, first stage - attentive probing (quick)

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-imagenet21k --reset-head --freeze-body --unfreeze-features --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode none --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --non-strict-weights --wds --wds-info ~/Datasets/imagenet-w21-webp-wds/_info.json --wds-class-file public_datasets_metadata/imagenet-21k-classes.txt --wds-training-split train
```

### Places 365

#### Places 365 - Attentive Probing (AVG -> AP)

Fine-tuning, first stage - linear probing

```sh
torchrun --nproc_per_node=2 train.py --network rope_vit_reg8_b14_ap --tag capi-places365 --reset-head --freeze-body --unfreeze-features --batch-size 384 --opt adamw --lr 0.0005 --lr-scheduler cosine --lr-cosine-min 1e-7 --epochs 5 --size 224 --aug-level 1 --smoothing-alpha 0.1 --rgb-mode none --amp --amp-dtype bfloat16 --compile --save-frequency 1 --resume-epoch 0 --non-strict-weights --data-path ~/Datasets/Places365/training --val-path ~/Datasets/Places365/validation
```
