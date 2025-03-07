# Reference Self-supervised Training Procedure

Before running any training scripts, set the `OMP_NUM_THREADS` environment variable appropriately for your system.

## SSL Pre-training

- [VICReg](#vicreg)

### VICReg

Use `--sync-bn` when batch size is 32 or below.

#### VICReg: EfficientNet v2 Medium

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train_vicreg --network efficientnet_v2_m --opt lars --lr 0.2 --lr-scheduler cosine --warmup-epochs 10 --batch-size 128 --epochs 400 --wd 0.000001 --amp --compile --data-path data/training data/raw_data data/detection_data/training ~/Datasets
```
