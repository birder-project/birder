# Training Scripts

Birder provides several training scripts to handle different model training scenarios. This guide explains the available training scripts and focuses on the important features of the main classification training script.

## Available Training Scripts

Birder includes the following main training scripts:

1. **Image Classification (`train.py`)**: Train models for image classification
2. **Object Detection (`train_detection.py`)**: Train models that can locate and identify objects within images
3. **Knowledge Distillation (`train_kd.py`)**: Train smaller models by transferring knowledge from larger models
4. **Masked Image Modeling (`train_mim.py`)**: Pre-train models using self-supervised masked image modeling techniques

Additionally, there are specialized training scripts for self-supervised learning methods like DINOv1 (`train_dino_v1.py`).

All scripts support a comprehensive set of options which can be viewed using the `--help` flag:

```sh
python -m birder.scripts.train --help
```

## General Usage

Training scripts can be used either directly or through the Birder package.
Direct execution assumes you're running from the repo root.

```sh
# Direct execution
python train.py --network densenet_161 --batch-size 64

# As module
python -m birder.scripts.train --network regnet_y_8g --batch-size 64
```

For distributed training with multiple GPUs, use `torchrun`:

```sh
torchrun --nproc_per_node=2 train.py --network resnet_v2_50 --batch-size 64
```

You can also run the same command as a module:

```sh
torchrun --nproc_per_node=2 -m birder.scripts.train --network resnet_v2_50 --batch-size 64
```

## Key Features

### WebDataset Integration

The training scripts support WebDataset, a format optimized for efficient deep learning training. This is particularly useful for large datasets or training on cloud infrastructure.

To enable WebDataset, use the `--wds` flag. The WebDataset integration in Birder is flexible and supports various configurations:

#### Directory-based WebDataset

When providing a directory path, Birder will:

- Scan the directory for tar files matching the specified suffix (using the split arguments)
- Automatically determine the dataset size by scanning the tar files (if size isn't explicitly specified)

```sh
python train.py --network resnet_v2_50 \
    --wds \
    --data-path /path/to/webdataset/train/directory \
    --val-path /path/to/webdataset/val/directory \
    --wds-class-file /path/to/classes.txt
```

#### Braced Notation Path

You can also use braced notation for tar file paths:

```sh
python train.py --network resnet_v2_50 \
    --wds \
    --data-path "/path/to/shards/train-{000000..000099}.tar" \
    --val-path "/path/to/shards/validation-{000000..000099}.tar" \
    --wds-class-file /path/to/classes.txt
```

#### WebDataset Info File

If available, you can provide a WebDataset info file:

```sh
python train.py --network resnet_v2_50 \
    --wds \
    --wds-info /path/to/_info.json \
    --wds-class-file /path/to/classes.txt
```

The info file contains metadata about the dataset, including shard locations and sizes. If you specify `--wds-train-size` or `--wds-val-size`, these values will take precedence over what's in the info file.

Note: A class file must always be provided when using WebDataset. This file maps class indices to class names.

#### Mixed Precision Training

Enable mixed precision training for faster training with reduced memory usage:

```sh
python train.py --network resnet_v2_50 --amp
```

You can choose between `float16` and `bfloat16` precision with `--amp-dtype`.

#### Model Compilation

Model compilation can significantly improve training speed:

```sh
python train.py --network resnet_v2_50 --compile
```

#### Gradient Accumulation

Train with effectively larger batch sizes on limited GPU memory:

```sh
python train.py --network resnet_v2_50 --batch-size 32 --grad-accum-steps 4  # Effective batch size: 128
```

## Other Training Scripts

While this documentation focuses on `train.py`, similar principles apply to other training scripts. For detailed information on these scripts, run:

```sh
python -m birder.scripts.train_detection --help
python -m birder.scripts.train_kd --help
python -m birder.scripts.train_mim --help
```

For detailed examples of using these scripts, please refer to the example configurations in the project repository.
