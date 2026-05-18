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

## Training Script Customization

For small changes to selected training scripts, use `TrainOverrides` instead of copying training scripts or monkey patching internal functions.
This keeps Birder's CLI arguments, validation, distributed setup, checkpoint handling, and training loop unchanged while replacing selected construction defaults.

Transform overrides receive the parsed training arguments and return the transform that should be used by the
training script. Loader and decoder overrides are called directly for each image.

```python
import torch
from torchvision.transforms import v2

from birder.common import training_utils
from birder.scripts import train
from birder.scripts.train import TrainOverrides


class AddGaussianNoise:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image + torch.randn_like(image) * self.sigma


def training_transform(args):
    return v2.Compose(
        [
            training_utils.get_training_transform(args),
            AddGaussianNoise(0.02),
        ]
    )


if __name__ == "__main__":
    train.main(
        TrainOverrides(training_transform=training_transform)
    )
```

Run the custom script with the same arguments as the standard training script:

```sh
python custom_train.py --network resnet_v2_50 --batch-size 64
```

Distributed training works the same way:

```sh
torchrun --nproc_per_node=4 custom_train.py --network resnet_v2_50 --batch-size 64
```

Import `TrainOverrides` from the training script module being customized.

Supported fields vary by script. Most training scripts support replacing:

- `training_transform`: builds the transform used for training samples
- `image_loader`: loads an image from a filesystem path
- `wds_image_decoder`: decodes a WebDataset image from its sample key and bytes

Classification-style scripts can also support `validation_transform` and scripts with custom batch assembly can expose a `training_collator` override.

For example, a custom MIM entry point can pre-train on log-mel spectrogram tensors using `torchaudio`:

```python
import torch
import torch.nn.functional as F
import torchaudio
import torchvision

from birder.scripts import train_mim
from birder.scripts.train_mim import TrainOverrides


TARGET_SAMPLE_RATE = 16_000


def audio_loader(path: str):
    return torchaudio.load(path)


class WaveformToLogMel:
    def __init__(self, args) -> None:
        self.size = args.size
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=1024,
            hop_length=160,
            n_mels=96,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, sample) -> torch.Tensor:
        waveform, sample_rate = sample
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform[..., :223 * 160]
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)

        spec = self.to_db(self.mel(waveform))
        spec = (spec - spec.mean()) / spec.std().clamp_min(1e-6)

        return spec.squeeze(0)


if __name__ == "__main__":
    torchvision.datasets.folder.IMG_EXTENSIONS = (".wav", ".flac", ".mp3")

    train_mim.main(
        TrainOverrides(
            image_loader=audio_loader,
            training_transform=WaveformToLogMel,
        )
    )
```

Run it like the standard MIM script, using a single input channel:

```sh
python custom_train_msm.py --network mae_vit --encoder vit_reg4_b16 --channels 1 --rgb-mean 0.0 --rgb-std 1.0 --size 96 224 --data-path data/audio
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
