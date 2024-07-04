# Birder

* [Introduction](#introduction)
* [Setup](#setup)

## Introduction

Birder is an open-source computer vision framework designed for wildlife imagery, with a specific focus on bird species classification and detection. This project leverages deep neural networks to provide robust models capable of handling real-world data challenges.

The project features:

* A diverse collection of classification and detection models
* Support for self-supervised pretraining
* Knowledge distillation training (teacher-student)
* Custom utilities and data augmentation techniques
* Comprehensive training scripts
* Advanced error analysis tools
* Extensive documentation and tutorials (hopefully...)

Unlike projects that aim to reproduce ImageNet training results from common papers, Birder is tailored for practical applications in ornithology, conservation, and wildlife photography.

For a complete list of supported bird species, please refer to [docs/classes.md](docs/classes.md).

As Ross Wightman eloquently stated in the [timm README](https://github.com/huggingface/pytorch-image-models#introduction):

> The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arXiv papers, etc. in the README, documentation, and code docstrings. Please let me know if I missed anything.

The same principle applies to Birder. We stand on the shoulders of giants in the fields of computer vision, machine learning, and ornithology. We've made every effort to acknowledge and credit the work that has influenced and contributed to this project. If you believe we've missed any attributions, please let us know by opening an issue.

## Setup

Birder can be installed either as a package or cloned from git.

### Option 1: Package Installation (Recommended for Users)

1. Update pip and wheel in your virtual environment:

    ```sh
    pip3 install --upgrade pip wheel
    ```

1. Install PyTorch: choose the version suitable for your hardware and drivers from PyTorch's official website [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

1. Install the `birder` package

    ```sh
    pip3 install birder
    ```

### Option 2: Cloning from Git (Recommended for Contributors or Advanced Users)

1. Clone the repository:

    ```sh
    git clone https://gitlab.com/birder/birder.git
    ```

1. Set up and activate a virtual environment:

    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

1. Update pip and install wheel

    ```sh
    pip3 install --upgrade pip wheel
    ```

1. Install PyTorch suitable for your hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

    ```sh
    # For CUDA
    pip3 install --upgrade -r requirements/requirements-pytorch-gpu.txt

    # For CPU
    pip3 install --upgrade -r requirements/requirements-pytorch-cpu.txt
    ```

1. Install development requirements:

    ```sh
    pip3 install --upgrade -r requirements/requirements-dev.txt
    ```

### Verifying the Installation

To verify that Birder is installed correctly, run:

```sh
python3 -c "import birder; print(birder.__version__)"
```

This should print the version of Birder.

## Getting Started

TBD

## Trained Models

Classification training procedures can be seen at [docs/training.md](docs/training.md)

TBD

### Image Pre-training

Data used in pre-training:

* iNaturalist 2021 (~3.3M)
* WebVision-2.0 (~1.5M random subset)
* imagenet-w21-webp-wds (~1M random subset)
* SA-1B (~200K random subset of 18 chunks)
* NABirds (~48K)
* Birdsnap v1.1 (~44K)
* CUB-200 2011 (~18K)
* The Birder dataset (~1M)

Total: ~7M images

Dataset information can be found at [public_datasets_metadata/](public_datasets_metadata/)

## TorchServe

Create model archive file (mar)

```sh
torch-model-archiver --model-name convnext_v2_4 --version 1.0 --handler birder/service/classification.py --serialized-file models/convnext_v2_4_0.pts --export-path ts
```

```sh
torch-model-archiver --model-name convnext_v2_4 --version 1.0 --handler birder/service/classification.py --serialized-file models/convnext_v2_4_0.pt2 --export-path ts --config-file ts/example_config.yaml
```

Run TorchServe

```sh
LOG_LOCATION=ts/logs METRICS_LOCATION=ts/logs torchserve --start --ncs --foreground --ts-config ts/config.properties --model-store ts/ --models convnext_v2_4.mar
```

Verify service is running

```sh
curl http://localhost:8080/ping
```

Run inference

```sh
curl http://localhost:8080/predictions/convnext_v2_4 -F "data=@data/validation/African crake/000001.jpeg"
```

## Detection

For annotation run the following

```sh
labelme --labels ../birder/data/detection_data/classes.txt --nodata --output ../birder/data/detection_data/training_annotations --flags unknown ../birder/data/detection_data/training
```

## Release

1. Make sure the full CI passes

   ```sh
   inv ci
   ```

1. Update CHANGELOG.

1. Bump version (`--major`, `--minor` or `--patch`)

    ```sh
    bumpver update --patch
    ```

1. Review the commit and tag and push.

1. Test the package

    ```sh
    docker build -f docker/test.Dockerfile . -t birder-package-test
    docker run birder-package-test:latest
    ```

1. Release to PyPI

    ```sh
    twine upload dist/*
    ```
