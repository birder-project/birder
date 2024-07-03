# Birder

This project aim to classify bird species using deep neural networks.

This is a very early stage of the project, mostly data collection at this point.

List of supported bird species [docs/classes.md](docs/classes.md).

As Ross Wightman wrote at the [timm README](https://github.com/huggingface/pytorch-image-models#introduction):

The work of many others is present here.
I've tried to make sure all source material is acknowledged via links to
github, arXiv papers, etc. in the README, documentation, and code docstrings. Please let me know if I missed anything.

The same applies here.

## Setup

This project can be either installed as a package or cloned form git.

### Package

It's recommended to first update the base pip and wheel packages in your venv

```sh
pip3 install --upgrade pip wheel
```

Next, install PyTorch suitable for you hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

Lastly, install the `birder` package

```sh
pip3 install birder
```

### Clone

```sh
git clone https://gitlab.com/birder/birder.git
```

After cloning the repository, setup up venv and activate it (recommended)

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Update pip and install wheel

```sh
pip3 install --upgrade pip wheel
```

Next, install PyTorch suitable for you hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).
The most common CPU and CUDA can be found in requirements file.

```sh
# For CUDA
pip3 install --upgrade -r requirements/requirements-pytorch-gpu.txt

# For CPU
pip3 install --upgrade -r requirements/requirements-pytorch-cpu.txt
```

Install dev requirements

```sh
pip3 install --upgrade -r requirements/requirements-dev.txt
```

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

1. Build the updated package

    ```sh
    python3 -m build
    ```

1. Test the package

    ```sh
    TODO
    ```

1. Release to PyPI

    ```sh
    twine upload dist/*
    ```
