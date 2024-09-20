# Birder

Birder is an open-source computer vision framework for wildlife image analysis, focusing on avian species.

* [Introduction](#introduction)
* [Setup](#setup)
* [Getting Started](#getting-started)
* [Pre-trained Models](#pre-trained-models)
* [Detection](#detection)
* [Licenses](#licenses)
* [Acknowledgments](#acknowledgments)

## Introduction

Birder is an open-source computer vision framework designed for wildlife imagery, specifically focused on bird species classification and detection. This project leverages deep neural networks to provide robust models that can handle real-world data challenges.

For comprehensive documentation, tutorials, and more visit the main documentation at [docs/README.md](docs/README.md).

The project features:

* A diverse collection of classification and detection models
* Support for self-supervised pre-training
* Knowledge distillation training (teacher-student)
* Custom utilities and data augmentation techniques
* Comprehensive training scripts
* Advanced error analysis tools
* Extensive documentation and tutorials (hopefully...)

Unlike projects that aim to reproduce ImageNet training results from common papers, Birder is tailored specifically for practical applications in ornithology, conservation, and wildlife photography.

As Ross Wightman eloquently stated in the [timm README](https://github.com/huggingface/pytorch-image-models#introduction):

> The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arXiv papers, etc. in the README, documentation, and code docstrings. Please let me know if I missed anything.

The same principle applies to Birder. We stand on the shoulders of giants in the fields of computer vision, machine learning, and ornithology. We've made every effort to acknowledge and credit the work that has influenced and contributed to this project. If you believe we've missed any attributions, please let us know by opening an issue.

## Setup

1. Ensure PyTorch 2.4 is installed on your system

1. Install the latest Birder version:

```sh
pip install birder
```

For detailed installation options, including source installation, refer to our [Setup Guide](docs/getting_started.md#setup).

## Getting Started

Once Birder is installed, you can start exploring its capabilities.

Birder provides pre-trained models that you can download using the `fetch-model` tool.
To download a model, use the following command:

```sh
python -m birder.tools fetch-model mobilenet_v3_large_1
```

Create a data directory and download an example image:

```sh
mkdir data
wget https://f000.backblazeb2.com/file/birder/data/img_001.jpeg -O data/img_001.jpeg
```

To classify bird images, use the `birder-predict` script as follows:

```sh
birder-predict -n mobilenet_v3_large -p 1 --show data/bird.jpeg
```

For more options and detailed usage of the prediction tool, run:

```sh
birder-predict --help
```

For more detailed usage instructions and examples, please refer to our [documentation](docs/README.md).

## Pre-trained Models

Birder provides a comprehensive suite of pre-trained models for bird species classification.

To explore the full range of available pre-trained models, use the `list-models` tool:

```sh
python -m birder.tools list-models --pretrained
```

This command displays a catalog of models ready for download.

### Model Nomenclature

The naming convention for Birder models encapsulates key information about their architecture and training approach.

Architecture: The first part of the model name indicates the core neural network structure (e.g., MobileNet, ResNet).

Training indicators:

* intermediate: Signifies models that underwent a two-stage training process, beginning with a large-scale weakly labeled dataset before fine-tuning on the primary dataset
* mim: Indicates models that leveraged self-supervised pre-training techniques, primarily Masked Autoencoder (MAE), prior to supervised training

Other tags:

* quantized: Model that has been quantized to reduce the computational and memory costs of running inference
* reparameterized: Model that has been restructured to simplify its architecture for optimized inference performance

Net Param: The number following the model name (e.g., 50, 1.0, 0.5), called the `net_param`, represents a specific configuration choice for the network. It represents a specific configuration choice for the network, which can affect aspects such as model size or complexity.

Epoch Number (optional): The last part of the model name may include an underscore followed by a number (e.g., `_0`, `_200`), which represents the epoch.

For instance, *resnext_50_intermediate_300* represents a ResNeXt model with a `net_param` of 50 that underwent intermediate training and is from epoch 300.

### Self-supervised Image Pre-training

Our pre-training process utilizes a diverse collection of image datasets.
This approach allows our models to learn rich, general-purpose visual representations before fine-tuning on specific bird classification tasks.

The pre-training dataset comprises:

* iNaturalist 2021 (~3.3M)
* WebVision-2.0 (~1.5M random subset)
* imagenet-w21-webp-wds (~1M random subset)
* SA-1B (~220K random subset of 20 chunks)
* NABirds (~48K)
* Birdsnap v1.1 (~44K)
* CUB-200 2011 (~18K)
* The Birder dataset (~2.5M)

Total: ~8.5M images

This carefully curated mix of datasets balances general visual knowledge with domain-specific bird imagery, enhancing the model's overall performance.

For detailed information about these datasets, including descriptions, citations, and licensing details, please refer to [docs/public_datasets.md](docs/public_datasets.md).

## Detection

Detection features are currently under development and will be available in future releases.

For annotation, run the following command:

```sh
labelme --labels ../birder/data/detection_data/classes.txt --nodata --output ../birder/data/detection_data/training_annotations --flags unknown ../birder/data/detection_data/training
```

## Project Status and Contributions

Birder is currently a personal project in active development. As the sole developer, I am focused on building and refining the core functionalities of the framework. At this time, I am not actively seeking external contributors.

However, I greatly appreciate the interest and support from the community. If you have suggestions, find bugs, or want to provide feedback, please feel free to:

* Open an issue in the project's issue tracker
* Use the project and share your experiences
* Star the repository if you find it useful

While I may not be able to incorporate external contributions at this stage, your input is valuable and helps shape the direction of Birder. I'll update this section if the contribution policy changes in the future.

Thank you for your understanding and interest in Birder!

## Licenses

### Code

The code in this project is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.
Some code is adapted from other projects.
There are notices with links to the references at the top of the file or at the specific class/function.
It is your responsibility to ensure compliance with licenses here and conditions of any dependent licenses.

If you think we've missed a reference or a license, please create an issue.

### Pretrained Weights

Some of the pretrained weights available here are pretrained on ImageNet. ImageNet was released for non-commercial research purposes only (<https://image-net.org/download>). It's not clear what the implications of that are for the use of pretrained weights from that dataset. It's best to seek legal advice if you intend to use the pretrained weights in a commercial product.

### Disclaimer

If you intend to use Birder, its pretrained weights, or any associated datasets in a commercial product, we strongly recommend seeking legal advice to ensure compliance with all relevant licenses and terms of use.

It's the user's responsibility to ensure that their use of this project, including any pretrained weights or datasets, complies with all applicable licenses and legal requirements.

## Acknowledgments

Birder owes much to the work of others in computer vision, machine learning, and ornithology.

Special thanks to:

* **Ross Wightman**: His work on [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) greatly inspired the design and approach of Birder.

* **Image Contributors**:
  * Yaron Schmid - from [YS Wildlife](https://www.yswildlifephotography.com/who-we-are)

  for their generous donations of bird photographs.

This project also benefits from numerous open-source libraries and ornithological resources.

If any attribution is missing, please open an issue to let us know.
