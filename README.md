# Birder

An open-source computer vision framework for wildlife image analysis, featuring state-of-the-art models for species classification and detection.

- [Introduction](#introduction)
- [Setup](#setup)
- [Getting Started](#getting-started)
- [Pre-trained Models](#pre-trained-models)
- [Detection](#detection)
- [Licenses](#licenses)
- [Acknowledgments](#acknowledgments)

## Introduction

Birder is an open-source computer vision framework designed for wildlife imagery analysis, offering robust classification and detection capabilities for various species. While initially developed with a focus on avian species, the framework's architecture and methodologies are applicable to a wide range of wildlife computer vision tasks. This project leverages deep neural networks to provide models that can handle real-world data challenges in natural environments.

For comprehensive documentation, tutorials, and more visit the main documentation at [docs/README.md](docs/README.md).

The project features:

- A diverse collection of classification and detection models
- Support for self-supervised pre-training
- Knowledge distillation training (teacher-student)
- Custom utilities and data augmentation techniques
- Comprehensive training scripts
- Advanced error analysis tools
- Extensive documentation and tutorials (hopefully...)

Unlike projects that aim to reproduce ImageNet training results from common papers, Birder is tailored specifically for practical applications in wildlife monitoring, conservation efforts, ecological research, and nature photography.

As Ross Wightman eloquently stated in the [timm README](https://github.com/huggingface/pytorch-image-models#introduction):

> The work of many others is present here. I've tried to make sure all source material is acknowledged via links to github, arXiv papers, etc. in the README, documentation, and code docstrings. Please let me know if I missed anything.

The same principle applies to Birder. We stand on the shoulders of giants in the fields of computer vision, machine learning, and ecology. We've made every effort to acknowledge and credit the work that has influenced and contributed to this project. If you believe we've missed any attributions, please let us know by opening an issue.

## Setup

1. Ensure PyTorch 2.7 is installed on your system

1. Install the latest Birder version:

```sh
pip install birder
```

For detailed installation options, including source installation, refer to our [Setup Guide](docs/getting_started.md#setup).

## Getting Started

![Example](docs/img/example.jpeg)

Check out the Birder Colab notebook for an interactive tutorial.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/birder-project/birder/blob/main/notebooks/getting_started.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Project-blue)](https://huggingface.co/birder-project)

If you prefer a local setup, follow the installation instructions below.

Once Birder is installed, you can start exploring its capabilities.

Birder provides pre-trained models that you can download using the `download-model` tool.
To download a model, use the following command:

```sh
python -m birder.tools download-model mvit_v2_t_il-all
```

Create a data directory and download an example image:

```sh
mkdir data
wget https://huggingface.co/spaces/birder-project/birder-image-classification/resolve/main/Eurasian%20teal.jpeg -O data/img_001.jpeg
```

To classify bird images, use the `birder-predict` script as follows:

```sh
birder-predict -n mvit_v2_t -t il-all --show data/img_001.jpeg
```

For more options and detailed usage of the prediction tool, run:

```sh
birder-predict --help
```

For more detailed usage instructions and examples, please refer to our [documentation](docs/README.md).

## Pre-trained Models

Birder provides a comprehensive suite of pre-trained models for wildlife species classification, with current models specialized for avian species recognition.

To explore the full range of available pre-trained models, use the `list-models` tool:

```sh
python -m birder.tools list-models --pretrained
```

This command displays a catalog of models ready for download.

### Model Nomenclature

The naming convention for Birder models encapsulates key information about their architecture and training approach.

Architecture: The first part of the model name indicates the core neural network structure (e.g., MobileNet, ResNet).

Training indicators:

- intermediate: Signifies models that underwent a two-stage training process, beginning with a large-scale weakly labeled dataset before fine-tuning on the primary dataset
- mim: Indicates models that leveraged self-supervised pre-training techniques, primarily Masked Autoencoder (MAE), prior to supervised training

Other tags:

- quantized: Model that has been quantized to reduce the computational and memory costs of running inference
- reparameterized: Model that has been restructured to simplify its architecture for optimized inference performance

Net Param: The number following the model name (e.g., 50, 1.0, 0.5), called the `net_param`, represents a specific configuration choice for the network. It represents a specific configuration choice for the network, which can affect aspects such as model size or complexity.

Epoch Number (optional): The last part of the model name may include an underscore followed by a number (e.g., `0`, `200`), which represents the epoch.

For instance, *mnasnet_1_intermediate_300* represents a MnasNet model with a `net_param` (alpha value in this case) of 1 that underwent intermediate training and is from epoch 300.

### Self-supervised Image Pre-training

Our pre-training process utilizes a diverse collection of image datasets, combining general imagery with wildlife-specific content.
This approach allows our models to learn rich, general-purpose visual representations before fine-tuning on specific classification tasks.

The pre-training dataset is composed of a mix of general images and bird specific imagery to improve downstream performance on the bird classification tasks.

For detailed information about these datasets, including descriptions, citations, and licensing details, please refer to [docs/public_datasets.md](docs/public_datasets.md).

## Detection

Detection features are currently under development and will be available in future releases.

## Project Status and Contributions

Birder is currently a personal project in active development. As the sole developer, I am focused on building and refining the core functionalities of the framework. At this time, I am not actively seeking external contributors.

However, I greatly appreciate the interest and support from the community. If you have suggestions, find bugs, or want to provide feedback, please feel free to:

- Open an issue in the project's issue tracker
- Use the project and share your experiences
- Star the repository if you find it useful

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

- **Ross Wightman**: His work on [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) greatly inspired the design and approach of Birder.

- **Image Contributors**:
    - Yaron Schmid - from [YS Wildlife](https://www.yswildlifephotography.com/who-we-are)

  for their generous donations of bird photographs.

This project also benefits from numerous open-source libraries and ornithological resources.

If any attribution is missing, please open an issue to let us know.
