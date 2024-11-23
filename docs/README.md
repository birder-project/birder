# Birder Documentation

Welcome to Birder, an open-source computer vision framework specialized in wildlife image analysis, with a focus on avian species identification and detection.

## What is Birder?

Birder is a powerful framework that combines state-of-the-art deep learning techniques with domain-specific optimizations for bird species classification and detection.
Whether you're a researcher in ornithology, a conservation scientist, or a wildlife photographer, Birder provides the tools you need to analyze and understand avian imagery.

### Key Features

* **Specialized Models**: Pre-trained models specifically optimized for bird species classification
* **Practical Focus**: Built for real-world applications in ornithology and conservation
* **Comprehensive Tools**: Suite of utilities for training, inference, and error analysis
* **Production Ready**: Integration with TorchServe for deployment
* **Extensive Documentation**: Detailed guides and tutorials for all skill levels

## Quick Start

Get started with Birder in minutes:

```bash
# Install Birder
pip install birder

# Download a pre-trained model
python -m birder.tools fetch-model mobilenet_v4_m_il-common

# Run inference on an image
birder-predict -n mobilenet_v4_m -t il-common --show your_image.jpeg
```

## Why Birder?

Unlike general-purpose computer vision frameworks, Birder is tailored specifically for avian species analysis:

* **Domain Expertise**: Models trained on diverse bird imagery datasets
* **Robust Performance**: Handles real-world challenges in wildlife photography
* **Active Development**: Regular updates and improvements
* **Open Source**: Apache 2.0 licensed code base

## Choose Your Path

### 👉 New to Birder?

Start with our [Getting Started Guide](getting_started.md) for installation and basic usage.

### 🔍 Looking for Pre-trained Models?

Explore our [Pre-trained Models](pretrained_models.md) section for available models and their capabilities.

### 🚀 Ready to Train?

Check out our [Training Guide](training_guide.md) for custom model training.

### 🛠️ Need Tools?

Browse our [Tools and Utilities](tools/index.md) section for helpful command-line tools.

## Table of Contents

1. [Getting Started](getting_started.md)
    * [Setup](getting_started.md#setup)
    * [Quick Start Guide](getting_started.md#quick-start-guide)

1. [Pre-trained Models](pretrained_models.md)

1. [Training](training_guide.md)

1. Inference

1. [Tools and Utilities](tools/index.md)
    * [Convert Model](tools/convert-model.md)
    * [Pack](tools/pack.md)
    * [Results](tools/results.md)

1. [TorchServe Integration](torchserve.md)

1. [Public Datasets](public_datasets.md)

1. [About](about.md)
