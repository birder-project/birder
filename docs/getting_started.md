# Getting Started

This section will guide you through the process of setting up Birder on your system and getting started with basic usage.

## Setup

Birder can be installed either as a package or cloned from Git.

### Option 1: Package Installation (Recommended for Users)

1. Set up and activate a virtual environment:

        python -m venv .venv
        source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

1. Update pip and wheel in your virtual environment:

        pip install --upgrade pip wheel

1. Install PyTorch 2.7 or above suitable for your hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

1. Install the `birder` package

        pip install birder

### Option 2: Cloning from Git (Recommended for Contributors or Advanced Users)

1. Clone the repository:

        git clone https://gitlab.com/birder/birder.git
        cd birder

1. Set up and activate a virtual environment:

        python -m venv .venv
        source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

1. Update pip and install wheel

        pip install --upgrade pip wheel

1. Install PyTorch 2.7 or above suitable for your hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

        # For CUDA
        pip install --upgrade -r requirements/requirements-pytorch-gpu.txt

        # For CPU
        pip install --upgrade -r requirements/requirements-pytorch-cpu.txt

1. Install development requirements:

        pip install --upgrade -r requirements/requirements-dev.txt

### Verifying the Installation

To verify that Birder is installed correctly, run:

```sh
python -c "import birder; print(birder.__version__)"
```

This should print the version of Birder.

## Quick Start Guide

### Image Classification (CLI)

Get started with Birder by classifying a single image, visualizing the results and exploring model decision-making through introspection techniques.

1. **Download a Pre-trained Model**

        python -m birder.tools download-model mvit_v2_t_il-all

1. **Download a Sample Image**

    Create a data directory and download an example image:

        mkdir data
        wget https://huggingface.co/spaces/birder-project/README/resolve/main/img_001.jpeg -O data/img_001.jpeg

1. **Classify the Image**

        birder-predict -n mvit_v2_t -t il-all --show data/img_001.jpeg

1. **Explore Model Decision-Making with Introspection**

    To gain insight into the model's decision-making process, we'll use Guided Backpropagation [^1], a technique that visualizes the input features most influential to the classification:

        python -m birder.tools introspection --method guided-backprop --network mvit_v2_t -t il-all data/img_001.jpeg

    This command generates a saliency map highlighting the pixels in the input image that most significantly influenced the model's classification decision.

    [^1]: [Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller. (2014). *Striving for Simplicity: The All Convolutional Net*. arXiv:1412.6806](https://arxiv.org/abs/1412.6806)

### Image Classification (API)

Birder provides a flexible Python API for performing bird image classification. Following the CLI example, here's how to perform inference programmatically:

```python
import birder
from birder.inference.classification import infer_image

# Load a pre-trained model
(net, model_info) = birder.load_pretrained_model("mvit_v2_t_il-all", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Perform inference on an image
image = "data/img_001.jpeg"  # Path to your image
(out, _) = infer_image(net, image, transform)
```

Alternatively, you can load Birder models directly using Torch Hub:

Notes:

- By default the model will be downloaded into `$TORCH_HOME` and not into the standard Birder directories.
- Replace all dashes with underscores as Torch Hub define all entry points as functions

```python
import torch

# Load a model using Torch Hub
net = torch.hub.load("birder-project/birder", "mvit_v2_t_il_all")
```
