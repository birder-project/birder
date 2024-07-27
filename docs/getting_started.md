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

1. Install PyTorch 2.3 or above suitable for your hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

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

1. Install PyTorch 2.3 or above suitable for your hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

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

### Simple Image Classification

Get started with Birder by classifying a single image, visualizing the results and exploring model decision-making through introspection techniques.

1. **Download a Pre-trained Model**

        python -m birder.tools fetch-model efficientnet_v2_s

1. **Download a Sample Image**

    Create a data directory and download an example image:

        mkdir data
        wget https://f000.backblazeb2.com/file/birder/data/img_001.jpeg -O data/img_001.jpeg

1. **Classify the Image**

        birder-predict -n efficientnet_v2_s --show data/img_001.jpeg

1. **Explore Model Decision-Making with Introspection**

    To gain insight into the model's decision-making process, we'll use Guided Backpropagation [^1], a technique that visualizes the input features most influential to the classification:

        python -m birder.tool introspection --method guided-backprop --network efficientnet_v2_s --image data/img_001.jpeg

    This command generates a saliency map highlighting the pixels in the input image that most significantly influenced the model's classification decision.

    [^1]: [Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller. (2014). *Striving for Simplicity: The All Convolutional Net*. arXiv:1412.6806](https://arxiv.org/abs/1412.6806)

### Camera Trap Analysis
