# Getting Started

This section will guide you through the process of setting up Birder on your system and getting started with basic usage.

## Setup

Birder can be installed either as a package or cloned from Git.

### Option 1: Package Installation (Recommended for Users)

1. Set up and activate a virtual environment:

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
    ```

1. Update pip and wheel in your virtual environment:

    ```sh
    pip install --upgrade pip wheel
    ```

1. Install PyTorch 2.3 or above suitable for your hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

1. Install the `birder` package

    ```sh
    pip install birder
    ```

### Option 2: Cloning from Git (Recommended for Contributors or Advanced Users)

1. Clone the repository:

    ```sh
    git clone https://gitlab.com/birder/birder.git
    cd birder
    ```

1. Set up and activate a virtual environment:

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
    ```

1. Update pip and install wheel

    ```sh
    pip install --upgrade pip wheel
    ```

1. Install PyTorch 2.3 or above suitable for your hardware and drivers (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

    ```sh
    # For CUDA
    pip install --upgrade -r requirements/requirements-pytorch-gpu.txt

    # For CPU
    pip install --upgrade -r requirements/requirements-pytorch-cpu.txt
    ```

1. Install development requirements:

    ```sh
    pip install --upgrade -r requirements/requirements-dev.txt
    ```

### Verifying the Installation

To verify that Birder is installed correctly, run:

```sh
python -c "import birder; print(birder.__version__)"
```

This should print the version of Birder.

## Quick Start Guide
