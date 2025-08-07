# Environment Variables

Birder recognizes several environment variables that allow users to customize its behavior, configure data paths, and control logging.
This document provides a comprehensive list of these variables and their uses.

## Usage

Environment variables can be set in your shell before running Birder commands or scripts:

```sh
export VARIABLE_NAME=value
python -m birder.scripts....
```

Alternatively, you can set them directly before a command:

```sh
VARIABLE_NAME=value python -m birder.scripts....
```

## Available Environment Variables

**`DATA_DIR`**  
Default root directory for data files and datasets.

**`LOG_LEVEL`**  
Minimum logging level.

**`WDS_SHUFFLE_SIZE`**  
Defines the size of the shuffle buffer when using WebDataset for training.

**`WDS_INITIAL_SIZE`**  
Number of samples to pre-fill WebDataset shuffle buffer.

**`DISABLE_CUSTOM_KERNELS`**  
Set to `1` to disable custom CUDA/CPU kernels. Useful for debugging or compatibility issues.
