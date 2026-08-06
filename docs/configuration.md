# Configuration

Birder can be configured with a local `birder.toml` file and environment variables.
Use these for defaults that should apply across commands and scripts.

## Local Configuration File

Birder reads `birder.toml` from the current working directory.
The file can override uppercase settings from `birder.conf.settings`.

```toml
BASE_DIR = "/mnt/birder"
LOG_LEVEL = "DEBUG"
TOP_K = 5
```

Path settings can be written as strings.
Derived paths follow their parent setting unless they are set explicitly.

Common settings:

**`BASE_DIR`**  
Root directory for data, models, training runs and results.

**`DATA_DIR`**  
Root directory for data files and datasets.

**`MODELS_DIR`**  
Directory for model checkpoints.

**`TRAINING_RUNS_PATH`**  
Directory for training logs.

**`RESULTS_DIR`**  
Directory for inference and evaluation outputs.

**`LOG_LEVEL`**  
Minimum logging level.

**`TOP_K`**  
Number of predictions shown in classification visualizations and used for top-k accuracy.

**`MAX_DETECTIONS`**  
COCO max detection thresholds for detection metrics.

For settings that support both mechanisms, environment variables are applied after `birder.toml`.
Currently this applies to `DATA_DIR`, `MODELS_DIR`, `TRAINING_RUNS_PATH`, `RESULTS_DIR` and `LOG_LEVEL`.

## Environment Variables

Environment variables can be set in your shell before running Birder commands or scripts:

```sh
export VARIABLE_NAME=value
python -m birder.scripts....
```

Alternatively, you can set them directly before a command:

```sh
VARIABLE_NAME=value python -m birder.scripts....
```

### Available Environment Variables

**`DATA_DIR`**  
Default root directory for data files and datasets.

**`MODELS_DIR`**  
Directory for model checkpoints.

**`TRAINING_RUNS_PATH`**  
Directory for training logs.

**`RESULTS_DIR`**  
Directory for inference and evaluation outputs.

**`LOG_LEVEL`**  
Minimum logging level.

**`WDS_SHUFFLE_SIZE`**  
Defines the size of the shuffle buffer when using WebDataset for training.

**`WDS_INITIAL_SIZE`**  
Number of samples to pre-fill WebDataset shuffle buffer.

**`DISABLE_CUSTOM_KERNELS`**  
Set to `1` to disable custom CUDA/CPU kernels. Useful for debugging or compatibility issues.

Individual kernels can be disabled while leaving the others enabled with the `DISABLE_CUSTOM_KERNELS_<KERNEL_NAME>` pattern, where `<KERNEL_NAME>` is the uppercase kernel identifier.
Set the relevant variable to `1` before loading the kernel.

**`COMPILE_RECOMPILE_LIMIT`**  
Overrides `torch.compiler.config.recompile_limit` in Birder training scripts.
If set, this takes precedence over `--compile-recompile-limit`.

**`COMPILE_ACCUMULATED_RECOMPILE_LIMIT`**  
Overrides `torch.compiler.config.accumulated_recompile_limit` in Birder training scripts.
If set, this takes precedence over `--compile-accumulated-recompile-limit`.

### Testing Environment Variables

These environment variables control which tests are executed when running the Birder test suite. They are intended for use during development and continuous integration to selectively enable slower or more resource-dependent tests.

**`SLOW_TESTS`**  
When set to a truthy value (for example `1`), enables execution of tests that are marked as slow.
If not set, slow tests are skipped to keep test runs fast by default.

**`NETWORK_TESTS`**  
When set to a truthy value (for example `1`), enables tests that require network access.
If not set, tests that depend on external network resources are skipped.
