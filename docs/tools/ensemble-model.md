# Ensemble Model

The `ensemble-model` tool allows you to create an ensemble model from multiple TorchScript or pt2 models. Ensemble models can improve prediction accuracy by combining the outputs of several individual models.

## Usage

```sh
python -m birder.tools ensemble-model [OPTIONS]
```

To see all available options and get detailed help, run:

```sh
python -m birder.tools ensemble-model --help
```

## Description

This tool enables you to create an ensemble model by combining multiple pre-trained Birder models. The ensemble model averages the predictions of its constituent models, potentially leading to improved overall performance.

Supports both TorchScript and pt2 model formats.

## Notes

* The ensembled model will be saved with the name "ensemble" in the models directory
* All models in the ensemble must have the same class-to-index definition
* If model signatures or RGB values differ among the input models, the tool will use those of the first specified model and log a warning
* The tool ensures that all input models are of the same format (either all TorchScript or all pt2)

## Ensemble Process

The ensemble model works by:

1. Loading all specified models
1. Verifying that all models have the same class-to-index definition
1. Creating a new `Ensemble` module that contains all the input models
1. For each input, the ensemble:
   * Passes the input through each model
   * Stacks the outputs
   * Computes the mean of all outputs
1. The final output is the averaged prediction across all models

For more detailed information about each option and its usage, refer to the help output of the tool.
