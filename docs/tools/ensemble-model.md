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

* The ensembled model will be saved with the name "ensemble"
* All models in the ensemble must have the same class-to-index definition
* If model signatures or RGB values differ, the tool will use those of the first specified model and log a warning
