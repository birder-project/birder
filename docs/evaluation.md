# Evaluation

The `birder.eval` module provides a dedicated CLI for model evaluation workflows:

- Classification accuracy checks on labeled datasets
- Adversarial robustness evaluation
- Standardized benchmark runs on external datasets

This module is intentionally separate from `birder-predict` and `birder.tools`.

## Usage

```sh
python -m birder.eval --help
python -m birder.eval <command> --help
```

## Available Commands

- `classification`: evaluate pre-trained classification models on a dataset
- `adversarial`: evaluate robustness of a trained model under adversarial attacks
- Benchmarks: `awa2`, `bioscan5m`, `fishnet`, `flowers102`, `fungiclef`, `nabirds`, `newt`, `plankton`, `plantdoc`, `plantnet`

## Minimal Examples

```sh
# Evaluate pre-trained models on a validation set
python -m birder.eval classification --filter '*eu-common*' --gpu data/validation_eu-common

# Evaluate one trained model with a PGD attack
python -m birder.eval adversarial -n resnet_v2_50 -t il-all -e 100 --method pgd --gpu data/validation_il-all

# Run a benchmark from saved embeddings
python -m birder.eval nabirds --embeddings results/nabirds/*.parquet --dataset-path ~/Datasets/nabirds
```

## Inputs and Outputs

- Core commands (`classification`, `adversarial`) read labeled image directories or WebDataset inputs
- Benchmark commands read feature parquet files (embeddings and/or logits) plus a benchmark dataset path
- Results are written under `results/` in task-specific subdirectories

## Example Workflow: Compare Models and Feature Types

Example: compare `bioclip-v1` vs `bioclip-v2` on [Plankton](https://b2share.eudat.eu/records/xvnrp-7ga56).

```sh
# 1) Export embeddings and logits for v1
python -m birder.scripts.predict -n vit_l14_pn -t bioclip-v1 \
  --gpu --parallel --batch-size 64 --fast-matmul --simple-crop \
  --save-embeddings --save-logits --output-format parquet --prefix plankton \
  ~/Datasets/plankton

# 2) Export embeddings and logits for v2
python -m birder.scripts.predict -n vit_l14_pn -t bioclip-v2 \
  --gpu --parallel --batch-size 64 --fast-matmul --simple-crop \
  --save-embeddings --save-logits --output-format parquet --prefix plankton \
  ~/Datasets/plankton

# 3) Evaluate all outputs with the same Plankton benchmark
python -m birder.eval plankton \
  --embeddings \
  results/plankton*bioclip-v*_embeddings.parquet \
  results/plankton*bioclip-v*_logits.parquet \
  --dataset-path ~/Datasets/plankton --gpu
```

This gives one same-protocol table to compare model version (`v1` vs `v2`) and feature type (`embeddings` vs `logits`).

For `bioclip-v1` and `bioclip-v2`, logits are the CLIP projection vectors.

## Dataset Helpers (Optional)

`birder.datahub.evaluation` provides dataset helper classes used by several benchmarks, including:
`AwA2`, `FishNet`, `FungiCLEF2023`, `NABirds`, `NeWT`, `Plankton`, `PlantDoc`, and `PlantNet`.

Use these helpers if you want programmatic dataset validation/download for supported datasets.

```python
from pathlib import Path

from birder.datahub.evaluation import AwA2

dataset = AwA2(Path("~/Datasets/Animals_with_Attributes2").expanduser(), download=False)
print(dataset.images_dir)
```

For complete options, use `python -m birder.eval <command> --help`.
