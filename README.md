# Birder

This project aim to classify bird species using deep neural networks.

This is a very early stage of the project, mostly data collection at this point.

List of supported bird species [docs/classes.md](docs/classes.md).

As Ross Wightman wrote at the [timm README](https://github.com/huggingface/pytorch-image-models#introduction):

The work of many others is present here.
I've tried to make sure all source material is acknowledged via links to
github, arXiv papers, etc. in the README, documentation, and code docstrings. Please let me know if I missed anything.

The same applies here.

## Setup

After cloning the repository, setup up venv and activate it (recommended).

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Update pip and install wheel

```sh
pip3 install --upgrade pip wheel
```

Install PyTorch for CPU or CUDA

```sh
# For CUDA
pip3 install --upgrade -r requirements/requirements-pytorch-gpu.txt

# For CPU
pip3 install --upgrade -r requirements/requirements-pytorch-cpu.txt
```

Install dev requirements

```sh
pip3 install --upgrade -r requirements/requirements-dev.txt
```

## Trained Models

Classification training procedures can be seen at [docs/training.md](docs/training.md)

TBD

### Image Pre-training

Data used in pre-training:

* iNaturalist 2021 (~3.3M)
* WebVision-2.0 (~1.5M random subset)
* imagenet-w21-webp-wds (~1M random subset)
* SA-1B (~200K random subset of 18 chunks)
* NABirds (~48K)
* Birdsnap v1.1 (~44K)
* CUB-200 2011 (~18K)
* The Birder dataset (~1M)

Total: ~7M images

Dataset information can be found at [public_datasets_metadata/](public_datasets_metadata/)

## Random Stuff

fdupes -r data/training data/validation data/testing data/raw_data

exiftool -all:all= -overwrite_original -ext jpeg .

find training/ -type f -name '*.jpeg' -print0 | parallel -0 'mkdir -p "training_webp/$(dirname {})" && mogrify -format webp -resize "1048576@>" -path "training_webp/$(dirname {})" {}' \;

find . -type f -name '*.*' -not -name '.*' | sed -Ee 's,.*/.+\.([^/]+)$,\1,' | sort | uniq -ci | sort -n

cat training.md | grep -E "^### " | sed -E 's/(#+) (.+)/\1:\2:\2/g' | awk -F ":" '{ gsub(/#/,"  ",$1); gsub(/[ ]/,"-",$3); print $1 "- [" $2 "](#" tolower($3) ")" }'

<https://www.israbirding.com/checklist/>

<https://www.birds.org.il/he/species-families>

python3 -m ipykernel install --user --name birder

cloc --fullpath --not-match-d='data/' --exclude-dir=.mypy_cache,.venv .

cat annotations_status.csv | column -t -s, --table-noextreme 8

## TorchServe

Create model archive file (mar)

```sh
torch-model-archiver --model-name convnext_v2_4 --version 1.0 --handler birder/service/classification.py --serialized-file models/convnext_v2_4_0.pts --export-path ts
```

```sh
torch-model-archiver --model-name convnext_v2_4 --version 1.0 --handler birder/service/classification.py --serialized-file models/convnext_v2_4_0.pt2 --export-path ts --config-file ts/example_config.yaml
```

Run TorchServe

```sh
LOG_LOCATION=ts/logs METRICS_LOCATION=ts/logs torchserve --start --ncs --foreground --ts-config ts/config.properties --model-store ts/ --models convnext_v2_4.mar
```

Verify service is running

```sh
curl http://localhost:8080/ping
```

Run inference

```sh
curl http://localhost:8080/predictions/convnext_v2_4 -F "data=@data/validation/African crake/000001.jpeg"
```

## Detection

For annotation run the following

```sh
labelme --labels ../birder/data/detection_data/classes.txt --nodata --output ../birder/data/detection_data/training_annotations --flags unknown ../birder/data/detection_data/training
```
