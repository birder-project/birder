# Inference

Birder provides powerful command-line tools for running inference using its pretrained models.
This document covers `python -m birder.scripts.predict` for image classification and `python -m birder.scripts.predict_detection` for object detection.
While these scripts serve different purposes, they share many common options for model loading, hardware configuration, and output handling.

## Image Classification Inference

The `python -m birder.scripts.predict` command allows you to perform image classification inference. This versatile tool is designed for classifying single images, entire directories of images, or even WebDataset archives, offering extensive options for visualization, detailed reporting, and various performance optimizations.

### Classification - Basic Usage

To classify images within a directory using a specified network and model tag, use the following command:

```sh
python -m birder.scripts.predict -n <network_name> -t <model_tag> data/my_images
```

For a comprehensive list of all available options and their detailed usage, run:

```sh
python -m birder.scripts.predict --help
```

### Classification - NaFlex Inference at Native Aspect Ratio

NaFlex models can run inference without forcing every image to the same aspect ratio.
Enable this mode explicitly with `--naflex`.
Each image is resized to a patch-aligned resolution that preserves its source aspect ratio as closely as possible.

```sh
python -m birder.scripts.predict \
    --network naflex_vit_b16 \
    --tag <model_tag> \
    --naflex \
    data/my_images
```

In NaFlex mode, `--size` defines the patch-token budget rather than an exact output shape.
For example, `--size 384` with a 16-pixel patch size permits at most 576 image patch tokens, while the actual height and width depend on each image's aspect ratio.
If `--size` is omitted, the budget is derived from the model signature.

Not all `predict` options are supported in NaFlex mode, unsupported flags, such as `--save-features`, will be rejected.

### Classification - Output Files

When running classification prediction, the script can generate several types of outputs depending on the flags you provide.
All outputs are saved under the configured results directory (default: `results/`) with automatically generated file names that encode the model, epoch, image size, number of classes, and other settings (e.g., `resnet_v2_50_200_e0_224px_crop1.0_10000.csv`).
You can also prepend or append text using `--prefix` and `--suffix`.

#### Results Files

`*.csv` or `*_sparse.csv` (depending on `--save-results` or `--save-sparse-results`)

- **Purpose**: These files store the classification results, including predicted labels and probabilities, along with sample paths and if available, ground truth labels.
- **Format**: CSV.
- **Content (for `--save-results`):**
    - `sample`: The path to the input image.
    - `label`: The ground truth label of the image (if available).
    - `prediction_names`: Predicted class name.
    - `<class_name_1>`, `<class_name_2>`, ...: Probability values for each class.
- **Content (for `--save-sparse-results`):** Similar to `--save-results` but only include the top-k predicted class probabilities to conserve space.

##### Note on Results Files

Although the results files are written with a `.csv` extension, they are not standard CSVs.

- The first line contains metadata (used internally by Birder).
- The actual CSV table with column headers begins on the second line.

These files are usually consumed with the [Results](tools/results.md) tool, which automatically handles the metadata.
If you want to load them manually with another library, make sure to skip the first line (e.g., `skip_rows=1` in Polars, `skiprows=1` in Pandas, or an equivalent option in your preferred library).

#### Output (Probabilities)

`*_output.csv` or `*_output.parquet` (with `--save-output`)

- **Purpose**: These files contain the model outputs (typically softmax probabilities) for each sample.
- **Format**: CSV or Parquet, determined by the `--output-format` argument.
- **Optional labels**: Add `--save-labels` to include a `label` column in these files.
- **Content**:
    - `sample`: The path to the input image.
    - `label` (optional): Ground-truth class index from the dataset (if available).
    - `prediction`: The predicted class name (based on the highest probability).
    - `<class_name_1>`, `<class_name_2>`, ...: Columns representing the output (e.g., probability) for each class.

#### Logits

`*_logits.csv` or `*_logits.parquet` (with `--save-logits`)

- **Purpose**: These files store the raw outputs from the final layer of the model before any activation functions are applied.
- **Format**: CSV or Parquet, determined by the `--output-format` argument.
- **Optional labels**: Add `--save-labels` to include a `label` column in these files.
- **Content**:
    - `sample`: The path to the input image.
    - `label` (optional): Ground-truth class index from the dataset (if available).
    - `<class_name_1>`, `<class_name_2>`, ...: Columns representing the logit value for each class.

#### Embeddings

`*_embeddings.csv` or `*_embeddings.parquet` (with `--save-embeddings`)

- **Purpose**: These files contain the embeddings vectors from the model.
- **Format**: CSV or Parquet, determined by the `--output-format` argument.
- **Optional labels**: Add `--save-labels` to include a `label` column in these files.
- **Content**:
    - `sample`: The path to the input image.
    - `label` (optional): Ground-truth class index from the dataset (if available).
    - In CSV format, `0`, `1`, `2`, ...: Columns representing each dimension of the embedding vector.
    - In Parquet format, `embedding`: Column with "Fixed sized list" (array) type.

#### Example Output Files

If you run a command like:

```sh
python -m birder.scripts.predict -n resnet_v2_50 -t my_model_tag data/my_images/ --save-results --save-output --save-embeddings --save-labels --output-format parquet
```

You might expect to see files similar to these in your `results` directory:

- `resnet_v2_50_100_e0_224px_crop1.0_1000.csv` (for `--save-results`)
- `resnet_v2_50_100_e0_224px_crop1.0_1000_output.parquet` (for `--save-output`)
- `resnet_v2_50_100_e0_224px_crop1.0_1000_embeddings.parquet` (for `--save-embeddings`)

## Object Detection Inference

The `python -m birder.scripts.predict_detection` command enables you to perform object detection inference using Birder's pretrained models.
This tool is designed for locating and identifying objects (e.g., birds) within images, providing bounding boxes and class labels.
It supports various input formats, visualization options, and performance optimizations.

### Detection - Basic Usage

To run detection inference on images within a directory using a specified network and its backbone, use the following command:

```sh
python -m birder.scripts.predict_detection -n <network_name> --backbone <backbone_name> data/my_detection_images/
```

### Detection - Sliding Window

Use sliding-window inference for large images where resizing the full image would hide small objects.
It runs native-resolution tiles and merges the tile detections.

```sh
python -m birder.scripts.predict_detection -n <network_name> --backbone <backbone_name> \
  --sliding-window --tile-size 640 --tile-overlap 128 \
  --sliding-window-global-size 800 --sliding-window-merge greedy_nmm \
  data/my_detection_images/
```

When sliding-window inference is enabled, `--batch-size` controls the tile batch size.
It cannot be combined with `--tta`.

For a comprehensive list of all available options and their detailed usage, run:

```sh
python -m birder.scripts.predict_detection --help
```
