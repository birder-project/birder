# Inference

Birder provides powerful command-line tools for running inference using its pre-trained models.
This document covers the usage of `birder-predict` for image classification and `birder-predict_detection` for object detection.
While these scripts serve different purposes, they share many common options for model loading, hardware configuration, and output handling.

## Image Classification Inference

The `birder-predict` script (or `python -m birder.scripts.predict`) allows you to perform image classification inference. This versatile tool is designed for classifying single images, entire directories of images, or even WebDataset archives, offering extensive options for visualization, detailed reporting, and various performance optimizations.

### Classification - Basic Usage

To classify images within a directory using a specified network and model tag, use the following command:

```sh
birder-predict -n <network_name> -t <model_tag> data/my_images/
```

For a comprehensive list of all available options and their detailed usage, run:

```sh
birder-predict --help
```

## Object Detection Inference

The `birder-predict_detection` script (or `python -m birder.scripts.predict_detection`) enables you to perform object detection inference using Birder's pre-trained models.
This tool is designed for locating and identifying objects (e.g., birds) within images, providing bounding boxes and class labels.
It supports various input formats, visualization options, and performance optimizations.

### Detection - Basic Usage

To run detection inference on images within a directory using a specified network and its backbone, use the following command:

```sh
birder-predict_detection -n <network_name> --backbone <backbone_name> data/my_detection_images/
```

For a comprehensive list of all available options and their detailed usage, run:

```sh
birder-predict_detection --help
```
