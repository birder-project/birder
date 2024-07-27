# TorchServe Integration with Birder

This guide explains how to use TorchServe with Birder models for deployment and inference.

## Table of Contents

1. [Introduction](#introduction)
2. [Preparing the Model](#preparing-the-model)
3. [Running TorchServe](#running-torchserve)
4. [Verifying the Service](#verifying-the-service)
5. [Running Inference](#running-inference)
6. [Docker Deployment](#docker-deployment)

## Introduction

TorchServe is a flexible and easy-to-use tool for serving PyTorch models. This guide will walk you through the process of preparing your Birder model for TorchServe, running the server, and performing inference.

## Preparing the Model

Before serving your model with TorchServe, you need to create a Model Archive (.mar) file. This file contains your model and any additional files needed for inference.

### Creating a Model Archive File

To create the .mar file, you'll need the location of the Python service file. You can obtain this by running the following command:

```sh
python -c "from birder.service import classification; print(classification.__file__)" 2> /dev/null
```

This command will output the path to the classification.py file in your Birder installation.

Now, use the `torch-model-archiver` command to create the .mar file. Here are two examples:

1. Basic usage:

        torch-model-archiver --model-name convnext_v2_tiny --version 1.0 --handler $(python -c "from birder.service import classification; print(classification.__file__)" 2> /dev/null) --serialized-file models/convnext_v2_tiny.pts --export-path ts

1. With a configuration file:

        torch-model-archiver --model-name convnext_v2_tiny --version 1.0 --handler $(python -c "from birder.service import classification; print(classification.__file__)" 2> /dev/null) --serialized-file models/convnext_v2_tiny.pt2 --export-path ts --config-file ts/example_config.yaml

The `$(python -c "..." 2> /dev/null)` part dynamically retrieves the correct path to the classification.py file (birder/service/classification.py at the repository).

Replace the model name, version, and file paths as necessary for your specific Birder model.

## Running TorchServe

Once you have your .mar file, you can start TorchServe using the following command:

```sh
LOG_LOCATION=ts/logs METRICS_LOCATION=ts/logs torchserve --start --ncs --foreground --ts-config ts/config.properties --model-store ts/ --models convnext_v2_4.mar
```

This command:

- Sets log and metrics locations
- Starts TorchServe in the foreground
- Uses a configuration file
- Specifies the model store directory
- Loads the convnext_v2_tiny model

## Verifying the Service

To ensure that TorchServe is running correctly, you can send a ping request:

```sh
curl http://localhost:8080/ping
```

If the service is running, you should receive a response indicating that the server is alive.

## Running Inference

To perform inference on an image using your deployed model, use the following curl command:

```sh
curl http://localhost:8080/predictions/convnext_v2_4 -F "data=@data/validation/African crake/000001.jpeg"
```

Replace the image path with the path to your test image.

## Docker Deployment

For containerized deployment, you can use the official PyTorch TorchServe Docker image:

```sh
docker run --rm -it -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -v $(pwd)/ts:/home/model-server/model-store:ro pytorch/torchserve:0.11.0-cpu torchserve --model-store /home/model-server/model-store --models convnext_v2_4.mar
```

This command:

- Runs a TorchServe container
- Maps necessary ports
- Mounts your local model store to the container
- Starts TorchServe with your Birder model

Remember to adjust the Docker image tag and model name as needed for your specific setup.
