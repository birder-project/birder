# convert-model

The `convert-model` tool allows you to convert PyTorch models to various formats, including TorchScript, TorchScript lite interpreter, pt2 standardized model representation and ONNX. This tool is essential for deployment in different environments.

## Usage

```sh
python -m birder.tools convert-model [OPTIONS]
```

## Description

This tool provides flexibility in converting Birder models to different formats, each serving specific purposes:

* **TorchScript**: For deployment in production environments that support TorchScript.
* **TorchScript lite interpreter**: For deployment on mobile or edge devices with limited resources.
* **pt2**: The new standardized model representation in PyTorch 2.0, offering improved performance and compatibility.
* **ONNX**: For cross-platform machine learning interoperability.

All converted formats are standalone, containing both the computation graph and weights, unlike the normal PyTorch (.pt) format. This makes them suitable for deployment without requiring the original model definition.

The pt2 format also supports `torch.compile`, enabling further optimizations and potential performance improvements.

## Notes

* The converted models will be saved in the appropriate format in the models directory.
* When converting detection models, make sure to specify both the main network and the backbone architecture.
* Only one conversion format (--pts, --lite, --pt2, or --onnx) can be specified at a time.
