# Convert Model

The `convert-model` tool allows you to convert PyTorch models to various formats, including TorchScript, TorchScript lite interpreter, pt2 standardized model representation, ONNX and reparameterized models. This tool is essential for deployment in different environments and for optimizing model performance.

## Usage

```sh
python -m birder.tools convert-model [OPTIONS]
```

To see all available options and get detailed help, run:

```sh
python -m birder.tools convert-model --help
```

This will display a comprehensive list of all options and their descriptions, ensuring you have access to the full functionality of the `convert-model` tool.

## Description

Birder provides flexibility in converting models to different formats, each serving specific purposes:

- **TorchScript (--pts)**: For deployment in production environments that support TorchScript
- **TorchScript lite interpreter (--lite)**: For deployment on mobile or edge devices with limited resources
- **pt2 (--pt2)**: The standardized model representation in PyTorch 2.0, offering improved performance and compatibility
- **ONNX (--onnx)**: For cross-platform machine learning interoperability
- **Reparameterized (--reparameterize)**: For optimizing model architecture and potentially improving inference performance

All converted formats, except for the reparameterized option, are standalone, containing both the computation graph and weights, unlike the normal PyTorch (.pt) format. This makes them suitable for deployment without requiring the original model definition.

The pt2 format supports `torch.compile`, enabling further optimizations and potential performance improvements.

## Notes

- Converted models are saved in the appropriate format in the models directory
- When converting detection models, specify both the main network and the backbone architecture
- Only one conversion format (--pts, --lite, --pt2, --onnx, or --reparameterize) can be specified at a time
- The reparameterize option is only available for compatible network architectures

For more detailed information about each option and its usage, refer to the help output of the tool.
