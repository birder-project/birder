# Convert Model

The `convert-model` tool allows you to convert PyTorch models to various formats, including TorchScript, TorchScript lite interpreter, pt2 standardized model representation, ONNX, Safetensors and reparameterized models. This tool is essential for deployment in different environments and for optimizing model performance.

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
- **Safetensors (--st)**: For a safer and potentially faster model storage format
- **Reparameterized (--reparameterize)**: For optimizing model architecture and potentially improving inference performance

All converted formats, except for the reparameterized option, are standalone, containing both the computation graph and weights, unlike the normal PyTorch (.pt) format. This makes them suitable for deployment without requiring the original model definition.

The pt2 format supports `torch.compile`, enabling further optimizations and potential performance improvements.

## Additional Features

The tool also supports:

- **Resizing models (--resize)**: Adjust the input dimensions of the model
- **Adding configuration (--add-config)**: Add custom configurations to an existing model
- **Adding backbone configuration (--add-backbone-config)**: Add custom configurations to an existing model's backbone
- **Generating configuration files (--config)**: Create a JSON configuration file for the model

## Notes

- Converted models are saved in the models directory with appropriate extensions
- When converting detection models, specify both the main network and the backbone architecture
- Only one conversion format can be specified at a time
- The reparameterize option is only available for compatible network architectures

For more detailed information about each option and its usage, refer to the help output of the tool.
