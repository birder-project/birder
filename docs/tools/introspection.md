# Introspection

The `introspection` tool provides computer vision explainability methods to visualize and understand what neural networks are learning. This tool helps researchers analyze which parts of an image the model focuses on when making predictions, offering insights into model behavior and decision-making processes.

## Usage

```sh
python -m birder.tools introspection [OPTIONS] IMAGE_PATH
```

To see all available options and get detailed help, run:

```sh
python -m birder.tools introspection --help
```

This will display a comprehensive list of all options and their descriptions, ensuring you have access to the full functionality of the `introspection` tool.

## Description

The tool supports multiple visualization methods:

- **GradCAM (Gradient-weighted Class Activation Mapping)**: Highlights the regions in the image that are most important for the model's prediction by using gradient information from the target layer
- **Guided Backpropagation**: Visualizes which pixels in the input image contribute most to the model's prediction through modified gradient backpropagation
- **Attention Rollout**: For transformer-based models, visualizes the attention flow through the network layers to show which image patches the model attends to
- **Transformer Attribution**: For transformer-based models, computes class-specific attribution maps showing which patches are most relevant for a particular prediction

Key features include:

- **Class-specific visualizations**: Visualize explanations for a specific target class or for the model's predicted class
- **Flexible layer selection**: Choose which layers to analyze for convolutional networks or specify attention layer names for transformers
- **Multiple fusion strategies**: For attention rollout, select how to combine attention heads (mean, max, or min)
- **Architecture support**: Works with both convolutional architectures and transformer-based models

## Notes

- The implementations provided are reference implementations and serve as starting points for explainability analysis
- Many networks will require custom adjustments to work correctly with these methods, particularly regarding layer selection and architecture-specific details
- These tools are not production-quality implementations for all Birder networks and may need adaptation for specific architectures
- Some methods have limitations with certain implementation patterns (for example, guided backpropagation works with activations defined as `nn.Module` objects but may not affect functional calls like `F.relu`)
- For GradCAM, you may need to specify the target block and layer number depending on your network architecture, and use the `--channels-last` flag for models using NHWC layout
- For transformer methods, ensure the attention layer name matches your model's architecture (default is "self_attention")

For more detailed information about each option and its usage, refer to the help output of the tool.
