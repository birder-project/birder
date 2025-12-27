# API

## Model Loading

Birder's model loading APIs are designed to cater to different user personas and use cases, providing flexible and intuitive approaches to model initialization.

### Loading Methods

#### `birder.load_pretrained_model`

**Target Audience:** Developers and Experimenters

- Optimized for quick model initialization with pre-trained weights
- Simplified, one-step loading process
- **Limitation:** Exclusively loads pre-existing weights, does not support custom fine-tuned model weights

#### `birder.load_model_with_cfg`

**Target Audience:** Data Scientists

- Inspired by HuggingFace's loading conventions
- Supports loading models from any specified path
- **Flexibility:** Weights loading is optional, allowing configuration retrieval without weight initialization

#### `torch.hub.load`

**Target Audience:** PyTorch Ecosystem Users

- Maintains familiar PyTorch Hub interface with `pretrained` boolean
- Model download governed by `TORCH_HOME` and `torch.hub.set_dir`
- **Important:** Downloads occur outside the standard Birder model directory, potentially impacting built-in scripts

#### `fs_ops.load_model` and `fs_ops.load_detection_model`

**Target Audience:** Advanced Birder Users

- Low-level API providing maximal configuration flexibility
- Requires deeper understanding of Birder's workflow
