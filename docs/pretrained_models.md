# Pre-trained Models

Birder provides a variety of pre-trained models optimized for different tasks, regions, and computational requirements. This guide explains our model taxonomy and helps you choose the right model for your needs.

## Model Tasks

Birder models are designed for several specific tasks:

### Classification Models

The primary task type, focused on identifying bird species from images. Classification models can be listed using:

```sh
python -m birder.tools list-models --classification
```

### Detection Models

Models that can locate and identify birds within images. These models handle multiple birds per image and provide bounding boxes. View available detection models with:

```sh
python -m birder.tools list-models --detection
```

## Model Naming Conventions

Our model names follow a systematic pattern that encodes important information about their training and capabilities:

### Training Indicators

- **intermediate**: Models trained in two stages - first on a large weakly-labeled dataset, then fine-tuned on curated data
- **mim**: Models that utilized masked image modeling pre-training
- **quantized**: Models optimized for reduced memory and computation requirements
- **reparameterized**: Models restructured for optimized inference

### Geographical Tags

Indicates the geographical region or scope of the training data:

- **il-common**: Specialized for common bird species in Israel
- **il-all**: Covers all bird species found in Israel
- **arabian-peninsula**: Comprehensive coverage of bird species in the Arabian Peninsula
- **eu-all**: Trained on all European bird species

## Choosing a Model

Consider these factors when selecting a model:

1. **Geographical Relevance**: Choose models with tags matching your region for best accuracy
2. **Computational Resources**:
    - For mobile/edge devices: Consider `quantized` models
    - For server deployment: Full models may provide better accuracy
3. **Task Requirements**:
    - Single bird identification: Use classification models
    - Multiple birds per image: Choose detection models
4. **Dataset Size**:
    - Small datasets: Consider `mim` or `intermediate` models for better generalization
    - Large datasets: Any model type suitable

## Example Usage

List all available pre-trained models:

```bash
python -m birder.tools list-models --pretrained
```

Get detailed model information:

```bash
python -m birder.tools list-models --pretrained --verbose
```

Filter models by pattern:

```bash
python -m birder.tools list-models --pretrained --filter "*mobile*"
```

Fetch specific model:

```bash
python -m birder.tools fetch-model mobilenet_v4_m_il-common
```

For detailed usage of model management tools, see the [Tools and Utilities](tools/index.md) section.
