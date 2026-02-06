# Loading Models

This page explains how to approach model loading in Birder at a workflow level.

## Choose a Loading Path

Use this rule of thumb:

1. `birder.load_pretrained_model_and_transform`  
   Use for most inference workflows. You get a model, metadata, and the matching transform in one step.
2. `birder.load_pretrained_model`  
   Use when you want explicit preprocessing control (custom transform behavior) but still load from the registry.
3. `birder.load_model_with_cfg`  
   Use when you already have a model config (dict or JSON) and optional local weights.

## Typical Workflow: Fast Inference Setup

```python
import birder

# Works for both classification and detection weights
(net, model_info, transform) = birder.load_pretrained_model_and_transform(
    "vit_l14_pn_bioclip-v2",
    inference=True,
)
```

Why this is usually the best default:

- It keeps preprocessing aligned with the model signature and RGB stats.
- It avoids manual setup mistakes.
- It is the shortest path from model name to runnable inference.

## Typical Workflow: Manual Preprocessing Control

```python
import birder

(net, model_info) = birder.load_pretrained_model("vit_l14_pn_bioclip-v2", inference=True)
```

Use this path when you need to tune preprocessing behavior for experiments.
For transform construction patterns, see [Transforms and Signatures](transforms_and_signatures.md).

## Typical Workflow: Load From Config + Optional Local Weights

```python
import birder

# cfg can be a dict or a path to a JSON config
net, cfg = birder.load_model_with_cfg("path/to/model_config.json", "path/to/model_weights.pt")
```

Use this when your source of truth is a local artifact, not a registry entry.

## Practical Notes

- Pre-trained loading is registry-driven: the entry name determines architecture, task, and weight metadata.
- `model_info` is the bridge between loading and inference setup (signature, class mapping, RGB stats).
- For detection checkpoints, registry metadata includes backbone information; you do not need to build it manually in normal usage.

## Common Mistakes

- Treating pre-trained entry names and architecture names as interchangeable. They are related, but not the same layer.
