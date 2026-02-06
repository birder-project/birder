# Model Registry

The model registry is Birder's central catalog for model discovery.
It lets you work with:

- Architecture families (for building/training models)
- Pre-trained entries (weights + metadata for loading/evaluation)

## Mental Model

Think in two layers:

1. **Network architecture name**  
   Example: `vit_l14_pn`, `hieradet_d_small`, `vit_reg4_b16`
2. **Pre-trained entry name**  
   Example: `vit_l14_pn_bioclip-v2`, `hieradet_d_small_dino-v2`

A pre-trained entry points back to an architecture and adds practical metadata such as resolution and available weight formats.

## Task-Aware Discovery

The registry is task-aware, so you can explore models by workflow:

- Image classification
- Object detection
- Masked image modeling (MIM)
- Self-supervised learning (SSL)

```python
from birder.model_registry import Task, registry

# Architectures by task
cls_arch = registry.list_models(task=Task.IMAGE_CLASSIFICATION)
det_arch = registry.list_models(task=Task.OBJECT_DETECTION)
mim_arch = registry.list_models(task=Task.MASKED_IMAGE_MODELING)
ssl_arch = registry.list_models(task=Task.SELF_SUPERVISED_LEARNING)

# Pre-trained entries by task
det_weights = registry.list_pretrained_models(task=Task.OBJECT_DETECTION)
print(cls_arch[:5], det_arch[:5], det_weights[:5])
```

You can also filter by naming patterns:

```python
from birder.model_registry import registry

print(registry.list_pretrained_models("*bioclip*"))
print(registry.list_models(include_filter="*convnext*"))
```

## Choosing What to Load

For inference workflows, use the top-level loader and let Birder handle model setup:

```python
import birder

(net, model_info) = birder.load_pretrained_model("vit_l14_pn_bioclip-v2", inference=True)
```

Use registry metadata to compare candidate checkpoints before loading:

```python
from birder.model_registry import registry

meta = registry.get_pretrained_metadata("vit_l14_pn_bioclip-v2")
print(meta["description"])
print(meta["task"])
print(meta["resolution"])
print(list(meta["formats"]))
```

## Registry-First Usage (Experimentation)

Use the registry directly when you want to assemble custom experiments, especially during training or architecture research.

```python
from birder.model_registry import registry

net = registry.net_factory("vit_l14_pn", num_classes=500)
```

## Extending the Registry

### Add a Local Architecture Alias

```python
from birder.model_registry import registry
from birder.net.rope_vit import RoPE_ViT

registry.register_model_config(
    "rope_vit_reg4_b14_custom",
    RoPE_ViT,
    config={
        "patch_size": 14,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "drop_path_rate": 0.15,
        "num_reg_tokens": 4,
        "attn_pool_head": True,
        "rope_temperature": 10000.0,
    },
)
```

### Add Your Own Weight Entry

```python
import birder
from birder.model_registry import registry

registry.register_weights(
    "rope_vit_reg4_b14_custom_plankton",
    {
        "url": "https://huggingface.co/your-org/rope_vit_reg4_b14_custom_plankton/resolve/main",
        "description": "Custom RoPE ViT model trained on a plankton dataset",
        "resolution": (384, 384),
        "formats": {"pt": {"file_size": 358.4, "sha256": "<sha256>"}},
        "net": {"network": "rope_vit_reg4_b14_custom", "tag": "plankton"},
    },
)

(net, model_info) = birder.load_pretrained_model("rope_vit_reg4_b14_custom_plankton", inference=True)
```

Use the same entry name and file stem (for example, `rope_vit_reg4_b14_custom_plankton.pt`) so default loading resolves cleanly.
For detection entries, add a `backbone` field in metadata.
