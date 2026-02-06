# Transforms and Signatures

This page explains how Birder model metadata drives preprocessing and output interpretation.

## Why This Matters

Most inference issues come from mismatched preprocessing, not from the model itself.
In practice, you should treat model metadata as the source of truth for transform setup.

## What `model_info` Gives You

When you load a pre-trained model, Birder returns `model_info` with:

- `signature`: input/output shape contract for the checkpoint
- `rgb_stats`: normalization stats expected by the model
- `class_to_idx`: class mapping (for classification and some detection workflows)

These values are what you should use to build transforms.

If you want the safest default path, use `load_pretrained_model_and_transform` as described in
[Model Loading](model_loading.md). This page focuses on manual transform control.

## Manual Pattern (When You Need Control)

Use this when you intentionally want custom preprocessing behavior.

### Classification

```python
import birder

(net, model_info) = birder.load_pretrained_model("hiera_abswin_base_mim-intermediate-eu-common", inference=True)
size = birder.get_size_from_signature(model_info.signature)
transform = birder.classification_transform(
    size,
    model_info.rgb_stats,
    center_crop=1.0,
    simple_crop=False,
)
```

### Detection

```python
import birder

(net, model_info) = birder.load_pretrained_model(
    "deformable_detr_boxref_coco_convnext_v2_tiny_imagenet21k",
    inference=True,
)
size = birder.get_size_from_signature(model_info.signature)
transform = birder.detection_transform(
    size,
    model_info.rgb_stats,
    dynamic_size=model_info.signature["dynamic"],
)
```

## Dynamic Size and Detection

For many detection checkpoints, inference uses dynamic resizing behavior.
In those cases, use the `dynamic` value from `model_info.signature` instead of hardcoding resize rules.

If you use `load_pretrained_model_and_transform`, this is handled automatically.

## Coordinate Systems

Detection pipelines often involve internal resizing before inference.
Birder's detection inference helpers postprocess boxes back to original image coordinates in standard workflows.
That means downstream consumers usually do not need to manually rescale boxes.
