# Inference Workflows

This page covers practical inference patterns using the Python API.

## Core Idea

Most inference pipelines have the same shape:

1. Load model (and usually transform)
2. Run inference on image(s)
3. Consume outputs in task-specific form

Birder supports this for both classification and detection.

## Single-Image Classification

```python
import birder
from birder.inference.classification import infer_image

net, model_info, transform = birder.load_pretrained_model_and_transform(
    "vit_l14_pn_bioclip-v2",
    inference=True,
)

probs, _ = infer_image(net, "data/img_001.jpeg", transform)
```

When you need embeddings instead of just probabilities:

```python
probs, embedding = infer_image(net, "data/img_001.jpeg", transform, return_embedding=True)
```

## Single-Image Detection

```python
import birder
from birder.inference.detection import infer_image

net, model_info, transform = birder.load_pretrained_model_and_transform(
    "deformable_detr_boxref_coco_convnext_v2_tiny_imagenet21k",
    inference=True,
)

detections = infer_image(net, "data/img_001.jpeg", transform, score_threshold=0.25)
# detections: {'boxes', 'labels', 'scores'}
```

## Batch Inference With DataLoaders

Use dataloader inference when you need throughput, dataset-scale export, or periodic callbacks.

Classification:

```python
from birder.inference.classification import infer_dataloader

sample_paths, outs, labels, embeddings = infer_dataloader(device, net, dataloader, chunk_size=None)
```

Detection:

```python
from birder.inference.detection import infer_dataloader

sample_paths, detections, targets = infer_dataloader(device, net, dataloader)
```

Detection dataloader results are postprocessed back to original image coordinates.

For large images, use sliding-window detection:

```python
sample_paths, detections, targets = infer_dataloader(
    device,
    net,
    dataloader,
    sliding_window_tile_size=(640, 640),
    sliding_window_overlap=(128, 128),
    sliding_window_global_size=(800, 800),
    sliding_window_merge_mode="greedy_nmm",
    sliding_window_merge_threshold=0.5,
    sliding_window_filter_threshold=0.25,
    sliding_window_tile_batch_size=16,
)
```

## Choosing Output Type (Classification)

- Probabilities: default, best for predictions and reports
- Logits: use `return_logits=True` for calibration/research workflows
- Embeddings: use `return_embedding=True` for retrieval, clustering, or downstream models

## TTA Guidance

- Classification TTA (`tta=True`) can improve stability for borderline cases at extra compute cost.
- Detection TTA (`tta=True`) runs multi-scale + horizontal flip and fuses predictions, use when quality matters more than latency.
