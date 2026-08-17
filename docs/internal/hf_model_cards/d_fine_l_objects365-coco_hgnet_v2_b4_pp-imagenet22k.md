---
tags:
- object-detection
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for d_fine_l_objects365-coco_hgnet_v2_b4_pp-imagenet22k

A D-FINE L object detector with an HGNet v2 B4 backbone pretrained on ImageNet-22K, trained on Objects365-2020, then fine-tuned on COCO 2017.
Training used multi-resolution inputs sampled from 480px to 800px.

An inference-optimized, structurally reparameterized checkpoint is also provided as
`d_fine_l_objects365-coco_hgnet_v2_b4_pp-imagenet22k_reparameterized`.

**Important:** The reparameterized checkpoint fuses compatible convolution and normalization branches and removes training-only decoder components. It is intended for inference and deployment.
For continued training or fine-tuning, use the standard checkpoint, `d_fine_l_objects365-coco_hgnet_v2_b4_pp-imagenet22k`, and reparameterize the model only after training is complete.

**Custom Kernels**: This model uses optimized custom kernel for Deformable Attention operations. If you encounter compilation issues or prefer to use pure PyTorch implementations, set the environment variable `DISABLE_CUSTOM_KERNELS=1` before loading the model.

## Model Details

- **Model Type:** Object detection
- **Model Stats:**
    - Params (M): 31.3
    - Input image size: 640 x 640
- **Dataset:** COCO 2017 (80 classes)

- **Papers:**
    - D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement: <https://arxiv.org/abs/2410.13842>

- **Metrics:**

    | Input size |   mAP | mAP (reparam.) | mAP@50 | mAP@50 (reparam.) | mAP@75 | mAP@75 (reparam.) |
    | ---------- | ----: | -------------: | -----: | ----------------: | -----: | ----------------: |
    | 512 x 512  | 54.00 |          53.99 |  71.85 |             71.84 |  58.64 |             58.64 |
    | 576 x 576  | 55.47 |          55.49 |  73.07 |             73.08 |  60.31 |             60.32 |
    | 608 x 608  | 55.77 |          55.77 |  73.34 |             73.37 |  60.81 |             60.79 |
    | 640 x 640  | 56.09 |          56.08 |  73.51 |             73.51 |  60.98 |             60.97 |
    | 672 x 672  | 56.21 |          56.21 |  73.56 |             73.56 |  61.21 |             61.21 |
    | 704 x 704  | 56.43 |          56.43 |  73.87 |             73.87 |  61.54 |             61.53 |

## Model Usage

### Object Detection

```python
import birder
from birder.inference.detection import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("d_fine_l_objects365-coco_hgnet_v2_b4_pp-imagenet22k", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.detection_transform(size, model_info.rgb_stats, dynamic_size=model_info.signature["dynamic"])

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("d_fine_l_objects365-coco_hgnet_v2_b4_pp-imagenet22k", inference=True)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
detections = infer_image(net, image, transform)
# detections is a dict with keys: 'boxes', 'labels', 'scores'
# boxes: torch.Tensor with shape (N, 4) in [x1, y1, x2, y2] format
# labels: torch.Tensor with shape (N,) containing class indices
# scores: torch.Tensor with shape (N,) containing confidence scores
```

### Reparameterized Inference

Use the `_reparameterized` checkpoint for inference-oriented deployment.
Birder automatically constructs the matching reparameterized architecture, no additional conversion flag is needed.

```python
import birder
from birder.inference.detection import infer_image

weights = "d_fine_l_objects365-coco_hgnet_v2_b4_pp-imagenet22k_reparameterized"
net, model_info, transform = birder.load_pretrained_model_and_transform(weights, inference=True)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
detections = infer_image(net, image, transform)
```

## Citation

```bibtex
@misc{peng2024dfineredefineregressiontask,
      title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
      author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
      year={2024},
      eprint={2410.13842},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.13842},
}
```
