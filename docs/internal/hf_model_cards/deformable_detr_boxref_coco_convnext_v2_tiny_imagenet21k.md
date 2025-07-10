---
tags:
- object-detection
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for deformable_detr_boxref_coco_convnext_v2_tiny_imagenet21k

A Deformable DETR with box refinement object detection model with ConvNeXt v2 Tiny backbone (pre-trained on ImageNet-21k) and trained on COCO 2017 dataset.

**Custom Kernels**: This model uses optimized custom kernels for Soft-NMS and Deformable Attention operations. If you encounter compilation issues or prefer to use pure PyTorch implementations, set the environment variable `DISABLE_CUSTOM_KERNELS=1` before loading the model.

## Model Details

- **Model Type:** Object detection
- **Model Stats:**
    - Params (M): 40.0
    - Input image size: 640 x 640 (short side)
- **Dataset:** COCO 2017 (91 classes)

- **Papers:**
    - Deformable DETR: Deformable Transformers for End-to-End Object Detection: <https://arxiv.org/abs/2010.04159>
    - Soft-NMS -- Improving Object Detection With One Line of Code: <https://arxiv.org/abs/1704.04503>
    - ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders: <https://arxiv.org/abs/2301.00808>

- **Metrics:**
    - mAP @ 512x512px: 41.04
    - mAP @ 640px (short side) BS=1 (w/o masking): 45.75
    - mAP @ 640px (short side) BS=2 (w. masking): 45.77
    - mAP @ 800px (short side) BS=1 (w/o masking): 46.68
    - mAP @ 800px (short side) BS=2 (w. masking): 46.63

## Model Usage

### Object Detection

```python
import birder
from birder.inference.detection import infer_image

(net, model_info) = birder.load_pretrained_model("deformable_detr_boxref_coco_convnext_v2_tiny_imagenet21k", inference=True)
# Can also load model with Soft-NMS by passing custom_config={"soft_nms": True}

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.detection_transform(size, model_info.rgb_stats, dynamic_size=model_info.signature["dynamic"])

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
detections = infer_image(net, image, transform)
# detections is a dict with keys: 'boxes', 'labels', 'scores'
# boxes: torch.Tensor with shape (N, 4) in [x1, y1, x2, y2] format
# labels: torch.Tensor with shape (N,) containing class indices
# scores: torch.Tensor with shape (N,) containing confidence scores
```

## Citation

```bibtex
@misc{zhu2021deformabledetrdeformabletransformers,
      title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
      author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
      year={2021},
      eprint={2010.04159},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2010.04159},
}

@misc{bodla2017softnmsimprovingobject,
      title={Soft-NMS -- Improving Object Detection With One Line of Code},
      author={Navaneeth Bodla and Bharat Singh and Rama Chellappa and Larry S. Davis},
      year={2017},
      eprint={1704.04503},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1704.04503},
}

@misc{woo2023convnextv2codesigningscaling,
      title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
      author={Sanghyun Woo and Shoubhik Debnath and Ronghang Hu and Xinlei Chen and Zhuang Liu and In So Kweon and Saining Xie},
      year={2023},
      eprint={2301.00808},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2301.00808},
}
```
