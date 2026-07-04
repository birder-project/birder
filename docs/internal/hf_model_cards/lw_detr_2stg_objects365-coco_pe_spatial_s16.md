---
tags:
- object-detection
- birder
- pytorch
library_name: birder
license: apache-2.0
base_model:
- birder-project/lw_detr_2stg_objects365_pe_spatial_s16
---

# Model Card for lw_detr_2stg_objects365-coco_pe_spatial_s16

A two-stage LW-DETR object detector with a PE-Spatial s16 backbone, initialized from the Objects365-2020 checkpoint and fine-tuned on COCO 2017.
Training used multi-resolution inputs sampled from 512px to 672px, followed by a final fixed-resolution stage at 640px.

**Custom Kernels**: This model uses optimized custom kernel for Deformable Attention operations. If you encounter compilation issues or prefer to use pure PyTorch implementations, set the environment variable `DISABLE_CUSTOM_KERNELS=1` before loading the model.

## Model Details

- **Model Type:** Object detection
- **Model Stats:**
    - Params (M): 32.2
    - Input image size: 640 x 640
- **Dataset:** COCO 2017 (80 classes)

- **Papers:**
    - LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection: <https://arxiv.org/abs/2406.03459>
    - Perception Encoder: The best visual embeddings are not at the output of the network: <https://arxiv.org/abs/2504.13181>

- **Metrics:**
    - mAP @ 576x576px: 53.25 (mAP@0.50 72.52, mAP@0.75 57.55)
    - mAP @ 608x608px: 53.96 (mAP@0.50 73.05, mAP@0.75 58.33)
    - mAP @ 640x640px: 54.58 (mAP@0.50 73.56, mAP@0.75 59.06)
    - mAP @ 672x672px: 54.74 (mAP@0.50 73.92, mAP@0.75 59.52)

## Model Usage

### Object Detection

```python
import birder
from birder.inference.detection import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("lw_detr_2stg_objects365-coco_pe_spatial_s16", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.detection_transform(size, model_info.rgb_stats, dynamic_size=model_info.signature["dynamic"])

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("lw_detr_2stg_objects365-coco_pe_spatial_s16", inference=True)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
detections = infer_image(net, image, transform)
# detections is a dict with keys: 'boxes', 'labels', 'scores'
# boxes: torch.Tensor with shape (N, 4) in [x1, y1, x2, y2] format
# labels: torch.Tensor with shape (N,) containing class indices
# scores: torch.Tensor with shape (N,) containing confidence scores
```

## Citation

```bibtex
@misc{chen2024lwdetrtransformerreplacementyolo,
      title={LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection},
      author={Qiang Chen and Xiangbo Su and Xinyu Zhang and Jian Wang and Jiahui Chen and Yunpeng Shen and Chuchu Han and Ziliang Chen and Weixiang Xu and Fanrong Li and Shan Zhang and Kun Yao and Errui Ding and Gang Zhang and Jingdong Wang},
      year={2024},
      eprint={2406.03459},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.03459},
}

@misc{bolya2025perceptionencoderbestvisual,
      title={Perception Encoder: The best visual embeddings are not at the output of the network},
      author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Dollár and Christoph Feichtenhofer},
      year={2025},
      eprint={2504.13181},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.13181},
}
```
