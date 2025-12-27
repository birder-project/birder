# Changelog

## Unreleased

### Added

- **Mosaic Schedule**: Added `--mosaic-stop-epoch` for linear mosaic decay and disabling mosaic after the specified epoch.
- **Detection Batch Multiscale**: Added `--batch-multiscale` for per-batch square resizing in detection training and show-det-iterator.
- **RT-DETR v1 Detection**: Added [RT-DETR v1](https://arxiv.org/abs/2304.08069) object detection model.
- **ViT Causal Attention**: Added optional causal attention support to ViT encoder blocks via `set_causal_attention()` method across all ViT variants (ViT, Simple ViT, DeiT, DeiT3, FlexiViT, ViT Parallel, ViT SAM and RoPE variants).
- **CSPNet DropBlock Regularization**: Added optional DropBlock2d support to CSPNet backbone blocks (`DarkBlock`, `BottleneckBlock`) for YOLO v4-style training regularization.
- **MAE ViT Learnable Positional Embedding**: Added optional `learnable_pos_embed` support for MAE ViT decoder, enabling learnable positional embeddings as an alternative to fixed sin-cos embeddings.

### Changed

- **Training Scheduler CLI**: Renamed `--warmup-iters/--cooldown-iters` to `--warmup-steps/--cooldown-steps` and replaced `--lr-scheduler-update iter` with `--lr-scheduler-update step` to align with optimizer-step terminology.

### Fixed

- **YOLO v2 / v3 / v4 Loss Normalization**: Fixed loss normalization to use consistent scaling across all loss terms.

## 0.1.8 - 2025-12-20

### Added

- **MobileClip v2**: Added [MobileClip v2](https://arxiv.org/abs/2508.20691) image encoders (i3, i4) featuring a 5-stage architecture.
- **MIM Training Options**: Added `--mask-ratio` and `--min-mask-size` CLI arguments to `train_mim.py` for configurable masking during Masked Image Modeling training.
- **MAE ViT Normalized Pixel Loss**: Added optional `norm_pix_loss` support for MAE ViT.

### Changed

- **MobileClip v1 Naming**: Renamed MobileClip models to use explicit v1 naming (`mobileclip_i0` -> `mobileclip_v1_i0`, etc.) for clarity and future extensibility.

### Fixed

- **YOLO v4 Loss Coefficients**: Adjusted YOLO v4 and YOLO v4 Tiny loss coefficients to better balance box regression and background losses.

## 0.1.7 - 2025-12-15

### Added

- **VoVNet v1**: Added [VoVNet v1 (One-Shot Aggregation)](https://arxiv.org/abs/1904.09730) classification models.
- **VoVNet v2**: Added [VoVNet v2 (ESE-VoVNet)](https://arxiv.org/abs/1911.06667) classification models.
- **YOLO v2 Detection**: Added [YOLO v2](https://arxiv.org/abs/1612.08242) object detection models.
- **YOLO v3 Detection**: Added [YOLO v3](https://arxiv.org/abs/1804.02767) object detection models.
- **YOLO v4 / YOLO v4 Tiny Detection**: Added [YOLO v4](https://arxiv.org/abs/2004.10934) and [YOLO v4 Tiny](https://arxiv.org/abs/2011.08036) object detection models.
- **Mosaic Augmentation**: Added mosaic data augmentation for object detection with `fixed_grid` and `random_center` modes.
- **Default Channels Configuration**: Added a centralized `DEFAULT_NUM_CHANNELS` setting for consistent default input channel configuration.

### Fixed

- **FocalNet Input Channels**: Fixed hardcoded input channels in the FocalNet stem to correctly use the `input_channels` parameter.
- **NextViT Input Channels**: Fixed hardcoded input channels in the NextViT stem to correctly use the `input_channels` parameter.
- **MIM Input Channels**: Fixed hardcoded channel values in MIM models to use the `input_channels` parameter, enabling support for non-RGB inputs.

## 0.1.6 - 2025-12-10

### Fixed

- **DINOv2 / Franca Weight Decay Scheduling**: Updated weight decay scheduling from per-epoch to per-step updates to match the original implementations.
- **Optimizer Compilation Detection**: Fixed the `compile_opt` condition in `get_optimizer` to correctly detect when optimizer compilation is enabled.

## 0.1.5 - 2025-12-08

### Added

- **Franca SSL**: Added [Franca](https://arxiv.org/abs/2507.14137), a self-supervised learning method based on Nested Matryoshka Clustering.

### Fixed

- **MMCR TrainTransform**: Fixed view generation to correctly produce `n_aug` augmented views instead of using integer division.
- **Distributed Training Initialization**: Fixed `init_distributed_mode` to correctly pass the `device_id` parameter and simplified barrier synchronization.

## 0.1.4 - 2025-12-04

### Fixed

- **DINOv2 Freeze Last Layer (Distillation)**: Implemented the missing `--freeze-last-layer-epochs` functionality for DINOv2 distillation training.

## 0.1.3 - 2025-12-03

### Fixed

- **DINOv2 iBOT Center Update**: Fixed incorrect tensor slicing in the iBOT patch loss center update when using the "centering" strategy.
- **Freeze Last Layer Gradient Handling**: Fixed `--freeze-last-layer-epochs` to cancel gradients after the backward pass (previously applied before backward and ineffective). Affects DINO v1, DINO v2, and iBOT training.
- **DINOv2 Freeze Last Layer**: Implemented the missing `--freeze-last-layer-epochs` functionality for standard DINOv2 training.
- **CAPI Loss Precision**: Updated CAPI loss computation to use double precision, matching the original implementation.

## 0.1.2 - 2025-10-14

### Added

- **ConvNeXt v1 SSL**: Added self-supervised pre-training support for ConvNeXt v1 models.

## 0.1.1 - 2025-10-08

### Added

- **Pretrained Models**:
    - `rdnet_t_ibot-bioscan5m`: RDNet-T encoder pre-trained with iBOT on the [BIOSCAN-5M](https://biodiversitygenomics.net/projects/5m-insects/) dataset.
    - `convnext_v1_tiny_eu-common`
    - `convnext_v2_tiny_eu-common`
    - `mvit_v2_s_yellowstone`

### Fixed

- **DINOv1 Backbone Loading**: Fixed an issue where the `--backbone-epoch` flag was ignored during DINOv1 model initialization.

## 0.1.0 - 2025-10-02

### Added

- Initial beta release
