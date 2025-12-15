# Changelog

## 0.1.7 - 2025-12-15

### Added

- **VoVNet v2**: Implemented [VoVNet v2 (ESE-VoVNet)](https://arxiv.org/abs/1911.06667) classification network.
- **VoVNet v1**: Implemented [VoVNet v1](https://arxiv.org/abs/1904.09730) (One-Shot Aggregation) classification network.
- **YOLO v2 Detection**: Implemented [YOLO v2](https://arxiv.org/abs/1612.08242) object detection network.
- **YOLO v3 Detection**: Implemented [YOLO v3](https://arxiv.org/abs/1804.02767) object detection network.
- **YOLO v4 Detection**: Implemented [YOLO v4](https://arxiv.org/abs/2004.10934) object detection network.
- **YOLO v4 Tiny Detection**: Implemented [YOLO v4 Tiny](https://arxiv.org/abs/2011.08036) object detection network.
- **Mosaic Augmentation**: Added mosaic data augmentation for object detection with two modes: `fixed_grid` and `random_center`.
- **Default Channels Setting**: Added centralized `DEFAULT_NUM_CHANNELS` setting for consistent default input channel configuration.

### Fixed

- **FocalNet Input Channels**: Fixed hardcoded input channels in FocalNet stem to use `input_channels` parameter.
- **NextViT Input Channels**: Fixed hardcoded input channels in NextViT stem to use `input_channels` parameter.
- **MIM Input Channels**: Fixed hardcoded channel values in MIM models to properly use `input_channels` parameter, enabling support for non-RGB inputs.

## 0.1.6 - 2025-12-10

### Fixed

- **DINOv2 / Franca Weight Decay Schedule**: Changed weight decay scheduling from per-epoch to per-step updates, matching the original implementation behavior.
- **Optimizer Capturable**: Fixed `compile_opt` check condition in `get_optimizer` to properly detect when optimizer compilation is enabled.

## 0.1.5 - 2025-12-08

### Added

- **Franca SSL**: Implemented [Franca](https://arxiv.org/abs/2507.14137) (Nested Matryoshka Clustering) self-supervised learning method.

### Fixed

- **MMCR TrainTransform**: Fixed view generation to correctly produce `n_aug` augmented views instead of using integer division.
- **Distributed Training Init**: Fixed `init_distributed_mode` to properly pass `device_id` parameter and simplified barrier synchronization.

## 0.1.4 - 2025-12-04

### Fixed

- **DINOv2 Dist. Freeze Last Layer**: Implemented the missing `--freeze-last-layer-epochs` functionality for DINO v2 distillation training.

## 0.1.3 - 2025-12-03

### Fixed

- **DINOv2 iBOT Center Update**: Fixed incorrect tensor slicing in iBOT patch loss center update when using the "centering" strategy.
- **Freeze Last Layer**: Fixed `--freeze-last-layer-epochs` to correctly cancel gradients after backward pass (previously called before backward, which had no effect). Affects DINO v1, DINO v2, and iBOT training scripts.
- **DINOv2 Freeze Last Layer**: Implemented the missing `--freeze-last-layer-epochs` functionality for DINO v2 training.
- **CAPI Loss Precision**: Added double precision casting in CAPI loss computation to match the original implementation.

## 0.1.2 - 2025-10-14

### Added

- ConvNeXt v1 SSL pre-training support

## 0.1.1 - 2025-10-08

### Fixed

- **DINOv1 Backbone Loading**: Fixed an issue where the `--backbone-epoch` flag for loading a pre-existing backbone was incorrectly ignored during DINOv1 model initialization.

### Added

- `rdnet_t_ibot-bioscan5m`: RDNet tiny encoder pre-trained with iBOT on the [BIOSCAN-5M](https://biodiversitygenomics.net/projects/5m-insects/) dataset.
- `convnext_v1_tiny_eu-common`
- `convnext_v2_tiny_eu-common`
- `mvit_v2_s_yellowstone`

## 0.1.0 - 2025-10-02

- First beta release
