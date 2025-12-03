# Changelog

## 0.1.2 - 2025-12-03

### Added

- ConvNeXt v1 SSL pre-training support

### Fixed

- **DINOv2 iBOT Center Update**: Fixed incorrect tensor slicing in iBOT patch loss center update when using the "centering" strategy.
- **Freeze Last Layer**: Fixed `--freeze-last-layer-epochs` to correctly cancel gradients after backward pass (previously called before backward, which had no effect). Affects DINO v1, DINO v2, and iBOT training scripts.
- **DINOv2 Freeze Last Layer**: Implemented the missing `--freeze-last-layer-epochs` functionality for DINO v2 training.
- **CAPI Loss Precision**: Added double precision casting in CAPI loss computation to match the original implementation.

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
