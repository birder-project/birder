# Changelog

## 0.4.3 - 2026-02-06

### Added

- **Eval Module**: New `birder.eval` module for model evaluation with subcommand-based CLI (`python -m birder.eval <command>`).
    - `classification`: Evaluate pretrained classification models (migrated from `evaluate.py` with WebDataset support and configurable `--num-workers`).
    - `adversarial`: Evaluate adversarial robustness with FGSM, PGD, DeepFool and SimBA attacks.
    - `awa2`: Run [AwA2](https://arxiv.org/abs/1707.00600) benchmark using MLP probe for 85-attribute multi-label classification (macro F1 metric).
    - `fishnet`: Run [FishNet](https://fishnet-2023.github.io/) benchmark using MLP probe for 9-trait multi-label classification (macro F1 metric).
    - `flowers102`: Run [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) benchmark using SimpleShot with full training set.
    - `bioscan5m`: Run [BIOSCAN-5M](https://arxiv.org/abs/2406.12723) benchmark - unsupervised embedding evaluation using UMAP + Agglomerative Clustering + Adjusted Mutual Information.
    - `fungiclef`: Run [FungiCLEF2023](https://www.imageclef.org/FungiCLEF2023) benchmark - 1,604 species classification using KNN.
    - `nabirds`: Run [NABirds](https://dl.allaboutbirds.org/nabirds) benchmark using KNN with configurable k neighbors.
    - `newt`: Run [NeWT](https://arxiv.org/abs/2103.16483) benchmark (164 binary tasks with SVM, micro-averaged accuracy, per-cluster breakdown).
    - `plankton`: Run [SYKE-plankton_IFCB_2022](https://b2share.eudat.eu/records/xvnrp-7ga56) benchmark - 50 class phytoplankton classification using linear probing.
    - `plantdoc`: Run [PlantDoc](https://arxiv.org/abs/1911.10317) benchmark using SimpleShot few-shot learning with configurable k-shot sampling.
    - `plantnet`: Run [PlantNet-300K](https://openreview.net/forum?id=eLYinD0TtIt) benchmark using SimpleShot few-shot learning with configurable k-shot sampling.
- **Eval Methods**: Added evaluation methods to `birder.eval.methods`:
    - `simpleshot`: [SimpleShot](https://arxiv.org/abs/1911.04623) few-shot classifier using nearest centroid with mean centering and L2 normalization.
    - `knn`: K-nearest neighbors with temperature-scaled softmax voting.
    - `mlp`: MLP probe for multi-label classification following AwA2 design (Linear -> Dropout -> Linear).
    - `ami`: AMI clustering using UMAP + Agglomerative Clustering + Adjusted Mutual Information.
    - `linear`: Linear probing on frozen embeddings - trains a single linear layer with cross-entropy loss.
    - `svm`: SVM classifier with StandardScaler preprocessing and RandomizedSearchCV hyperparameter tuning.
- **AwA2 Datahub**: Added [AwA2](https://arxiv.org/abs/1707.00600) (Animals with Attributes 2) dataset to `birder.datahub.evaluation` for attribute prediction benchmark.
- **FishNet Datahub**: Added [FishNet](https://fishnet-2023.github.io/) dataset to `birder.datahub.evaluation` for trait prediction benchmark.
- **FungiCLEF2023 Datahub**: Added [FungiCLEF2023](https://www.imageclef.org/FungiCLEF2023) dataset to `birder.datahub.evaluation` for fungi species classification benchmark.
- **NABirds Datahub**: Added [NABirds](https://dl.allaboutbirds.org/nabirds) dataset to `birder.datahub.evaluation` for bird species classification benchmark.
- **NeWT Datahub**: Added [NeWT](https://arxiv.org/abs/2103.16483) (Natural World Tasks) dataset to `birder.datahub.evaluation` for benchmark evaluation.
- **Plankton Datahub**: Added [SYKE-plankton_IFCB_2022](https://b2share.eudat.eu/records/xvnrp-7ga56) dataset to `birder.datahub.evaluation` for phytoplankton classification benchmark.
- **PlantDoc Datahub**: Added [PlantDoc](https://arxiv.org/abs/1911.10317) dataset to `birder.datahub.evaluation` for plant disease classification benchmark.
- **PlantNet Datahub**: Added [PlantNet-300K](https://openreview.net/forum?id=eLYinD0TtIt) dataset to `birder.datahub.evaluation` for plant species classification benchmark.
- **Convenience Loader**: Added `load_pretrained_model_and_transform` to return a pretrained model and its default inference transform.
- **Detection PT2 Export**: All detection models now support `torch.export`, except Faster R-CNN, which is currently unsupported due to data-dependent control flow in the RPN.
- **Model Conversion**: Added `--opset` and `--simplify` arguments to `convert-model` for better control over ONNX export.
- **Image Loader Selection**: Added `--img-loader` to `predict` CLI to select the image decoding backend (`tv` or `pil`), matching the existing training option.
- **Channels-Last Inference**: Added `--channels-last` to `predict`, `predict_detection`, `birder.eval classification` and `benchmark` CLIs to opt into channels-last memory format.
- **Channels-Last Training**: Added `--channels-last` to `train`, `train_detection` and `train_kd` CLIs to opt into channels-last memory format during training.

### Changed

- **Detection ImageList**: Changed `ImageList.image_sizes` from `list[tuple[int, int]]` to `torch.Tensor` with shape `(B, 2)`.
- **Sinkhorn Queues (SSL)**: Queue tensors, pointers and full-state are now registered buffers so they save/load with the model state dict.

### Fixed

- **MAE Hiera Unpatchify**: Fixed incorrect pixel order in `unpatchify` to match the `[C, H, W]` order from `get_pixel_label_2d`.
- **CSWin Transformer PT2 Export**: Fixed dynamic batch shape specialization caused by Python `int()` casting during window reconstruction. All classification models are now exportable.

## 0.4.2 - 2026-01-30

### Added

- **LW-DETR Detection**: Added [LW-DETR](https://arxiv.org/abs/2406.03459) object detection model.
- **Backbone Layer Decay**: Added `--backbone-layer-decay` CLI argument to apply layer-wise learning rate decay (LLRD) specifically to backbone parameters during detection training.
- **Pretrained Models**:
    - `vit_b16_pn_bioclip-v1`: Added [BioCLIP](https://arxiv.org/abs/2311.18803) ViT-B/16 pretrained weights.

### Changed

- **BioCLIP v2 Weights (Breaking)**: Updated `vit_l14_pn_bioclip-v2` model weights to include visual projection layer (head).
- **Compatibility**: Tested with PyTorch 2.10.

## 0.4.1 - 2026-01-23

### Added

- **Plain DETR Detection**: Added [Plain DETR](https://arxiv.org/abs/2308.01904) object detection model with Box-to-Pixel Relative Position Bias (BoxRPB).
- **RT-DETR v2 Detection**: Added [RT-DETR v2](https://arxiv.org/abs/2407.17140) object detection model.
- **Detection Multiscale Step**: Added `--multiscale-step` to control multiscale size increments and collator `size_divisible` padding for detection training.
- **ConvNeXt v1 Isotropic**: Added isotropic ConvNeXt v1 variants.
- **Training Top-K Accuracy**: Added optional top-k accuracy tracking during classification training.

### Changed

- **ViT-family Detection Outputs**: ViT, Simple ViT, DeiT, DeiT3, FlexiViT, ViT_Parallel and RoPE variants now support `out_indices` for multi-stage detection features.
- **DINOv2 Loss Vectorization**: Vectorized `DINOLoss.forward` using einsum, improving training throughput.
- **Franca Loss Vectorization**: Vectorized `DINOLossMRL.forward` using einsum, improving training throughput.
- **Benchmark CLI**: `benchmark.py` can now benchmark detection networks.

### Fixed

- **Training Metrics Per-Epoch Reset**: Updated all training scripts to reset metrics at the beginning of each epoch, ensuring per-epoch statistics are independent.

## 0.4.0 - 2026-01-18

### Changed

- **ViT Attention Consolidation (Breaking)**:
    - Dropped the `nn.MultiheadAttention` path, ViT now always uses the custom attention implementation.
    - Renamed ViT block attributes `self_attention` → `attn` and `ln1/ln2` → `norm1/norm2` (affects hooks, tooling and checkpoint key names).
- **Checkpoint States Filenames (Breaking)**: Training state checkpoints now use a `.pt` extension (e.g., `_states.pt`).
- **Network Architecture Consolidation (Breaking)**:
    - Merged `MobileNet_v3_Large` and `MobileNet_v3_Small` into a unified `MobileNet_v3` class. Use model configs (`mobilenet_v3_large_*` and `mobilenet_v3_small_*`) instead of separate classes.
    - Merged `SE_ResNet_v1`, `SE_ResNet_v2` and `SE_ResNeXt` into their respective base classes (`ResNet_v1`, `ResNet_v2`, `ResNeXt`). SE variants are now configured via the `squeeze_excitation` config parameter rather than separate classes.
- **Pretrained Models**:
    - **Breaking**: Dropped `tiny_vit_5m_il-common`
    - **Breaking**: Dropped `efficientformer_v2_s0_il-common` and `efficientformer_v2_s1_il-common`
    - **Breaking**: Dropped `mobilenet_v2_1_0_il-common`
    - **Breaking**: Dropped `mobilevit_v1_xxs_il-common` and `mobilevit_v1_xs_il-common`
    - **Breaking**: Dropped `mobilevit_v2_1_0_il-common` and `mobilevit_v2_1_5_il-common`
    - **Breaking**: Dropped `repghost_1_0_il-common`
    - **Breaking**: Dropped `squeezenext_1_0_il-common`

### Fixed

- **CaiT LayerNorm eps**: Fixed an incorrect LayerNorm `eps` in the CaiT init.
- **MaxViT Downsampling Projection**: Fixed an incorrect AvgPool kernel size.
- **HGNet v1/v2**: Fixed an incorrect padding.
- **TinyViT MBConv Activation**: Fixed incorrect activation order at MBConv.
- **DenseNet Final Normalization**: Added the missing terminal BatchNorm + ReLU to the final DenseNet stage.
- **DPN Activation**: Fixed incorrect activation function in DPN norm_act (ELU -> ReLU).
- **EfficientFormer v2 ConvMLP Activation**: Fixed incorrect activation function in ConvMLP (ReLU -> GELU).
- **MobileNet v2**: Restored upstream parity by removing the redundant 1×1 expand conv in `t=1` inverted residual blocks.
- **MobileViT v1**: Removed stem and final convolutional layers biases.
- **MobileViT v2**: Removed stem bias and removed the extra SiLU activation after the `conv_proj` to match upstream.
- **NFNet Final Activation**: Added missing terminal `GammaAct`.
- **RepGhostNet Squeeze-Excite Activation**: Fixed incorrect SE activation configuration to match original.
- **DETR Matchers**: Penalize non-finite costs in Hungarian matching to avoid invalid assignments.

## 0.3.3 - 2026-01-16

### Added

- **Pretrained Models**:
    - `hieradet_small_sam2_1`: Added Meta SAM 2.1 HieraDet small pretrained weights.

### Changed

- **HieraDet Dynamic Window Naming (Breaking)**: Renamed HieraDet models using dynamic window sizes from `hieradet_*` to `hieradet_d_*`.
    - Affected models:
        - `hieradet_d_small_dino-v2` (previously `hieradet_small_dino-v2`)
        - `hieradet_d_small_dino-v2-imagenet12k` (previously `hieradet_small_dino-v2-imagenet12k`)
        - `hieradet_d_small_dino-v2-inat21` (previously `hieradet_small_dino-v2-inat21`)

### Fixed

- **Hieradet Forward Padding**: Fixed an incorrect padding in the HieraDet forward pass.

## 0.3.2 - 2026-01-15

### Added

- **ViT QK Normalization**: Added optional QK normalization support (`qk_norm` config option) to all ViT variants (ViT, FlexiViT, RoPE ViT, RoPE FlexiViT).
- **Extended DINOv2 Metrics**: Added target entropy and center drift tracking to training and distillation (`--no-extended-metrics` to disable).
- **Feature PCA Visualization**: Added Feature PCA method for visualizing feature maps using Principal Component Analysis.

### Fixed

- **Avg Model Tool**: Avoid averaging non-floating `state_dict` entries (e.g. `num_batches_tracked`, `relative_position_index`).
- **Model Registry**: Fixed model config registration to consistently register on the active registry instance.

## 0.3.1 - 2026-01-14

### Added

- **Detection Confusion Matrix**: Added detection confusion matrix (background class, score/IoU thresholds) with CLI flags in `det-results`.
- **Fused Optimizer Flag**: Added `--opt-fused` CLI flag to request fused optimizer implementations when supported.
- **Sinkhorn Queueing (SSL)**: Added optional Sinkhorn queueing for DINOv2, DINOv2 distillation, Franca and CAPI training.
- **Pretrained Models**:
    - `gc_vit_xxt_il-common`

### Fixed

- **CAPI Global Sinkhorn**: Fixed the non-position-wise Sinkhorn-Knopp path to keep assignments aligned with logits.

## 0.3.0 - 2026-01-11

### Added

- **Infinite Samplers**: Added `InfiniteSampler`, `InfiniteDistributedSampler` and `InfiniteRASampler` for continuous dataset iteration.
- **Virtual Epochs**: Added `--steps-per-epoch` virtual epoch support across training scripts.
- **Agreement Metrics**: Added prototype/patch agreement tracking for applicable SSL training (DINO v1/v2 and iBOT).
- **Custom Kernel Toggle API**: Added `set_custom_kernels_enabled` / `is_custom_kernels_enabled` to control custom kernel loading in code.
- **Pretrained Models**:
    - `lit_v1_t_il-common`

### Changed

- **Detection Results CLI**: Expanded `det-results` to support multi-file comparison tables, per-class filtering, short summaries and summary CSV output.
- **Quantize Model Tool**: Updated `quantize-model` to use PT2E/TorchAO quantization instead of deprecated FX/TorchScript.
- **SWAttention/MSDA Custom Ops**: Routed custom kernels through `torch.library.custom_op` to allow full-graph `torch.compile` capture.

### Fixed

- **Adjust Size Device / Grad**: Ensure `adjust_size` updates run without grad tracking and keep buffers/parameters on the active device.
- **Sampler Seeding**: Ensure training samplers consistently honor `--seed` across distributed and non-distributed modes.
- **Soft-NMS GPU Sync**: Removed host-side `.item()` synchronization from the soft-nms kernel implementation.

## 0.2.3 - 2026-01-09

### Added

- **LIT v2**: Added [LIT v2](https://arxiv.org/abs/2205.13213) classification models with HiLo attention.
- **LIT v1**: Added [LIT v1](https://arxiv.org/abs/2105.14217) classification models.
- **GC-ViT**: Added [Global Context Vision Transformer](https://arxiv.org/abs/2206.09959) classification models with dynamic size support (adapted from timm).
- **YOLO Custom Anchors**: Allow YOLO models to consume user-provided anchor specs or auto-anchors JSON.
- **Detection TTA (WBF)**: Added test time augmentations with Weighted Boxes Fusion in detection inference.
- **EfficientDet Dynamic Size**: Support dynamic multiscale inputs for EfficientDet.
- **Embedding Distillation**: Added embedding matching distillation to `train_kd.py` via `--type embedding`.
- **Custom Layer LR Scaling**: Added `--custom-layer-lr-scale` CLI argument to apply custom learning rate scales to specific layers by name (e.g., `--custom-layer-lr-scale offset_conv=0.01,attention=0.5`).
- **Custom Layer Weight Decay**: Added `--custom-layer-wd` CLI argument to apply custom weight decay to specific layers by name (e.g., `--custom-layer-wd offset_conv=0.0`).
- **Pretrained Models**:
    - `vovnet_v2_19_il-common`
    - `vovnet_v2_39_il-common`
    - **Breaking**: Dropped `vit_l16_eu-common`

### Fixed

- **Layer-wise LR Decay**: Fixed critical bug where `lr_scale` from layer decay and custom layer LR scaling was not being applied to actual learning rates.
- **CocoInference Bounding Boxes**: Fixed incorrect bounding box coordinates in detection inference.
- **Pack Tool**: Fixed multiprocessing reliability issues, hangs and signal handling.

## 0.2.2 - 2026-01-01

### Added

- **ResNet-D Variants**: Added ResNet-D support to ResNet v1 based on the ["Bag of Tricks for Image Classification with Convolutional Neural Networks"](https://arxiv.org/abs/1812.01187) paper, featuring deep stem and average pooling downsample.
- **DeepFool Attack**: Added [DeepFool](https://arxiv.org/abs/1511.04599) adversarial attack support.
- **SimBA Attack**: Added [SimBA](https://arxiv.org/abs/1905.07121) (Simple Black-box Attack) adversarial support.
- **YOLO Auto Anchors**: Added a k-means anchor fitting tool for YOLO models.
- **Transformer Attribution**: Added [Transformer Attribution](https://arxiv.org/abs/2012.09838) for transformer interpretability.

### Changed

- **Data2Vec2 Target Normalization**: Align target normalization more closely with the paper (per-layer instance norm before averaging).
- **EMA Step Alignment**: Align EMA updates and warmup to optimizer steps (including grad accumulation).
- **Adversarial Attack Refactor**: Major refactor of adversarial attack implementations and CLI wiring.
- **Introspection Module Refactor**: Major refactor of interpretability methods.
- **Detection Training Loss Logging**: Enhanced detection training to log individual loss components (e.g., `loss_objectness`, `loss_rpn_box_reg`, `labels`, `giou`, etc.).

### Fixed

- **YOLO v3 / v4 Dynamic Anchor Scaling**: Removed automatic anchor scaling during multiscale training/inference as it caused prediction instability.
- **MultiStepLR Absolute Milestones**: Fixed MultiStepLR scheduler to treat `--lr-steps` as absolute epoch/step positions rather than relative to warmup end.
- **DINOv2 Distillation Teacher Mode**: Keep the teacher in eval mode during distillation training.
- **Data2Vec Mask Selection**: Use keep tokens rather than masked tokens when selecting targets for loss computation.

## 0.2.1 - 2025-12-28

### Added

- **YOLO Augmentation**: Added a `yolo` detection augmentation preset for YOLO-style training pipelines.

### Fixed

- **YOLO v3 / v4 Dynamic Anchors**: Scale anchors per input size when running in dynamic-size mode.
- **Detection Batch Multiscale Targets**: Fixed batch multiscale resizing to rescale target boxes alongside images.

## 0.2.0 - 2025-12-27

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
- **Freeze Last Layer Gradient Handling**: Fixed `--freeze-last-layer-epochs` to cancel gradients after the backward pass (previously applied before backward and ineffective). Affects DINOv1, DINOv2 and iBOT training.
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
