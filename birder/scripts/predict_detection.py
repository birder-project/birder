import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.data.collators.detection import inference_collate_fn
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.coco import CocoInference
from birder.data.datasets.coco import build_label_mapping_indices
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.webdataset import make_wds_detection_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.detection import InferenceTransform
from birder.inference.data_parallel import DetectionInferenceDataParallel
from birder.inference.detection import infer_dataloader
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.results.detection import Results
from birder.results.gui import show_detections

logger = logging.getLogger(__name__)


def _remap_detection_labels(
    detections: list[dict[str, torch.Tensor]], label_remap: Optional[dict[int, int]]
) -> list[dict[str, torch.Tensor]]:
    if label_remap is None:
        return detections

    remapped_detections: list[dict[str, torch.Tensor]] = []
    for detection in detections:
        remapped_detection = dict(detection)
        remapped_detection["labels"] = torch.tensor(
            [label_remap[int(label.item())] for label in detection["labels"]],
            dtype=detection["labels"].dtype,
            device=detection["labels"].device,
        )
        remapped_detections.append(remapped_detection)

    return remapped_detections


def save_output(
    output_path: Path, sample_paths: list[str], class_to_idx: dict[str, int], detections: list[dict[str, torch.Tensor]]
) -> None:
    detection_list = [{k: v.cpu().numpy().tolist() for k, v in detection.items()} for detection in detections]
    output = dict(zip(sample_paths, detection_list))
    output["class_to_idx"] = class_to_idx
    logger.info(f"Saving output at {output_path}")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


def predict(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    elif args.mps is True:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.parallel is True and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} {device} devices")
    else:
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)

        logger.info(f"Using device {device}")

    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    amp_dtype: torch.dtype = getattr(torch, args.amp_dtype)
    network_name = lib.get_detection_network_name(
        args.network, tag=args.tag, backbone=args.backbone, backbone_tag=args.backbone_tag
    )
    net, (class_to_idx, signature, rgb_stats, *_) = fs_ops.load_detection_model(
        device,
        args.network,
        config=args.model_config,
        tag=args.tag,
        reparameterized=args.reparameterized,
        backbone=args.backbone,
        backbone_config=args.backbone_model_config,
        backbone_tag=args.backbone_tag,
        backbone_reparameterized=args.backbone_reparameterized,
        epoch=args.epoch,
        new_size=args.size,
        quantized=args.quantized,
        inference=True,
        pts=args.pts,
        pt2=args.pt2,
        st=args.st,
        dtype=model_dtype,
    )
    model_class_to_idx = class_to_idx
    logger.debug(f"Model loaded with {len(class_to_idx)} classes")
    logger.debug(f"RGB stats: {rgb_stats}")
    logger.debug(f"Model signature dynamic={signature['dynamic']}")

    if args.dynamic_size is True or args.max_size is not None or args.no_resize is True or args.tta is True:
        net.set_dynamic_size()
    if args.dynamic_size is True or args.max_size is not None or args.no_resize is True:
        # Disable cuDNN for dynamic sizes to avoid per-size algorithm selection overhead
        torch.backends.cudnn.enabled = False

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.channels_last is True:
        net = net.to(memory_format=torch.channels_last)
        logger.debug("Using channels-last memory format")

    if args.parallel is True and torch.cuda.device_count() > 1:
        net = DetectionInferenceDataParallel(net, compile_replicas=args.compile)
    else:
        if args.compile is True:
            net = torch.compile(net)
        elif args.compile_backbone is True:
            net.backbone.detection_features = torch.compile(net.backbone.detection_features)

    if args.size is None:
        args.size = lib.get_size_from_signature(signature)
        logger.debug(f"Using size={args.size}")

    score_threshold = args.min_score

    label_mapping: Optional[dict[str, str]] = None
    if args.label_mapping is not None:
        with open(args.label_mapping, "r", encoding="utf-8") as handle:
            label_mapping = json.load(handle)

    if label_mapping is None:
        detection_label_remap = None
    else:
        class_to_idx, detection_label_remap = build_label_mapping_indices(model_class_to_idx, label_mapping)

    # Process per-class minimum scores
    class_min_scores: dict[str, float] = {}
    if args.class_min_score is not None:
        for class_name, score_str in args.class_min_score:
            score = float(score_str)
            if class_name not in class_to_idx:
                logger.warning(f"Class '{class_name}' from --class-min-score not found in evaluation classes")
            else:
                class_min_scores[class_name] = score
                logger.info(f"Using minimum score {score} for class '{class_name}'")

    # Set label colors
    cmap = plt.get_cmap("jet")
    color_list = []
    for c in np.linspace(0, 1, len(class_to_idx) + 1):  # Include background
        rgb = cmap(c)[0:3]
        rgb = tuple(int(x * 255) for x in rgb)
        color_list.append(rgb)

    batch_size = args.batch_size
    inference_transform = InferenceTransform(args.size, rgb_stats, args.dynamic_size, args.max_size, args.no_resize)
    if args.wds is True:
        wds_path: str | list[str]
        if args.wds_info is not None:
            wds_path, dataset_size = wds_args_from_info(args.wds_info, args.wds_split)
            if args.wds_size is not None:
                dataset_size = args.wds_size
        else:
            wds_path, dataset_size = prepare_wds_args(args.data_path[0], args.wds_size, device, select_suffix="json")

        labeled = True
        root_path = Path("")
        if model_class_to_idx == class_to_idx:
            label_remap = None
        else:
            label_remap = detection_label_remap

        dataset = make_wds_detection_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=args.shuffle,
            transform=inference_transform,
            samples_names=True,
            return_orig_sizes=True,
            label_remap=label_remap,
        )
        num_samples = dataset_size
        inference_loader = make_wds_loader(
            dataset,
            batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=inference_collate_fn,
            world_size=1,
            pin_memory=False,
            exact=True,
        )

    elif args.coco_json_path is not None:
        labeled = True
        root_path = Path(args.data_path[0])
        dataset = CocoInference(
            root_path,
            args.coco_json_path,
            transforms=inference_transform,
        )
        if label_mapping is not None:
            dataset.convert_annotations_with_label_mapping(label_mapping, target_class_to_idx=class_to_idx)

        if dataset.class_to_idx != class_to_idx:
            logger.warning("Dataset class to index differs from model")

        num_samples = len(dataset)
        inference_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=inference_collate_fn,
        )
    else:
        labeled = False
        root_path = Path("")
        dataset = make_image_dataset(
            args.data_path,
            {},
            transforms=inference_transform,
            return_orig_sizes=True,
        )

        num_samples = len(dataset)
        inference_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=inference_collate_fn,
        )

    show_flag = args.show is True

    def batch_callback(
        file_paths: list[str],
        _inputs: torch.Tensor,
        detections: list[dict[str, torch.Tensor]],
        _targets: list[dict[str, Any]],
        _image_sizes: list[tuple[int, int]],
    ) -> None:
        # Show flags
        if show_flag is True:
            for img_path, detection in zip(file_paths, _remap_detection_labels(detections, detection_label_remap)):
                show_detections(
                    str(root_path.joinpath(img_path)),
                    detection,
                    class_to_idx=class_to_idx,
                    score_threshold=score_threshold,
                    class_min_scores=class_min_scores,
                    color_list=color_list,
                )

    # Sort out output file names
    epoch_str = ""
    if args.epoch is not None:
        epoch_str = f"_e{args.epoch}"

    base_output_path = f"{network_name}_{len(class_to_idx)}{epoch_str}_{args.size[0]}px_{num_samples}"
    if args.tta is True:
        base_output_path = f"{base_output_path}_tta"
    if args.model_dtype != "float32":
        base_output_path = f"{base_output_path}_{args.model_dtype}"
    if args.prefix is not None:
        base_output_path = f"{args.prefix}_{base_output_path}"
    if args.suffix is not None:
        base_output_path = f"{base_output_path}_{args.suffix}"

    output_path = settings.RESULTS_DIR.joinpath(f"{base_output_path}_output.json")

    # Inference
    tic = time.time()
    with torch.inference_mode():
        sample_paths, detections, targets = infer_dataloader(
            device,
            net,
            inference_loader,
            tta=args.tta,
            channels_last=args.channels_last,
            model_dtype=model_dtype,
            amp=args.amp,
            amp_dtype=amp_dtype,
            num_samples=num_samples,
            batch_callback=batch_callback,
        )

    detections = _remap_detection_labels(detections, detection_label_remap)

    toc = time.time()
    rate = num_samples / (toc - tic)
    logger.info(f"{lib.format_duration(toc-tic)} to process {num_samples:,} samples ({rate:.2f} samples/sec)")

    # Save output
    if args.save_output is True:
        save_output(output_path, sample_paths, class_to_idx, detections)

    # Handle results
    if labeled is True:
        results = Results(sample_paths, targets, detections, class_to_idx)
        if args.save_results is True:
            results.save(f"{base_output_path}.json")

        results.log_short_report()

    else:
        if args.save_results is True:
            logger.warning("Annotations not provided, unable to save results")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run detection prediction on directories and/or files",
        epilog=(
            "Usage example:\n"
            "python -m birder.scripts.predict_detection --network faster_rcnn --backbone resnext_101 "
            "-e 0 data/detection_data/validation\n"
            "python predict_detection.py --network retinanet --backbone resnext_101 "
            "-e 0 --shuffle --show --gpu --compile data/detection_data/training\n"
            "python predict_detection.py -n yolo_v4 --backbone csp_resnet_50 --backbone-tag imagenet1k "
            "-t coco -e 19 --batch-size 1 --gpu --gpu-id 1 --coco-json-path "
            "~/Datasets/cocodataset/annotations/instances_val2017.json ~/Datasets/cocodataset/val2017\n"
            "python predict_detection.py --network faster_rcnn -t coco --backbone csp_resnet_50 "
            "--backbone-tag imagenet1k -e 0 --batch-size 1 --gpu --gpu-id 1 "
            "--coco-json-path data/detection_data/validation_annotations_coco.json data/detection_data\n"
            "python predict_detection.py -n lw_detr_2stg --tag coco --backbone rope_i_vit_reg1_s16_pn_npn_avg_c1 "
            "--backbone-tag pe-spatial --backbone-model-config '{\"out_indices\":[5,8,11]}' -e 49 --batch-size 4 "
            "--gpu --parallel --size 640 --wds --wds-info ~/Datasets/cocodataset/val2017_packed/_info.json\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("-n", "--network", type=str, help="the neural network to use (i.e. faster_rcnn)")
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the model default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument("--backbone", type=str, help="the neural network to used as backbone")
    parser.add_argument(
        "--backbone-model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the backbone default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument("--backbone-tag", type=str, help="backbone training log tag (loading only)")
    parser.add_argument(
        "--backbone-reparameterized", default=False, action="store_true", help="load reparameterized backbone"
    )
    parser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    parser.add_argument("--quantized", default=False, action="store_true", help="load quantized model")
    parser.add_argument("-t", "--tag", type=str, help="model tag (from the training phase)")
    parser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    parser.add_argument("--pts", default=False, action="store_true", help="load torchscript network")
    parser.add_argument("--pt2", default=False, action="store_true", help="load standardized model")
    parser.add_argument("--st", "--safetensors", default=False, action="store_true", help="load Safetensors weights")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-backbone", default=False, action="store_true", help="enable backbone only compilation"
    )
    parser.add_argument("--channels-last", default=False, action="store_true", help="use channels-last memory format")
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="model dtype to use",
    )
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    parser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument("--tta", default=False, action="store_true", help="test time augmentation (oversampling)")
    parser.add_argument("--min-score", type=float, default=0.5, help="prediction score threshold")
    parser.add_argument(
        "--class-min-score",
        action="append",
        nargs=2,
        metavar=("CLASS", "SCORE"),
        help="set custom minimum score for specific class (can be used multiple times)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        metavar=("H", "W"),
        help=(
            "target image size as [height, width], if --dynamic-size is enabled, "
            "uses the smaller dimension as target size while preserving aspect ratio (defaults to model's signature)"
        ),
    )
    parser.add_argument(
        "--max-size",
        type=int,
        help="maximum size for the longer edge of resized images, when specified, enables dynamic sizing",
    )
    parser.add_argument(
        "--dynamic-size",
        default=False,
        action="store_true",
        help="allow variable image sizes while preserving aspect ratios",
    )
    parser.add_argument(
        "--no-resize", default=False, action="store_true", help="process images at original size without resizing"
    )
    parser.add_argument("--batch-size", type=int, default=8, metavar="N", help="the batch size")
    parser.add_argument("-j", "--num-workers", type=int, default=4, metavar="N", help="number of preprocessing workers")
    parser.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
    )
    parser.add_argument("--show", default=False, action="store_true", help="show image predictions")
    parser.add_argument("--shuffle", default=False, action="store_true", help="predict samples in random order")
    parser.add_argument("--save-results", default=False, action="store_true", help="save results object")
    parser.add_argument("--save-output", default=False, action="store_true", help="save raw output as JSON")
    parser.add_argument("--prefix", type=str, help="add prefix to output file")
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use (ignored in parallel mode)")
    parser.add_argument("--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device")
    parser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpus")
    parser.add_argument("--wds", default=False, action="store_true", help="predict a webdataset directory")
    parser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds dataset")
    parser.add_argument("--wds-info", type=str, action="append", metavar="FILE", help="one or more wds info file paths")
    parser.add_argument("--wds-split", type=str, default="validation", metavar="NAME", help="wds dataset split to load")
    parser.add_argument("--coco-json-path", type=str, help="COCO json path")
    parser.add_argument(
        "--label-mapping", type=str, metavar="FILE", help="JSON mapping of source labels to target labels"
    )
    parser.add_argument("data_path", nargs="*", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)

    if args.network is None:
        raise cli.ValidationError("--network is required")
    if args.backbone is None:
        raise cli.ValidationError("--backbone is required")
    if registry.exists(args.backbone, net_type=DetectorBackbone) is False:
        raise cli.ValidationError(
            f"--backbone {args.backbone} not supported, see list-models tool for available options"
        )
    if args.min_score >= 1 or args.min_score <= 0.0:
        raise cli.ValidationError(f"--min-score must be in range of (0, 1.0), got {args.min_score}")
    if args.class_min_score is not None:
        for class_name, score_str in args.class_min_score:
            try:
                score = float(score_str)
                if score >= 1.0 or score <= 0.0:
                    raise cli.ValidationError(
                        f"--class-min-score for '{class_name}' must be in range of (0, 1.0), got {score}"
                    )
            except ValueError as e:
                raise cli.ValidationError(f"--class-min-score value must be a valid float, got '{score_str}'") from e
    if args.parallel is True and args.gpu is False:
        raise cli.ValidationError("--parallel requires --gpu to be set")
    if args.parallel is True and args.compile_backbone is True:
        raise cli.ValidationError("--parallel cannot be used with --compile-backbone")
    if args.compile is True and args.compile_backbone is True:
        raise cli.ValidationError("--compile cannot be used with --compile-backbone")
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")
    if args.wds is False and len(args.data_path) == 0:
        raise cli.ValidationError("Must provide at least one data source, --data-path or --wds")
    if args.wds is True:
        if args.coco_json_path is not None:
            raise cli.ValidationError("--wds cannot be used with --coco-json-path")
        if args.wds_info is None and len(args.data_path) == 0:
            raise cli.ValidationError("--wds requires a data path unless --wds-info is provided")
        if len(args.data_path) > 1:
            raise cli.ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")
        if args.wds_info is None and len(args.data_path) == 1:
            data_path = args.data_path[0]
            if "://" in data_path and args.wds_size is None:
                raise cli.ValidationError("--wds-size is required for remote --data-path")

        if args.show is True:
            raise cli.ValidationError("--wds cannot be used with --show")

    if args.coco_json_path is not None and len(args.data_path) > 1:
        raise cli.ValidationError(f"--coco-json-path can have at most 1 --data-path, got {len(args.data_path)}")
    if args.label_mapping is not None and args.coco_json_path is None and args.wds is False:
        raise cli.ValidationError("--label-mapping requires --coco-json-path or --wds")


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    parser.set_defaults(**kwargs)
    args = parser.parse_args([])
    validate_args(args)

    return args


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

    if settings.RESULTS_DIR.exists() is False:
        logger.info(f"Creating {settings.RESULTS_DIR} directory...")
        settings.RESULTS_DIR.mkdir(parents=True)

    predict(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
