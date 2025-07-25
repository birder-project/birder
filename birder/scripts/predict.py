import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from torch.utils.data import DataLoader

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import inference_preset
from birder.inference.classification import infer_dataloader_iter
from birder.results.classification import Results
from birder.results.gui import show_top_k

logger = logging.getLogger(__name__)


def save_embeddings(
    embeddings_path: Path, sample_paths: list[str], embedding_list: list[npt.NDArray[np.float32]], append: bool
) -> None:
    embeddings = np.concatenate(embedding_list, axis=0)
    data = {
        "sample": sample_paths,
        **{f"{i}": embeddings[:, i] for i in range(embeddings.shape[-1])},
    }
    embeddings_df = pl.DataFrame(data).sort("sample", descending=False)

    if append is False:
        logger.info(f"Saving embeddings at {embeddings_path}")
        embeddings_df.write_csv(embeddings_path)
    else:
        logger.info(f"Adding embeddings to {embeddings_path}")
        with open(embeddings_path, "a", encoding="utf-8") as handle:
            embeddings_df.write_csv(handle, include_header=False)


def save_output(
    output_path: Path, sample_paths: list[str], label_names: list[str], outs: npt.NDArray[np.float32], append: bool
) -> None:
    output_df = pl.DataFrame(
        {
            "sample": sample_paths,
            "prediction": np.array(label_names)[outs.argmax(axis=1)],
            **{name: outs[:, i] for i, name in enumerate(label_names)},
        }
    )
    output_df = output_df.sort("sample", descending=False)
    if append is False:
        logger.info(f"Saving output at {output_path}")
        output_df.write_csv(output_path)
    else:
        logger.info(f"Adding output to {output_path}")
        with open(output_path, "a", encoding="utf-8") as handle:
            output_df.write_csv(handle, include_header=False)


def handle_show_flags(
    args: argparse.Namespace,
    img_path: str,
    prob: npt.NDArray[np.float32],
    label: int,
    class_to_idx: dict[str, int],
) -> None:
    # Show prediction
    if args.show is True:
        show_top_k(img_path, prob, class_to_idx, label)
    elif args.show_top_below is not None and args.show_top_below > prob.max():
        show_top_k(img_path, prob, class_to_idx, label)

    # Show mistake (if label exists)
    if label != settings.NO_LABEL:
        if args.show_target_below is not None and args.show_target_below > prob[label]:
            show_top_k(img_path, prob, class_to_idx, label)

        elif args.show_mistakes is True:
            if label != np.argmax(prob):
                show_top_k(img_path, prob, class_to_idx, label)

        elif args.show_out_of_k is True:
            if label not in np.argsort(prob)[::-1][0 : settings.TOP_K]:
                show_top_k(img_path, prob, class_to_idx, label)

        elif args.show_class is not None:
            idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
            if args.show_class == idx_to_class[np.argmax(prob)]:  # type: ignore
                show_top_k(img_path, prob, class_to_idx, label)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
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
    if args.amp_dtype is None:
        amp_dtype = torch.get_autocast_dtype(device.type)
    else:
        amp_dtype = getattr(torch, args.amp_dtype)

    network_name = lib.get_network_name(args.network, net_param=args.net_param, tag=args.tag)
    (net, (class_to_idx, signature, rgb_stats, *_)) = fs_ops.load_model(
        device,
        args.network,
        net_param=args.net_param,
        config=args.model_config,
        tag=args.tag,
        epoch=args.epoch,
        new_size=args.size,
        quantized=args.quantized,
        inference=True,
        reparameterized=args.reparameterized,
        pts=args.pts,
        pt2=args.pt2,
        st=args.st,
        dtype=model_dtype,
    )

    if args.ignore_dir_names is True:
        num_classes = len(class_to_idx)
        class_to_idx = fs_ops.class_to_idx_from_paths(args.data_path, hierarchical=args.hierarchical)
        assert len(class_to_idx) == num_classes

    if args.class_mapping is not None:
        num_classes = len(class_to_idx)
        class_mapping = fs_ops.read_json_class_file(args.class_mapping)
        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
        idx_to_class.update(dict(zip(class_mapping.values(), class_mapping.keys())))
        class_to_idx = dict(zip(idx_to_class.values(), idx_to_class.keys()))
        assert len(class_to_idx) == num_classes

    if args.show_class is not None:
        if args.show_class not in class_to_idx:
            logger.warning("Selected 'show class' is not part of the model classes")

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.compile is True:
        net = torch.compile(net)
        if args.save_embeddings is True:
            net.embedding = torch.compile(net.embedding)

    if args.parallel is True and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    if args.size is None:
        args.size = lib.get_size_from_signature(signature)
        logger.debug(f"Using size={args.size}")

    batch_size = args.batch_size
    inference_transform = inference_preset(args.size, rgb_stats, args.center_crop)
    if args.wds is True:
        wds_path: str | list[str]
        if args.wds_info is not None:
            (wds_path, dataset_size) = wds_args_from_info(args.wds_info, args.wds_split)
            if args.wds_size is not None:
                dataset_size = args.wds_size
        else:
            (wds_path, dataset_size) = prepare_wds_args(args.data_path[0], args.wds_size, device)

        num_samples = dataset_size
        dataset = make_wds_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=args.shuffle,
            samples_names=True,
            transform=inference_transform,
        )
        inference_loader = make_wds_loader(
            dataset,
            batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=None,
            world_size=1,
            pin_memory=False,
        )

    else:
        dataset = make_image_dataset(
            args.data_path, class_to_idx, transforms=inference_transform, hierarchical=args.hierarchical
        )
        num_samples = len(dataset)
        inference_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

    show_flag = (
        args.show is True
        or args.show_top_below is not None
        or args.show_target_below is not None
        or args.show_mistakes is True
        or args.show_out_of_k is True
        or args.show_class is not None
    )

    def batch_callback(file_paths: list[str], out: npt.NDArray[np.float32], batch_labels: list[int]) -> None:
        # Show flags
        if show_flag is True:
            for img_path, prob, label in zip(file_paths, out, batch_labels):
                handle_show_flags(args, img_path, prob, label, class_to_idx)

    # Sort out output file names
    epoch_str = ""
    if args.epoch is not None:
        epoch_str = f"_e{args.epoch}"

    base_output_path = (
        f"{network_name}_{len(class_to_idx)}{epoch_str}_{args.size[0]}px_crop{args.center_crop}_{num_samples}"
    )
    if args.tta is True:
        base_output_path = f"{base_output_path}_tta"
    if args.model_dtype != "float32":
        base_output_path = f"{base_output_path}_{args.model_dtype}"
    if args.suffix is not None:
        base_output_path = f"{base_output_path}_{args.suffix}"

    embeddings_path = settings.RESULTS_DIR.joinpath(f"{base_output_path}_embeddings.csv")
    output_path = settings.RESULTS_DIR.joinpath(f"{base_output_path}_output.csv")
    label_names = list(class_to_idx.keys())

    # Inference
    tic = time.time()
    infer_iter = infer_dataloader_iter(
        device,
        net,
        inference_loader,
        args.save_embeddings,
        args.tta,
        model_dtype,
        args.amp,
        amp_dtype,
        num_samples,
        batch_callback=batch_callback,
        chunk_size=args.chunk_size,
        **args.forward_kwargs,
    )
    append = False  # Append mode for outputs, only False for the first batch
    summary_list = []
    with torch.inference_mode():
        for sample_paths, outs, labels, embedding_list in infer_iter:
            # Save embeddings
            if args.save_embeddings is True:
                save_embeddings(embeddings_path, sample_paths, embedding_list, append=append)

            if len(class_to_idx) > 0:
                # Save output
                if args.save_output is True:
                    save_output(output_path, sample_paths, label_names, outs, append=append)

                # Handle results
                results = Results(sample_paths, labels, label_names, output=outs)
                if results.missing_all_labels is False:
                    if args.save_results is True:
                        results.save(f"{base_output_path}.csv", append=append)
                    if args.chunk_size is None:
                        results.log_short_report()

                else:
                    logger.warning("No labeled samples found")

                # Summary
                if args.summary is True:
                    summary_list.append(results.prediction_names.value_counts())

            append = True

    toc = time.time()
    rate = num_samples / (toc - tic)
    (minutes, seconds) = divmod(toc - tic, 60)
    logger.info(f"{int(minutes):0>2}m{seconds:04.1f}s to classify {num_samples:,} samples ({rate:.2f} samples/sec)")

    #  Print summary
    if args.summary is True:
        summary_df = pl.concat(summary_list).group_by("prediction_names").agg(pl.col("count").sum())
        summary_df = summary_df.sort(by="count", descending=True)
        indent_size = summary_df["prediction_names"].str.len_chars().max() + 2  # type: ignore[operator]
        for specie_name, count in summary_df.iter_rows():
            logger.info(f"{specie_name:<{indent_size}} {count}")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run prediction on directories and/or files",
        epilog=(
            "Usage example:\n"
            "python predict.py --network resnet_v2_50 --pts --gpu --save-output data/Unknown\n"
            "python predict.py -n fastvit_t8 -t il-common_reparameterized -e 0 --batch-size 256 --reparameterized "
            "--gpu --save-results data/validation_il-common_packed\n"
            "python predict.py --network inception_resnet_v2 -e 100 --gpu --show-out-of-k data/validation\n"
            "python predict.py --network inception_v3 --gpu --batch-size 256 --save-results data/validation/*crane\n"
            "python predict.py -n efficientnet_v2_m -e 0 --gpu --save-embeddings data/testing\n"
            "python predict.py -n efficientnet_v1_b4 -e 300 --gpu --save-embeddings "
            "data/*/Alpine\\ swift --suffix alpine_swift\n"
            "python predict.py -n mobilevit_v2 -p 1.5 -t intermediate -e 80 --gpu --save-results "
            "--wds data/validation_packed\n"
            "python predict.py -n efficientnet_v2_m -t intermediate --show-class Unknown data/raw_data\n"
            "python predict.py -n convnext_v2_tiny -t intermediate -e 70 --gpu --gpu-id 1 --compile --fast-matmul "
            "--show-class Unknown data/raw_data\n"
            "python predict.py -n rope_vit_reg4_b14 --gpu --amp --ignore-dir-names data/imagenet-v2\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("-n", "--network", type=str, help="the neural network to use (i.e. resnet_v2)")
    parser.add_argument("-p", "--net-param", type=float, help="network specific parameter, required by some networks")
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help=(
            "override the model default configuration, accepts key-value pairs or JSON "
            "('drop_path_rate=0.2' or '{\"units\": [3, 24, 36, 3], \"dropout\": 0.2}'"
        ),
    )
    parser.add_argument(
        "--forward-kwargs",
        action=cli.FlexibleDictAction,
        help=(
            "additional model forward/embedding keyword args, accepts key-value pairs or JSON "
            "('patch_size=12' or '{\"patch_size\": 20}'"
        ),
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
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    parser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument("--tta", default=False, action="store_true", help="test time augmentation (oversampling)")
    parser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    parser.add_argument("-j", "--num-workers", type=int, default=8, metavar="N", help="number of preprocessing workers")
    parser.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
    )
    parser.add_argument(
        "--chunk-size", type=int, metavar="N", help="process in chunks of N samples to reduce memory usage"
    )
    parser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during inference")
    parser.add_argument("--show", default=False, action="store_true", help="show image predictions")
    parser.add_argument("--show-top-below", type=float, help="show when top prediction is below given threshold")
    parser.add_argument("--show-target-below", type=float, help="show when target prediction is below given threshold")
    parser.add_argument("--show-mistakes", default=False, action="store_true", help="show only mis-classified images")
    parser.add_argument("--show-out-of-k", default=False, action="store_true", help="show images not in the top-k")
    parser.add_argument("--show-class", type=str, help="show specific class predictions")
    parser.add_argument("--shuffle", default=False, action="store_true", help="predict samples in random order")
    parser.add_argument("--summary", default=False, action="store_true", help="log prediction summary")
    parser.add_argument("--save-results", default=False, action="store_true", help="save results object")
    parser.add_argument("--save-output", default=False, action="store_true", help="save raw output as CSV")
    parser.add_argument("--save-embeddings", default=False, action="store_true", help="save embedding layer outputs")
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use (ignored in parallel mode)")
    parser.add_argument("--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device")
    parser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpus")
    parser.add_argument("--wds", default=False, action="store_true", help="predict a webdataset directory")
    parser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds dataset")
    parser.add_argument("--wds-info", type=str, metavar="FILE", help="wds info file path")
    parser.add_argument("--wds-split", type=str, default="validation", metavar="NAME", help="wds dataset split to load")
    parser.add_argument(
        "--ignore-dir-names",
        default=False,
        action="store_true",
        help="use directory indices (0, 1, 2, ...) as class labels instead of directory names",
    )
    parser.add_argument(
        "--class-mapping", type=str, help="json file with alternative class names (can be partial mapping)"
    )
    parser.add_argument(
        "--hierarchical",
        default=False,
        action="store_true",
        help="use hierarchical directory structure for labels (e.g., 'dir1/subdir2' -> 'dir1_subdir2' label)",
    )
    parser.add_argument("data_path", nargs="*", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)
    if args.forward_kwargs is None:
        args.forward_kwargs = {}

    if args.network is None:
        raise cli.ValidationError("--network is required")
    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")
    if args.parallel is True and args.gpu is False:
        raise cli.ValidationError("--parallel requires --gpu to be set")
    if args.parallel is True and args.compile is True:
        raise cli.ValidationError("--parallel cannot be used with --compile")
    if args.save_embeddings is True and args.parallel is True:
        raise cli.ValidationError("--save-embeddings cannot be used with --parallel")
    if args.save_embeddings is True and args.tta is True:
        raise cli.ValidationError("--save-embeddings cannot be used with --tta")
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")

    if args.wds is False and len(args.data_path) == 0:
        raise cli.ValidationError("Must provide at least one data source, --data-path or --wds")
    if args.wds is True and len(args.data_path) > 1:
        raise cli.ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")
    if args.wds is True and args.hierarchical is True:
        raise cli.ValidationError("--wds cannot be used with --hierarchical")
    if args.wds is True and args.ignore_dir_names is True:
        raise cli.ValidationError("--wds cannot be used with --ignore-dir-names")
    if args.wds is True and (  # pylint: disable=too-many-boolean-expressions
        args.show is True
        or args.show_top_below is not None
        or args.show_target_below is not None
        or args.show_mistakes is True
        or args.show_out_of_k is True
        or args.show_class is not None
    ):
        raise cli.ValidationError("--wds cannot be used with any of the 'show' flags")


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
    logger = logging.getLogger(__spec__.name)
    main()
