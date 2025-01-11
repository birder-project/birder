import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
import torch.amp
from torch.utils.data import DataLoader

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.dataloader.webdataset import make_wds_loader
from birder.datasets.directory import make_image_dataset
from birder.datasets.webdataset import make_wds_dataset
from birder.datasets.webdataset import wds_size
from birder.inference.classification import infer_dataloader
from birder.results.classification import Results
from birder.results.gui import show_top_k
from birder.transforms.classification import inference_preset


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
    if label != -1:
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


# pylint: disable=too-many-locals,too-many-branches
def predict(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.parallel is True and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} {device} devices")
    else:
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)

        logging.info(f"Using device {device}")

    network_name = lib.get_network_name(args.network, net_param=args.net_param, tag=args.tag)
    (net, class_to_idx, signature, rgb_stats) = fs_ops.load_model(
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
    )

    if args.show_class is not None:
        if args.show_class not in class_to_idx:
            logging.warning("Select show class is not part of the model classes")

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.compile is True:
        net = torch.compile(net)
        if args.save_embedding is True:
            net.embedding = torch.compile(net.embedding)

    if args.parallel is True and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    if args.size is None:
        args.size = lib.get_size_from_signature(signature)[0]
        logging.debug(f"Using size={args.size}")

    batch_size = args.batch_size
    inference_transform = inference_preset((args.size, args.size), rgb_stats, args.center_crop)
    if args.wds is True:
        (wds_path, _) = fs_ops.wds_braces_from_path(Path(args.data_path[0]))
        dataset_size = wds_size(wds_path, device)
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
            num_workers=8,
            prefetch_factor=2,
            collate_fn=None,
            world_size=1,
            pin_memory=False,
        )

    else:
        dataset = make_image_dataset(args.data_path, class_to_idx, transforms=inference_transform)
        num_samples = len(dataset)
        inference_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=8,
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

    tic = time.time()
    with torch.inference_mode():
        (sample_paths, outs, labels, embedding_list) = infer_dataloader(
            device,
            net,
            inference_loader,
            args.save_embedding,
            args.tta,
            args.amp,
            num_samples,
            batch_callback=batch_callback,
        )

    toc = time.time()
    rate = len(outs) / (toc - tic)
    (minutes, seconds) = divmod(toc - tic, 60)
    logging.info(f"{int(minutes):0>2}m{seconds:04.1f}s to classify {len(outs):,} samples ({rate:.2f} samples/sec)")

    label_names = list(class_to_idx.keys())

    # Save embeddings
    epoch_str = ""
    if args.epoch is not None:
        epoch_str = f"_e{args.epoch}"

    base_output_path = (
        f"{network_name}_{len(class_to_idx)}{epoch_str}_{args.size}px_crop{args.center_crop}_{num_samples}"
    )
    if args.tta is True:
        base_output_path = f"{base_output_path}_tta"
    if args.suffix is not None:
        base_output_path = f"{base_output_path}_{args.suffix}"

    if args.save_embedding is True:
        embeddings = np.concatenate(embedding_list, axis=0)
        embeddings_df = pl.DataFrame(embeddings)
        embeddings_df = pl.DataFrame(
            {
                "sample": sample_paths,
                **{f"{i}": embeddings[:, i] for i in range(embeddings.shape[-1])},
            }
        )
        embeddings_df = embeddings_df.sort("sample", descending=False)
        embeddings_path = settings.RESULTS_DIR.joinpath(f"{base_output_path}_embeddings.csv")
        logging.info(f"Saving embeddings at {embeddings_path}")
        embeddings_df.write_csv(embeddings_path)

    # Save output
    if args.save_output is True:
        output_df = pl.DataFrame(
            {
                "sample": sample_paths,
                "prediction": np.array(label_names)[outs.argmax(axis=1)],
                **{name: outs[:, i] for i, name in enumerate(label_names)},
            }
        )
        output_df = output_df.sort("sample", descending=False)
        output_path = settings.RESULTS_DIR.joinpath(f"{base_output_path}_output.csv")
        logging.info(f"Saving output at {output_path}")
        output_df.write_csv(output_path)

    # Handle results
    results = Results(sample_paths, labels, label_names, output=outs)
    if results.missing_all_labels is False:
        if args.save_results is True:
            results.save(f"{base_output_path}.csv")

        results.log_short_report()

    else:
        logging.warning("No labeled samples found")

    # Summary
    if args.summary is True:
        summary_df = results.prediction_names.value_counts(sort=True)
        indent_size = summary_df["prediction_names"].str.len_chars().max() + 2  # type: ignore[operator]
        for specie_name, count in summary_df.iter_rows():
            logging.info(f"{specie_name:<{indent_size}} {count}")


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
            "python predict.py -n efficientnet_v2_m -e 0 --gpu --save-embedding data/testing\n"
            "python predict.py -n efficientnet_v1_b4 -e 300 --gpu --save-embedding "
            "data/*/Alpine\\ swift --suffix alpine_swift\n"
            "python predict.py -n mobilevit_v2 -p 1.5 -t intermediate -e 80 --gpu --save-results "
            "--wds data/validation_packed\n"
            "python predict.py -n efficientnet_v2_m -t intermediate --show-class Unknown data/raw_data\n"
            "python predict.py -n convnext_v2_tiny -t intermediate -e 70 --gpu --gpu-id 1 --compile --fast-matmul "
            "--show-class Unknown data/raw_data\n"
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
    parser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    parser.add_argument("--quantized", default=False, action="store_true", help="load quantized model")
    parser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    parser.add_argument(
        "-r", "--reparameterized", default=False, action="store_true", help="load reparameterized model"
    )
    parser.add_argument("--pts", default=False, action="store_true", help="load torchscript network")
    parser.add_argument("--pt2", default=False, action="store_true", help="load standardized model")
    parser.add_argument("--st", default=False, action="store_true", help="load Safetensors weights")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    parser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument("--tta", default=False, action="store_true", help="test time augmentation (oversampling)")
    parser.add_argument("--size", type=int, help="image size for inference (defaults to model signature)")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    parser.add_argument("--center-crop", type=float, default=1.0, help="Center crop ratio to use during inference")
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
    parser.add_argument("--save-embedding", default=False, action="store_true", help="save features layer output")
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use (ignored in parallel mode)")
    parser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpu's")
    parser.add_argument("--wds", default=False, action="store_true", help="predict a webdataset directory")
    parser.add_argument("data_path", nargs="+", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    assert args.network is not None
    assert args.center_crop <= 1 and args.center_crop > 0, "Center crop ratio must be between 0 and 1"
    assert args.parallel is False or args.gpu is True
    assert args.save_embedding is False or args.parallel is False
    assert args.save_embedding is False or args.tta is False
    assert args.parallel is False or args.compile is False
    assert args.wds is False or len(args.data_path) == 1
    assert args.wds is False or (
        args.show is False and args.show_mistakes is False and args.show_out_of_k is False and args.show_class is None
    )


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    args = argparse.Namespace(**kwargs)
    args = parser.parse_args([], args)
    validate_args(args)

    return args


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

    if settings.RESULTS_DIR.exists() is False:
        logging.info(f"Creating {settings.RESULTS_DIR} directory...")
        settings.RESULTS_DIR.mkdir(parents=True)

    predict(args)


if __name__ == "__main__":
    main()
