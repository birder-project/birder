import argparse
import logging
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

from birder.common import cli
from birder.common.lib import get_network_name
from birder.conf import settings
from birder.core.dataloader.webdataset import make_wds_loader
from birder.core.datasets.directory import ImageListDataset
from birder.core.datasets.webdataset import make_wds_dataset
from birder.core.datasets.webdataset import wds_size
from birder.core.inference import inference
from birder.core.results.classification import Results
from birder.core.results.gui import show_top_k
from birder.core.transforms.classification import inference_preset


def handle_show_flags(
    args: argparse.Namespace,
    img_path: str,
    prob: npt.NDArray[np.float32],
    label: int,
    class_to_idx: dict[str, int],
) -> None:
    # Show prediction
    if args.show is True:
        img = pil_loader(img_path)
        show_top_k(img, img_path, prob, label, class_to_idx)

    # Show mistake (if label exists)
    elif label != -1:
        if args.show_below is not None and args.show_below > prob[label]:
            img = pil_loader(img_path)
            show_top_k(img, img_path, prob, label, class_to_idx)

        elif args.show_mistakes is True:
            if label != np.argmax(prob):
                img = pil_loader(img_path)
                show_top_k(img, img_path, prob, label, class_to_idx)

        elif args.show_out_of_k is True:
            if label not in np.argsort(prob)[::-1][0 : settings.TOP_K]:
                img = pil_loader(img_path)
                show_top_k(img, img_path, prob, label, class_to_idx)

        elif args.show_class is not None:
            idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
            if args.show_class == idx_to_class[np.argmax(prob)]:  # type: ignore
                img = pil_loader(img_path)
                show_top_k(img, img_path, prob, label, class_to_idx)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def predict(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.parallel is True and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} {device} devices")
    else:
        logging.info(f"Using device {device}")

    network_name = get_network_name(args.network, net_param=args.net_param, tag=args.tag)
    (net, class_to_idx, signature, rgb_values) = cli.load_model(
        device,
        args.network,
        net_param=args.net_param,
        tag=args.tag,
        epoch=args.epoch,
        new_size=args.size,
        quantized=args.quantized,
        inference=True,
        pts=args.pts,
        pt2=args.pt2,
    )

    if args.show_class is not None:
        if args.show_class not in class_to_idx:
            logging.warning("Select show class is not part of the model classes")

    if args.fast_matmul is True:
        torch.set_float32_matmul_precision("high")

    if args.compile is True:
        net = torch.compile(net)
        if args.save_embedding is True:
            net.embedding = torch.compile(net.embedding)

    if args.parallel is True and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    if args.size is None:
        args.size = signature["inputs"][0]["data_shape"][2]
        logging.debug(f"Using size={args.size}")

    batch_size = args.batch_size
    inference_transform = inference_preset((args.size, args.size), args.center_crop, rgb_values)
    if args.wds is True:
        (wds_path, _) = cli.wds_braces_from_path(Path(args.data_path[0]))
        dataset_size = wds_size(wds_path, device)
        num_samples = dataset_size
        dataset = make_wds_dataset(
            wds_path,
            batch_size,
            dataset_size=dataset_size,
            shuffle=args.shuffle,
            samples_names=True,
            transform=inference_transform,
        )
        inference_loader = make_wds_loader(
            dataset,
            batch_size,
            shuffle=False,  # Shuffle is done at the wds dataset
            num_workers=8,
            prefetch_factor=2,
            collate_fn=None,
            world_size=1,
            pin_memory=False,
        )

    else:
        samples = cli.samples_from_paths(args.data_path, class_to_idx=class_to_idx)
        num_samples = len(samples)
        assert num_samples > 0, "Couldn't find any images"

        dataset = ImageListDataset(samples, transforms=inference_transform)
        inference_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=8,
        )

    embeddings: list[npt.NDArray[np.float32]] = []
    outs: list[npt.NDArray[np.float32]] = []
    labels: list[int] = []
    sample_paths: list[str] = []
    tic = time.time()
    with tqdm(total=num_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, targets in inference_loader:
            # Predict
            inputs = inputs.to(device)

            with torch.autocast(enabled=args.amp, device_type=device.type):
                (out, embedding) = inference.predict(net, inputs, return_embedding=args.save_embedding)

            outs.append(out)
            if embedding is not None:
                embeddings.append(embedding)

            # Set labels and sample list
            batch_labels = list(targets.cpu().numpy())
            labels.extend(batch_labels)
            sample_paths.extend(file_paths)

            # Show flags
            if (
                args.show is True
                or args.show_below is not None
                or args.show_mistakes is True
                or args.show_out_of_k is True
                or args.show_class is not None
            ):
                for img_path, prob, label in zip(file_paths, out, batch_labels):
                    handle_show_flags(args, img_path, prob, label, class_to_idx)

            # Update progress bar
            progress.update(n=batch_size)

    outs = list(np.concatenate(outs, axis=0))

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
        f"{network_name}_{len(class_to_idx)}{epoch_str}_{args.size}px_" f"crop{args.center_crop}_{num_samples}"
    )
    if args.suffix is not None:
        base_output_path = f"{base_output_path}_{args.suffix}"

    if args.save_embedding is True:
        embeddings = list(np.concatenate(embeddings, axis=0))
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.insert(0, "sample", sample_paths)
        embeddings_df = embeddings_df.sort_values(by="sample", ascending=True)
        embeddings_path = settings.RESULTS_DIR.joinpath(f"{base_output_path}_embeddings.csv")
        logging.info(f"Saving results at {embeddings_path}")
        embeddings_df.to_csv(embeddings_path, index=False)

    # Save output
    if args.save_output is True:
        output_df = pd.DataFrame(outs, columns=label_names)
        output_df.insert(0, "prediction", output_df.idxmax(axis=1))
        output_df.insert(0, "sample", sample_paths)
        output_df = output_df.sort_values(by="sample", ascending=True)
        output_path = settings.RESULTS_DIR.joinpath(f"{base_output_path}_output.csv")
        logging.info(f"Saving results at {output_path}")
        output_df.to_csv(output_path, index=False)

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
        summary = results.prediction_names.value_counts(sort=True).to_string()
        for line in summary.splitlines():
            logging.info(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run prediction on directories and/or files",
        epilog=(
            "Usage example:\n"
            "python predict.py --network resnet_v2 --net-param 50 --epoch 0 --pts --gpu "
            "--save-output data/Unknown\n"
            "python predict.py --network maxvit --net-param 1 --epoch 0 --pts --gpu "
            "--summary data/*.jpeg\n"
            "python predict.py --network densenet -p 121 -e 90 --shuffle --gpu --show data/validation\n"
            "python predict.py --network inception_resnet_v2 -e 100 --gpu --show-out-of-k data/validation\n"
            "python predict.py --network inception_v3 -e 200 --gpu --save-results data/validation/*crane\n"
            "python predict.py -n efficientnet_v2_m -e 200 --gpu --save-embedding data/validation\n"
            "python predict.py -n efficientnet_v1 -p 4 -e 300 --gpu --save-embedding "
            "data/*/Alpine\\ swift --suffix alpine_swift\n"
            "python predict.py -n convnext_v1 -p 2 -e 0 --gpu --parallel data/testing\n"
            "python predict.py -n mobilevit_v2 -p 1.5 -t intermediate -e 80 --gpu --save-results "
            "--wds data/validation_packed\n"
            "python predict.py -n efficientnet_v2_m -t intermediate --show-class Unknown data/raw_data\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("-n", "--network", type=str, required=True, help="the neural network to use (i.e. resnet_v2)")
    parser.add_argument("-p", "--net-param", type=float, help="network specific parameter, required for most networks")
    parser.add_argument("-e", "--epoch", type=int, help="model checkpoint to load")
    parser.add_argument("--quantized", default=False, action="store_true", help="load quantized model")
    parser.add_argument("-t", "--tag", type=str, help="model tag (from training phase)")
    parser.add_argument("--pts", default=False, action="store_true", help="load torchscript network")
    parser.add_argument("--pt2", default=False, action="store_true", help="load standardized model")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument("--amp", default=False, action="store_true", help="use torch.autocast")
    parser.add_argument(
        "--fast-matmul",
        default=False,
        action="store_true",
        help="use fast matrix multiplication (affects precision)",
    )
    parser.add_argument("--size", type=int, default=None, help="image size for inference (defaults to model signature)")
    parser.add_argument("--batch-size", type=int, default=32, help="the batch size")
    parser.add_argument("--center-crop", type=float, default=1.0, help="Center crop ratio to use during inference")
    parser.add_argument("--show", default=False, action="store_true", help="show image predictions")
    parser.add_argument(
        "--show-below", type=float, default=None, help="show when target prediction is below given threshold"
    )
    parser.add_argument("--show-mistakes", default=False, action="store_true", help="show only mis-classified images")
    parser.add_argument("--show-out-of-k", default=False, action="store_true", help="show images not in the top-k")
    parser.add_argument("--show-class", type=str, help="show specific class predictions")
    parser.add_argument("--shuffle", default=False, action="store_true", help="predict samples in random order")
    parser.add_argument("--summary", default=False, action="store_true", help="log prediction summary")
    parser.add_argument("--save-results", default=False, action="store_true", help="save results object")
    parser.add_argument("--save-output", default=False, action="store_true", help="save raw output as CSV")
    parser.add_argument(
        "--save-embedding", default=False, action="store_true", help="save features layer output as HDF5"
    )
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpu's")
    parser.add_argument("--wds", default=False, action="store_true", help="predict a webdataset directory")
    parser.add_argument("data_path", nargs="+", help="data files path (directories and files)")
    args = parser.parse_args()

    assert args.center_crop <= 1 and args.center_crop > 0, "Center crop ratio must be between 0 and 1"
    assert args.parallel is False or args.gpu is True
    assert args.save_embedding is False or args.parallel is False
    assert args.parallel is False or args.compile is False
    assert args.wds is False or len(args.data_path) == 1
    assert args.wds is False or (
        args.show is False and args.show_mistakes is False and args.show_out_of_k is False and args.show_class is False
    )

    if settings.RESULTS_DIR.exists() is False:
        logging.info(f"Creating {settings.RESULTS_DIR} directory...")
        settings.RESULTS_DIR.mkdir(parents=True)

    predict(args)


if __name__ == "__main__":
    main()
