import argparse
import logging
import time
from typing import Any

import polars as pl
import torch
import torch.amp

import birder
from birder.common import cli
from birder.conf import settings
from birder.model_registry import registry


def dummy(arg: Any) -> None:
    type(arg)


# pylint: disable=too-many-locals,too-many-branches
def benchmark(args: argparse.Namespace) -> None:
    output_path = "benchmark"
    if args.suffix is not None:
        output_path = f"{output_path}_{args.suffix}"

    benchmark_path = settings.RESULTS_DIR.joinpath(f"{output_path}.csv")
    if benchmark_path.exists() is True and args.append is False:  # pylint: disable=no-else-raise
        logging.warning("Benchmark file already exists... aborting")
        raise SystemExit(1)
    elif benchmark_path.exists() is True:
        logging.info(f"Loading {benchmark_path}...")
        existing_df = pl.read_csv(benchmark_path)
    else:
        existing_df = None

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logging.info(f"Using device {device}")

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    input_channels = 3
    num_classes = 500
    results = []
    model_list = birder.list_pretrained_models(args.filter)
    for model_name in model_list:
        model_info = registry.get_pretrained_info(model_name)
        network = model_info["net"]["network"]
        net_param = model_info["net"].get("net_param", None)
        if args.size is None:
            size = model_info["resolution"]
        else:
            size = (args.size, args.size)

        # Check if model already benchmarked at this configuration
        if existing_df is not None:
            combination_exists = existing_df.filter(
                **{
                    "model_name": model_name,
                    "device": device.type,
                    "compile": args.compile,
                    "amp": args.amp,
                    "fast_matmul": args.fast_matmul,
                    "size": size[0],
                    "batch_size": args.batch_size,
                }
            ).is_empty()
            if combination_exists is False:
                logging.info(f"{model_name} at the current configuration is already on file, skipping...")
                continue

        # Initialize model
        net = registry.net_factory(network, input_channels, num_classes, net_param=net_param, size=size[0])
        net.to(device)
        for param in net.parameters():
            param.requires_grad = False

        net.eval()
        if args.compile is True:
            torch.compiler.reset()
            net = torch.compile(net)

        sample_shape = (args.batch_size, input_channels) + size

        # Warmup
        logging.info(f"Starting warmup for {model_name}")
        with torch.inference_mode():
            with torch.amp.autocast(device.type, enabled=args.amp):
                for _ in range(args.warmup):
                    output = net(torch.randn(sample_shape, device=device))

        # Benchmark
        logging.info(f"Starting benchmark for {model_name}")
        with torch.inference_mode():
            with torch.amp.autocast(device.type, enabled=args.amp):
                t_start = time.perf_counter()
                for _ in range(args.repeats):
                    for _ in range(args.bench_iter):
                        output = net(torch.randn(sample_shape, device=device))

                t_end = time.perf_counter()
                t_elapsed = t_end - t_start

        dummy(output)

        num_samples = args.repeats * args.bench_iter * args.batch_size
        samples_per_sec = num_samples / t_elapsed
        results.append(
            {
                "model_name": model_name,
                "device": device.type,
                "compile": args.compile,
                "amp": args.amp,
                "fast_matmul": args.fast_matmul,
                "size": size[0],
                "batch_size": args.batch_size,
                "samples_per_sec": samples_per_sec,
            }
        )
        logging.info(f"{model_name} ran at {samples_per_sec:.2f} samples / sec")

    results_df = pl.DataFrame(results)

    if args.append is True and existing_df is not None:
        include_header = False
        mode = "a"
    else:
        include_header = True
        mode = "w"

    logging.info(f"Saving results at {benchmark_path}")
    with open(benchmark_path, mode=mode, encoding="utf-8") as handle:
        results_df.write_csv(handle, include_header=include_header)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Benchmark pretrained models",
        epilog=(
            "Usage example:\n"
            "python benchmark.py --compile --suffix all\n"
            "python benchmark.py --filter '*il-common*' --compile --suffix il-common\n"
            "python benchmark.py --filter '*il-common*' --suffix il-common\n"
            "python benchmark.py --filter '*il-common*' --batch-size 512 --gpu\n"
            "python benchmark.py --filter '*il-common*' --batch-size 512 --gpu --warmup 20\n"
            "python benchmark.py --filter '*il-common*' --batch-size 512 --gpu --fast-matmul --compile "
            "--suffix il-common --append\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("--filter", type=str, help="models to benchmark (fnmatch type filter)")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    parser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument("--size", type=int, help="image size for inference (defaults to model signature)")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N", help="the batch size")
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    parser.add_argument("--warmup", type=int, default=20, metavar="N", help="number of warmup iterations")
    parser.add_argument("--repeats", type=int, default=4, metavar="N", help="number of repetitions")
    parser.add_argument("--bench-iter", type=int, default=500, metavar="N", help="number of benchmark iterations")
    parser.add_argument("--append", default=False, action="store_true", help="append to existing output file")

    return parser


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    args = argparse.Namespace(**kwargs)
    args = parser.parse_args([], args)

    return args


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    if settings.RESULTS_DIR.exists() is False:
        logging.info(f"Creating {settings.RESULTS_DIR} directory...")
        settings.RESULTS_DIR.mkdir(parents=True)

    benchmark(args)


if __name__ == "__main__":
    main()
