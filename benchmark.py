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


def dummy(arg: Any) -> None:
    type(arg)


def benchmark(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logging.info(f"Using device {device}")

    if args.fast_matmul is True:
        torch.set_float32_matmul_precision("high")

    results = []
    model_list = birder.list_pretrained_models(args.filter)
    for model_name in model_list:
        (net, _, signature, _) = birder.load_pretrained_model(model_name, inference=True, device=device)
        if args.compile is True:
            net = torch.compile(net)

        if args.size is None:
            size = birder.get_size_from_signature(signature)
        else:
            size = (args.size, args.size)

        sample_shape = (args.batch_size, signature["inputs"][0]["data_shape"][1]) + size

        # Warmup
        logging.info(f"Starting warmup for {model_name}")
        with torch.amp.autocast(device.type, enabled=args.amp):
            for _ in range(args.warmup):
                output = net(torch.randn(sample_shape, device=device))

        # Benchmark
        logging.info(f"Starting benchmark for {model_name}")
        with torch.amp.autocast(device.type, enabled=args.amp):
            t_start = time.perf_counter()
            for _ in range(args.bench_iter):
                output = net(torch.randn(sample_shape, device=device))

            t_end = time.perf_counter()
            t_elapsed = t_end - t_start

        dummy(output)

        num_samples = args.bench_iter * args.batch_size
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

    results_df = pl.DataFrame(results)

    output_path = f"benchmark_{device.type}"
    if args.suffix is not None:
        output_path = f"{output_path}_{args.suffix}"

    benchmark_path = settings.RESULTS_DIR.joinpath(f"{output_path}.csv")
    logging.info(f"Saving results at {benchmark_path}")
    results_df.write_csv(benchmark_path)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Benchmark pretrained models",
        epilog=(
            "Usage example:\n"
            "python benchmark.py --compile --suffix all"
            "python benchmark.py --filter '*il-common*' --compile --suffix il-common\n"
            "python benchmark.py --filter '*il-common*' --batch-size 512 --gpu\n"
            "python benchmark.py --filter '*il-common*' --batch-size 512 --gpu --warmup 20\n"
            "python benchmark.py --filter '*il-common*' --batch-size 512 --gpu --fast-matmul --compile "
            "--suffix il-common\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("--filter", type=str, help="models to benchmark (fnmatch type filter)")
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    parser.add_argument(
        "--fast-matmul",
        default=False,
        action="store_true",
        help="use fast matrix multiplication (affects precision)",
    )
    parser.add_argument("--size", type=int, default=None, help="image size for inference (defaults to model signature)")
    parser.add_argument("--batch-size", type=int, default=1, help="the batch size")
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, help="gpu id to use (ignored in parallel mode)")
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup iterations")
    parser.add_argument("--bench-iter", type=int, default=100, help="number of benchmark iterations")

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
