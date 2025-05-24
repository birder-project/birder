import argparse
import logging
import multiprocessing as mp
import time
from typing import Any

import polars as pl
import torch
import torch.amp

import birder
from birder.common import cli
from birder.conf import settings
from birder.model_registry import registry

logger = logging.getLogger(__name__)


def dummy(arg: Any) -> None:
    type(arg)


def throughput_benchmark(
    net: torch.nn.Module, device: torch.device, sample_shape: tuple[int, ...], model_name: str, args: argparse.Namespace
) -> tuple[float, int]:
    # Sanity
    logger.info(f"Sanity check for {model_name}")

    if args.amp_dtype is None:
        amp_dtype = torch.get_autocast_dtype(device.type)
    else:
        amp_dtype = getattr(torch, args.amp_dtype)

    batch_size = sample_shape[0]
    while batch_size > 0:
        with torch.inference_mode():
            with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
                try:
                    output = net(torch.rand(sample_shape, device=device))
                    output = net(torch.rand(sample_shape, device=device))
                    break
                except Exception:  # pylint: disable=broad-exception-caught
                    batch_size -= 32
                    sample_shape = (batch_size, *sample_shape[1:])
                    logger.info(f"Reducing batch size to: {batch_size}")

    if batch_size <= 0:
        logger.warning(f"Aborting {model_name} !!!")
        return (-1.0, 0)

    # Warmup
    logger.info(f"Starting warmup for {model_name}")
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            for _ in range(args.warmup):
                output = net(torch.rand(sample_shape, device=device))

    # Benchmark
    logger.info(f"Starting benchmark for {model_name}")
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            t_start = time.perf_counter()
            for _ in range(args.repeats):
                for _ in range(args.bench_iter):
                    output = net(torch.rand(sample_shape, device=device))

            t_end = time.perf_counter()
            t_elapsed = t_end - t_start

    dummy(output)

    return (t_elapsed, batch_size)


def memory_benchmark(
    sync_peak_memory: Any, sample_shape: tuple[int, ...], model_name: str, args: argparse.Namespace
) -> None:
    logger.info(f"Starting memory benchmark for {model_name}")
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    if args.amp_dtype is None:
        amp_dtype = torch.get_autocast_dtype(device.type)
    else:
        amp_dtype = getattr(torch, args.amp_dtype)

    (net, _) = birder.load_pretrained_model(model_name, inference=True, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            sample = torch.rand(sample_shape, device=device)
            for _ in range(5):
                net(sample)

    peak_memory: float = torch.cuda.max_memory_allocated(device)
    sync_peak_memory.value = peak_memory


# pylint: disable=too-many-branches,too-many-locals
def benchmark(args: argparse.Namespace) -> None:
    mp.set_start_method("spawn")

    torch_version = torch.__version__
    output_path = "benchmark"
    if args.suffix is not None:
        output_path = f"{output_path}_{args.suffix}"

    benchmark_path = settings.RESULTS_DIR.joinpath(f"{output_path}.csv")
    if benchmark_path.exists() is True and args.append is False:  # pylint: disable=no-else-raise
        logger.warning("Benchmark file already exists... aborting")
        raise SystemExit(1)
    elif benchmark_path.exists() is True:
        logger.info(f"Loading {benchmark_path}...")
        existing_df = pl.read_csv(benchmark_path)
    else:
        existing_df = None

    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.single_thread is True:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    input_channels = 3
    results = []
    model_list = birder.list_pretrained_models(args.filter)
    for model_name in model_list:
        model_metadata = registry.get_pretrained_metadata(model_name)
        size = model_metadata["resolution"]

        # Check if model already benchmarked at this configuration
        if existing_df is not None:
            combination_exists = existing_df.filter(
                **{
                    "model_name": model_name,
                    "device": device.type,
                    "single_thread": args.single_thread,
                    "compile": args.compile,
                    "amp": args.amp,
                    "fast_matmul": args.fast_matmul,
                    "size": size[0],
                    "max_batch_size": args.max_batch_size,
                    "memory": args.memory,
                }
            ).is_empty()
            if combination_exists is False:
                logger.info(f"{model_name} at the current configuration is already on file, skipping...")
                continue

        sample_shape = (args.max_batch_size, input_channels) + size

        if args.memory is True:
            samples_per_sec = None
            sync_peak_memory = mp.Value("d", 0.0)
            p = mp.Process(target=memory_benchmark, args=(sync_peak_memory, sample_shape, model_name, args))
            p.start()
            p.join()
            peak_memory = sync_peak_memory.value / (1024 * 1024)
            logger.info(f"{model_name} used {peak_memory:.2f}MB")
        else:
            # Initialize model
            (net, _) = birder.load_pretrained_model(model_name, inference=True, device=device)
            if args.compile is True:
                torch.compiler.reset()
                net = torch.compile(net)

            peak_memory = None
            (t_elapsed, batch_size) = throughput_benchmark(net, device, sample_shape, model_name, args)
            if t_elapsed < 0.0:
                continue

            num_samples = args.repeats * args.bench_iter * batch_size
            samples_per_sec = num_samples / t_elapsed
            logger.info(f"{model_name} ran at {samples_per_sec:.2f} samples / sec")

        results.append(
            {
                "model_name": model_name,
                "device": device.type,
                "single_thread": args.single_thread,
                "compile": args.compile,
                "amp": args.amp,
                "fast_matmul": args.fast_matmul,
                "size": size[0],
                "max_batch_size": args.max_batch_size,
                "memory": args.memory,
                "torch_version": torch_version,
                "samples_per_sec": samples_per_sec,
                "peak_memory": peak_memory,
            }
        )

    results_df = pl.DataFrame(results)

    if args.append is True and existing_df is not None:
        include_header = False
        mode = "a"
    else:
        include_header = True
        mode = "w"

    logger.info(f"Saving results at {benchmark_path}")
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
            "python benchmark.py --filter '*il-common*' --max-batch-size 512 --gpu\n"
            "python benchmark.py --filter '*il-common*' --max-batch-size 512 --gpu --warmup 20\n"
            "python benchmark.py --filter '*il-common*' --max-batch-size 512 --gpu --fast-matmul --compile "
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
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    parser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument("--max-batch-size", type=int, default=1, metavar="N", help="the max batch size to try")
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument("--single-thread", default=False, action="store_true", help="use CPU with a single thread")
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    parser.add_argument("--warmup", type=int, default=20, metavar="N", help="number of warmup iterations")
    parser.add_argument("--repeats", type=int, default=3, metavar="N", help="number of repetitions")
    parser.add_argument("--bench-iter", type=int, default=300, metavar="N", help="number of benchmark iterations")
    parser.add_argument("--memory", default=False, action="store_true", help="benchmark memory instead of throughput")
    parser.add_argument("--append", default=False, action="store_true", help="append to existing output file")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    assert args.single_thread is False or args.gpu is False
    assert args.memory is False or args.gpu is True
    assert args.memory is False or args.compile is False


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
        logger.info(f"Creating {settings.RESULTS_DIR} directory...")
        settings.RESULTS_DIR.mkdir(parents=True)

    benchmark(args)


if __name__ == "__main__":
    logger = logging.getLogger(__spec__.name)
    main()
