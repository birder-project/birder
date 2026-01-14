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
from birder.model_registry import Task
from birder.model_registry import registry

logger = logging.getLogger(__name__)


def dummy(arg: Any) -> None:
    type(arg)


def prepare_model(net: torch.nn.Module) -> None:
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)


def throughput_benchmark(
    net: torch.nn.Module, device: torch.device, sample_shape: tuple[int, ...], model_name: str, args: argparse.Namespace
) -> tuple[float, int]:
    # Sanity
    if args.amp_dtype is None:
        amp_dtype = torch.get_autocast_dtype(device.type)
    else:
        amp_dtype = getattr(torch, args.amp_dtype)

    logger.info(
        f"Sanity check for {model_name}: size={sample_shape[2:]} device={device.type} compile={args.compile} "
        f"amp={args.amp} amp_dtype={amp_dtype}"
    )

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
                    logger.info(f"Reducing batch size to {batch_size}")

    if batch_size <= 0:
        logger.warning(f"Aborting benchmark for {model_name}: batch size reduced to 0")
        return (-1.0, 0)

    # Warmup
    logger.info(f"Warmup for {model_name}: {args.warmup} iterations")
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            for _ in range(args.warmup):
                output = net(torch.rand(sample_shape, device=device))

    # Benchmark
    logger.info(f"Throughput benchmark for {model_name}: repeats={args.repeats} bench_iter={args.bench_iter}")
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)

            t_start = time.perf_counter()
            for _ in range(args.repeats):
                for _ in range(args.bench_iter):
                    output = net(torch.rand(sample_shape, device=device))

            if device.type == "cuda":
                torch.cuda.synchronize(device=device)

            t_end = time.perf_counter()
            t_elapsed = t_end - t_start

    dummy(output)

    return (t_elapsed, batch_size)


def memory_benchmark(
    sync_peak_memory: Any, sample_shape: tuple[int, ...], model_name: str, args: argparse.Namespace
) -> None:
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

    logger.info(
        f"Memory benchmark for {model_name}: batch={sample_shape[0]} size={sample_shape[2:]} device={device.type} "
        f"compile={args.compile} amp={args.amp} amp_dtype={amp_dtype}"
    )

    if args.plain is True:
        size = (sample_shape[2], sample_shape[3])
        input_channels = sample_shape[1]
        net = registry.net_factory(model_name, input_channels, 0, size=size)
        net.to(device)
        prepare_model(net)

    else:
        (net, _) = birder.load_pretrained_model(model_name, inference=True, device=device)
        if args.size is not None:
            size = (sample_shape[2], sample_shape[3])
            net.adjust_size(size)

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
    if args.plain is True:
        output_path = "benchmark_plain"
    else:
        output_path = "benchmark"

    if args.suffix is not None:
        output_path = f"{output_path}_{args.suffix}"

    benchmark_path = settings.RESULTS_DIR.joinpath(f"{output_path}.csv")
    if args.dry_run is True:
        logger.debug("Dry run enabled, results will not be read or written")
        existing_df = None
    elif benchmark_path.exists() is True and args.append is False:
        logger.warning(f"Benchmark file {benchmark_path} already exists... aborting")
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
    if args.plain is True:
        model_list = args.models or []
        if len(model_list) == 0:
            model_list = registry.list_models(include_filter=args.filter, task=Task.IMAGE_CLASSIFICATION)

    else:
        model_list = birder.list_pretrained_models(args.filter)

    logger.info(f"Found {len(model_list)} models to benchmark")
    for model_name in model_list:
        if args.plain is True:
            if args.size is not None:
                size = args.size
            else:
                size = registry.get_default_size(model_name)

        else:
            model_metadata = registry.get_pretrained_metadata(model_name)
            if args.size is not None:
                size = args.size
            else:
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
                logger.info(f"Skipping {model_name}: configuration already exists in {benchmark_path}")
                continue

        sample_shape = (args.max_batch_size, input_channels) + size

        if args.memory is True:
            samples_per_sec = None
            sync_peak_memory = mp.Value("d", 0.0)
            p = mp.Process(target=memory_benchmark, args=(sync_peak_memory, sample_shape, model_name, args))
            p.start()
            p.join()
            peak_memory = sync_peak_memory.value / (1024 * 1024)
            logger.info(f"{model_name} peak memory: {peak_memory:.2f} MB")
        else:
            # Initialize model
            if args.plain is True:
                net = registry.net_factory(model_name, input_channels, 0, size=size)
                net.to(device)
                prepare_model(net)
            else:
                (net, _) = birder.load_pretrained_model(model_name, inference=True, device=device)
                if args.size is not None:
                    net.adjust_size(size)

            if args.compile is True:
                torch.compiler.reset()
                net = torch.compile(net)

            peak_memory = None
            (t_elapsed, batch_size) = throughput_benchmark(net, device, sample_shape, model_name, args)
            if t_elapsed < 0.0:
                continue

            num_samples = args.repeats * args.bench_iter * batch_size
            samples_per_sec = num_samples / t_elapsed
            logger.info(f"{model_name} throughput: {samples_per_sec:.2f} samples/s (batch={batch_size})")

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

    if args.dry_run is True:
        logger.info("Dry run enabled, skipping saving outputs")
        return

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
        description="Benchmark models",
        epilog=(
            "Usage example:\n"
            "python -m birder.scripts.benchmark --compile --suffix all\n"
            "python -m birder.scripts.benchmark --filter '*il-common*' --compile --suffix il-common\n"
            "python -m birder.scripts.benchmark --filter '*il-common*' --suffix il-common\n"
            "python -m birder.scripts.benchmark --filter '*il-common*' --max-batch-size 512 --gpu\n"
            "python -m birder.scripts.benchmark --filter '*il-common*' --max-batch-size 512 --gpu --warmup 20\n"
            "python -m birder.scripts.benchmark --filter '*il-common*' --max-batch-size 512 --gpu --fast-matmul "
            "--compile --suffix il-common --append\n"
            "python -m birder.scripts.benchmark --plain --models rdnet_t convnext_v1_tiny --bench-iter 50 --repeats 1 "
            "--gpu --size 416 --dry-run\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("--filter", type=str, help="models to benchmark (fnmatch type filter)")
    parser.add_argument("--models", nargs="+", help="plain network names to benchmark")
    parser.add_argument("--plain", default=False, action="store_true", help="benchmark plain networks without weights")
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
    parser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="input size override (defaults to model resolution)"
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
    parser.add_argument("--dry-run", default=False, action="store_true", help="run without reading or writing results")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)

    if args.single_thread is True and args.gpu is True:
        raise cli.ValidationError("--single-thread cannot be used with --gpu")
    if args.memory is True and args.gpu is False:
        raise cli.ValidationError("--memory requires --gpu")
    if args.memory is True and args.compile is True:
        raise cli.ValidationError("--memory cannot be used with --compile")
    if args.plain is False and args.models is not None:
        raise cli.ValidationError("--models can only be used with --plain")


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

    if args.dry_run is False:
        if settings.RESULTS_DIR.exists() is False:
            logger.info(f"Creating {settings.RESULTS_DIR} directory...")
            settings.RESULTS_DIR.mkdir(parents=True)

    benchmark(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
