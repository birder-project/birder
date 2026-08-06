import argparse
import logging
import multiprocessing as mp
import signal
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import NamedTuple
from typing import Optional

import polars as pl
import torch
import torch.amp

import birder
from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.model_registry import Task
from birder.model_registry import registry
from birder.net.base import DetectorBackbone

# Spawned multiprocessing workers execute this module as __mp_main__
# Use the canonical module name from the spec so worker logs remain under the configured birder logger hierarchy
logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


class WeightsSpec(NamedTuple):
    model_name: str
    weights_path: Path
    cfg_path: Path


class BenchmarkOutcome(NamedTuple):
    result: Optional[dict[str, Any]]
    error: Optional[str]


def dummy(arg: Any) -> None:
    type(arg)


def prepare_model(net: torch.nn.Module) -> None:
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)


def init_plain_model(
    model_name: str, sample_shape: tuple[int, ...], device: torch.device, args: argparse.Namespace
) -> torch.nn.Module:
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    size = (sample_shape[2], sample_shape[3])
    input_channels = sample_shape[1]
    if args.backbone is not None:
        backbone = registry.net_factory(args.backbone, args.num_classes, input_channels, size=size)
        net = registry.detection_net_factory(model_name, args.num_classes, backbone, size=size)
    else:
        net = registry.net_factory(model_name, args.num_classes, input_channels, size=size)

    net.to(device, dtype=model_dtype)
    if args.channels_last is True:
        net = net.to(memory_format=torch.channels_last)
        logger.debug("Using channels-last memory format")

    prepare_model(net)

    return net


def resolve_weights_specs(weights: Optional[list[str]]) -> list[WeightsSpec]:
    if weights is None:
        return []

    resolved_specs = []
    for weight_path in weights:
        weights_path = Path(weight_path)
        cfg_path = weights_path.with_suffix(".json")
        if weights_path.exists() is False:
            raise cli.ValidationError(f"weights file not found: {weights_path}")
        if cfg_path.exists() is False:
            raise cli.ValidationError(f"config file not found for {weights_path}: expected {cfg_path}")

        resolved_specs.append(WeightsSpec(model_name=weights_path.stem, weights_path=weights_path, cfg_path=cfg_path))

    return resolved_specs


def get_weights_size(spec: WeightsSpec) -> tuple[int, int]:
    cfg = fs_ops.read_config_from_path(spec.cfg_path)

    return lib.get_size_from_signature(cfg["signature"])


def init_weights_model(spec: WeightsSpec, device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    net, _ = fs_ops.load_model_with_cfg(spec.cfg_path, spec.weights_path)
    net.to(device, dtype=model_dtype)
    if args.size is not None:
        net.adjust_size(args.size)
    if args.channels_last is True:
        net = net.to(memory_format=torch.channels_last)
        logger.debug("Using channels-last memory format")

    prepare_model(net)

    return net


def throughput_benchmark(
    net: torch.nn.Module, device: torch.device, sample_shape: tuple[int, ...], model_name: str, args: argparse.Namespace
) -> tuple[float, int]:
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)

    def _make_sample() -> torch.Tensor:
        sample = torch.rand(sample_shape, device=device, dtype=model_dtype)
        if args.channels_last is True:
            sample = sample.to(memory_format=torch.channels_last)

        return sample

    # Sanity
    if args.amp_dtype is None:
        amp_dtype = torch.get_autocast_dtype(device.type)
    else:
        amp_dtype = getattr(torch, args.amp_dtype)

    logger.info(
        f"Sanity check for {model_name}: size={sample_shape[2:]} device={device.type} compile={args.compile} "
        f"model_dtype={model_dtype} amp={args.amp} amp_dtype={amp_dtype} channels_last={args.channels_last}"
    )

    batch_size = sample_shape[0]
    pool_size = args.pool_size
    input_pool: list[torch.Tensor] = []
    while batch_size > 0:
        input_pool = [_make_sample() for _ in range(pool_size)]
        with torch.inference_mode():
            with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
                try:
                    for sample in input_pool:
                        output = net(sample)
                    break
                except Exception as e:  # pylint: disable=broad-exception-caught
                    input_pool = []
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    batch_size -= 32
                    sample_shape = (batch_size, *sample_shape[1:])
                    logger.info(f"Error in sanity check: {e}")
                    logger.info(f"Reducing batch size to {batch_size}")

    if batch_size <= 0:
        logger.warning(f"Aborting benchmark for {model_name}: batch size reduced to 0")
        return (-1.0, 0)

    # Warmup
    logger.info(f"Warmup for {model_name}: {args.warmup} iterations")
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            for _ in range(args.warmup):
                output = net(input_pool[_ % pool_size])

    if args.cooldown > 0.0:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

        time.sleep(args.cooldown)

    # Benchmark
    logger.info(f"Throughput benchmark for {model_name}: repeats={args.repeats} bench_iter={args.bench_iter}")
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)

            t_elapsed = 0.0
            for repeat_idx in range(args.repeats):
                t_start = time.perf_counter()
                for i in range(args.bench_iter):
                    output = net(input_pool[i % pool_size])

                if device.type == "cuda":
                    torch.cuda.synchronize(device=device)

                t_elapsed += time.perf_counter() - t_start
                if repeat_idx < args.repeats - 1 and args.cooldown > 0.0:
                    time.sleep(args.cooldown)

    dummy(output)

    return (t_elapsed, batch_size)


def memory_benchmark(sample_shape: tuple[int, ...], model_spec: str | WeightsSpec, args: argparse.Namespace) -> float:
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
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

    if isinstance(model_spec, WeightsSpec):
        model_name = model_spec.model_name
    else:
        model_name = model_spec

    logger.info(
        f"Memory benchmark for {model_name}: batch={sample_shape[0]} size={sample_shape[2:]} device={device.type} "
        f"compile={args.compile} model_dtype={model_dtype} amp={args.amp} amp_dtype={amp_dtype} "
        f"channels_last={args.channels_last}"
    )

    if args.plain is True:
        net = init_plain_model(model_name, sample_shape, device, args)
    elif isinstance(model_spec, WeightsSpec):
        net = init_weights_model(model_spec, device, args)

    else:
        net, _ = birder.load_pretrained_model(model_name, inference=True, device=device, dtype=model_dtype)
        if args.size is not None:
            size = (sample_shape[2], sample_shape[3])
            net.adjust_size(size)
        if args.channels_last is True:
            net = net.to(memory_format=torch.channels_last)
            logger.debug("Using channels-last memory format")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
            sample = torch.rand(sample_shape, device=device, dtype=model_dtype)
            if args.channels_last is True:
                sample = sample.to(memory_format=torch.channels_last)
            for _ in range(5):
                net(sample)

    return torch.cuda.max_memory_allocated(device)  # type: ignore[no-any-return]


def run_model_benchmark(
    model_spec: str | WeightsSpec, sample_shape: tuple[int, ...], args: argparse.Namespace
) -> dict[str, Any]:
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.single_thread is True:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    if isinstance(model_spec, WeightsSpec):
        model_name = model_spec.model_name
    else:
        model_name = model_spec

    if args.memory is True:
        samples_per_sec = None
        peak_memory = memory_benchmark(sample_shape, model_spec, args) / (1024 * 1024)
        logger.info(f"{model_name} peak memory: {peak_memory:.2f} MB")
    else:
        if isinstance(model_spec, WeightsSpec):
            net = init_weights_model(model_spec, device, args)
        elif args.plain is True:
            net = init_plain_model(model_name, sample_shape, device, args)
        else:
            net, _ = birder.load_pretrained_model(model_name, inference=True, device=device, dtype=model_dtype)
            if args.size is not None:
                net.adjust_size((sample_shape[2], sample_shape[3]))
            if args.channels_last is True:
                net = net.to(memory_format=torch.channels_last)
                logger.debug("Using channels-last memory format")

        if args.compile is True:
            torch.compiler.reset()
            net = torch.compile(net)

        peak_memory = None
        t_elapsed, batch_size = throughput_benchmark(net, device, sample_shape, model_name, args)
        if t_elapsed < 0.0:
            raise RuntimeError("sanity check failed for every attempted batch size")

        num_samples = args.repeats * args.bench_iter * batch_size
        samples_per_sec = num_samples / t_elapsed
        ms_per_sample = 1000.0 * t_elapsed / num_samples
        logger.info(
            f"{model_name} throughput: {samples_per_sec:.2f} samples/s, {ms_per_sample:.2f} ms/sample "
            f"(batch={batch_size})"
        )

    return {
        "model_name": model_name,
        "device": device.type,
        "single_thread": args.single_thread,
        "compile": args.compile,
        "model_dtype": args.model_dtype,
        "amp": args.amp,
        "fast_matmul": args.fast_matmul,
        "channels_last": args.channels_last,
        "size": sample_shape[2],
        "max_batch_size": args.max_batch_size,
        "memory": args.memory,
        "torch_version": torch.__version__,
        "samples_per_sec": samples_per_sec,
        "peak_memory": peak_memory,
    }


def model_benchmark_worker(
    result_connection: Any,
    model_spec: str | WeightsSpec,
    sample_shape: tuple[int, ...],
    args: argparse.Namespace,
) -> None:
    try:
        result = run_model_benchmark(model_spec, sample_shape, args)
        result_connection.send(BenchmarkOutcome(result=result, error=None))
    except Exception:  # pylint: disable=broad-exception-caught
        result_connection.send(BenchmarkOutcome(result=None, error=traceback.format_exc()))
    finally:
        result_connection.close()


def process_exit_description(exit_code: Optional[int]) -> str:
    if exit_code is None:
        return "worker did not exit"
    if exit_code < 0:
        try:
            signal_name = signal.Signals(-exit_code).name
        except ValueError:
            signal_name = f"signal {-exit_code}"

        return f"worker terminated by {signal_name}"

    return f"worker exited with code {exit_code}"


def benchmark(args: argparse.Namespace) -> None:
    mp_context = mp.get_context("spawn")

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

    # Used just for bookkeeping, actual init happens on each worker
    if args.gpu is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device {device}")

    include_header = existing_df is None
    attempted_benchmarks = 0
    successful_benchmarks = 0
    failures: list[tuple[str, str]] = []
    model_list: Sequence[str | WeightsSpec]
    if len(args.weights) > 0:
        model_list = args.weights
    elif args.plain is True:
        model_list = args.models or []
        if len(model_list) == 0:
            task = Task.OBJECT_DETECTION if args.backbone is not None else Task.IMAGE_CLASSIFICATION
            model_list = registry.list_models(include_filter=args.filter, task=task)

    else:
        model_list = birder.list_pretrained_models(args.filter)

    logger.info(f"Found {len(model_list)} models to benchmark")
    for model_spec in model_list:
        if isinstance(model_spec, WeightsSpec):
            model_name = model_spec.model_name
        else:
            model_name = model_spec

        if isinstance(model_spec, WeightsSpec):
            cfg = fs_ops.read_config_from_path(model_spec.cfg_path)
            input_channels = lib.get_channels_from_signature(cfg["signature"])
            if args.size is not None:
                size = args.size
            else:
                size = lib.get_size_from_signature(cfg["signature"])

        elif args.plain is True:
            input_channels = settings.DEFAULT_NUM_CHANNELS
            if args.size is not None:
                size = args.size
            else:
                size = registry.get_default_size(model_name)

        else:
            model_metadata = registry.get_pretrained_metadata(model_name)
            input_channels = settings.DEFAULT_NUM_CHANNELS
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
                    "model_dtype": args.model_dtype,
                    "amp": args.amp,
                    "fast_matmul": args.fast_matmul,
                    "channels_last": args.channels_last,
                    "size": size[0],
                    "max_batch_size": args.max_batch_size,
                    "memory": args.memory,
                }
            ).is_empty()
            if combination_exists is False:
                logger.info(f"Skipping {model_name}: configuration already exists in {benchmark_path}")
                continue

        sample_shape = (args.max_batch_size, input_channels) + size
        attempted_benchmarks += 1

        # Keep this process boundary even though Python exceptions are handled in the worker. torch.compile can
        # terminate the interpreter with native failures such as SIGABRT or SIGSEGV, which try/except cannot catch.
        result_connection, worker_connection = mp_context.Pipe(duplex=False)
        worker = mp_context.Process(
            target=model_benchmark_worker,
            args=(worker_connection, model_spec, sample_shape, args),
            name=f"benchmark-{model_name}",
        )
        worker.start()
        worker_connection.close()
        try:
            outcome = result_connection.recv()
        except EOFError:
            outcome = None
        except BaseException:
            if worker.is_alive() is True:
                worker.terminate()

            worker.join()
            raise
        finally:
            result_connection.close()

        worker.join()
        if worker.exitcode != 0:
            failures.append((model_name, process_exit_description(worker.exitcode)))
            continue
        if outcome is None:
            failures.append((model_name, "worker exited without returning a result"))
            continue
        if outcome.error is not None:
            failures.append((model_name, outcome.error))
            continue

        assert outcome.result is not None
        successful_benchmarks += 1
        if args.dry_run is False:
            mode = "w" if include_header is True else "a"
            logger.info(f"Saving successful result for {model_name} at {benchmark_path}")
            with open(benchmark_path, mode=mode, encoding="utf-8") as handle:
                pl.DataFrame([outcome.result]).write_csv(handle, include_header=include_header)

            include_header = False

    if args.dry_run is True:
        logger.info("Dry run enabled, skipping saving outputs")

    if len(failures) > 0:
        failure_count = len(failures)
        failure_summary = (
            f"{failure_count}/{attempted_benchmarks} attempted benchmarks failed; {successful_benchmarks} succeeded"
        )
        logger.error(failure_summary)
        for model_name, error in failures:
            logger.error(f"Benchmark failed for {model_name}:\n{error}")

        raise SystemExit(1)

    logger.info(f"All {attempted_benchmarks} attempted benchmarks succeeded")


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
            "python -m birder.scripts.benchmark --plain --models retinanet --backbone resnet_v1_50 --num-classes 91 "
            "--size 640 --gpu --dry-run\n"
            "python -m birder.scripts.benchmark --weights models/vit_reg4_s14_nps_ls_dino-v2-lvd142m.pt "
            "--gpu --dry-run\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("--filter", type=str, help="models to benchmark (fnmatch type filter)")
    parser.add_argument("--weights", nargs="+", help="weight files to benchmark using sibling .json configs")
    parser.add_argument("--models", nargs="+", help="plain network names to benchmark")
    parser.add_argument("--plain", default=False, action="store_true", help="benchmark plain networks without weights")
    parser.add_argument("--backbone", type=str, help="backbone name for plain detection benchmarks")
    parser.add_argument(
        "--num-classes", type=int, default=0, metavar="N", help="number of classes for plain benchmarks"
    )
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
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
    parser.add_argument("--warmup", type=int, default=10, metavar="N", help="number of warmup iterations")
    parser.add_argument("--repeats", type=int, default=3, metavar="N", help="number of repetitions")
    parser.add_argument("--bench-iter", type=int, default=300, metavar="N", help="number of benchmark iterations")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="cooldown after warmup and between benchmark repetitions",
    )
    parser.add_argument("--pool-size", type=int, default=1, metavar="N", help="number of input tensors to rotate")
    parser.add_argument("--memory", default=False, action="store_true", help="benchmark memory instead of throughput")
    parser.add_argument("--append", default=False, action="store_true", help="append to existing output file")
    parser.add_argument("--dry-run", default=False, action="store_true", help="run without reading or writing results")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)
    args.weights = resolve_weights_specs(args.weights)

    if args.single_thread is True and args.gpu is True:
        raise cli.ValidationError("--single-thread cannot be used with --gpu")
    if args.memory is True and args.gpu is False:
        raise cli.ValidationError("--memory requires --gpu")
    if args.memory is True and args.compile is True:
        raise cli.ValidationError("--memory cannot be used with --compile")
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")
    if len(args.weights) > 0 and args.plain is True:
        raise cli.ValidationError("--weights cannot be used with --plain")
    if len(args.weights) > 0 and args.filter is not None:
        raise cli.ValidationError("--weights cannot be used with --filter")
    if len(args.weights) > 0 and args.models is not None:
        raise cli.ValidationError("--weights cannot be used with --models")
    if args.plain is False and args.models is not None:
        raise cli.ValidationError("--models can only be used with --plain")
    if args.backbone is not None and args.plain is False:
        raise cli.ValidationError("--backbone can only be used with --plain")
    if args.backbone is not None and registry.exists(args.backbone, net_type=DetectorBackbone) is False:
        raise cli.ValidationError(
            f"--backbone {args.backbone} not supported, see list-models tool for available options"
        )


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
    main()
