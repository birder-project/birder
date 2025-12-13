import argparse
import logging
import os
import typing
from typing import Optional
from typing import get_args

from birder.common.cli import ValidationError
from birder.common.training_utils import OptimizerType
from birder.common.training_utils import SchedulerType
from birder.conf import settings
from birder.data.datasets.coco import MosaicType
from birder.data.transforms.classification import AugType
from birder.data.transforms.classification import RGBMode
from birder.data.transforms.detection import AugType as DetAugType

logger = logging.getLogger(__name__)


def add_compile_args(parser: argparse.ArgumentParser, teacher: bool = False, backbone: bool = False) -> None:
    group = parser.add_argument_group("Compilation parameters")
    group.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    if teacher is True:
        group.add_argument(
            "--compile-teacher", default=False, action="store_true", help="enable teacher only compilation"
        )
    if backbone is True:
        group.add_argument(
            "--compile-backbone", default=False, action="store_true", help="enable backbone only compilation"
        )

    group.add_argument(
        "--compile-opt", default=False, action="store_true", help="enable compilation for optimizer step"
    )


def add_optimization_args(parser: argparse.ArgumentParser, default_batch_size: int = 32) -> None:
    group = parser.add_argument_group("Optimization parameters")
    group.add_argument("--batch-size", type=int, default=default_batch_size, metavar="N", help="the batch size")
    group.add_argument("--opt", type=str, choices=list(get_args(OptimizerType)), default="sgd", help="optimizer to use")
    group.add_argument("--momentum", type=float, default=0.9, metavar="M", help="optimizer momentum")
    group.add_argument("--nesterov", default=False, action="store_true", help="use nesterov momentum")
    group.add_argument("--opt-eps", type=float, help="optimizer epsilon (None to use the optimizer default)")
    group.add_argument("--opt-betas", type=float, nargs="+", help="optimizer betas (None to use the optimizer default)")
    group.add_argument("--opt-alpha", type=float, help="optimizer alpha (None to use the optimizer default)")
    group.add_argument("--clip-grad-norm", type=float, help="the maximum gradient norm")
    group.add_argument(
        "--grad-accum-steps", type=int, default=1, metavar="N", help="number of steps to accumulate gradients"
    )


def add_lr_wd_args(parser: argparse.ArgumentParser, backbone_lr: bool = False, wd_end: bool = False) -> None:
    group = parser.add_argument_group("Learning rate and regularization parameters")
    group.add_argument("--lr", type=float, default=0.1, metavar="LR", help="base learning rate")
    group.add_argument("--bias-lr", type=float, metavar="LR", help="learning rate of biases")
    if backbone_lr is True:
        group.add_argument("--backbone-lr", type=float, metavar="LR", help="backbone learning rate")

    group.add_argument(
        "--lr-scale", type=int, help="reference batch size for LR scaling, if provided, LR will be scaled accordingly"
    )
    group.add_argument(
        "--lr-scale-type", type=str, choices=["linear", "sqrt"], default="linear", help="learning rate scaling type"
    )
    group.add_argument("--wd", type=float, default=0.0001, metavar="WD", help="weight decay")
    if wd_end is True:
        group.add_argument(
            "--wd-end", type=float, metavar="WD", help="final value of the weight decay (None for constant wd)"
        )

    group.add_argument("--norm-wd", type=float, metavar="WD", help="weight decay for Normalization layers")
    group.add_argument(
        "--bias-weight-decay", type=float, metavar="WD", help="weight decay for bias parameters of all layers"
    )
    group.add_argument(
        "--transformer-embedding-decay",
        type=float,
        metavar="WD",
        help="weight decay for embedding parameters for vision transformer models",
    )
    group.add_argument("--layer-decay", type=float, help="layer-wise learning rate decay (LLRD)")
    group.add_argument("--layer-decay-min-scale", type=float, help="minimum layer scale factor clamp value")
    group.add_argument(
        "--layer-decay-no-opt-scale", type=float, help="layer scale threshold below which parameters are frozen"
    )


def add_lr_scheduler_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Learning rate scheduler parameters")
    group.add_argument(
        "--lr-scheduler-update",
        type=str,
        choices=["epoch", "iter"],
        default="epoch",
        help="when to apply learning rate scheduler update",
    )
    group.add_argument(
        "--lr-scheduler",
        type=str,
        choices=list(get_args(SchedulerType)),
        default="constant",
        help="learning rate scheduler",
    )
    group.add_argument(
        "--lr-step-size",
        type=int,
        default=40,
        metavar="N",
        help="decrease lr every step-size epochs (for step scheduler only)",
    )
    group.add_argument(
        "--lr-steps", type=int, nargs="+", help="decrease lr every step-size epochs (multistep scheduler only)"
    )
    group.add_argument(
        "--lr-step-gamma",
        type=float,
        default=0.75,
        help="multiplicative factor of learning rate decay (for step scheduler only)",
    )
    group.add_argument(
        "--lr-cosine-min",
        type=float,
        default=0.000001,
        help="minimum learning rate (for cosine annealing scheduler only)",
    )
    group.add_argument(
        "--lr-power", type=float, default=1.0, help="power of the polynomial (for polynomial scheduler only)"
    )
    group.add_argument(
        "--lr-warmup-decay",
        type=float,
        default=0.01,
        help="multiplicative factor for learning rate at the start of warmup",
    )


def add_input_args(parser: argparse.ArgumentParser, size_help: Optional[str] = None) -> None:
    group = parser.add_argument_group("Input parameters")
    if size_help is None:
        size_help = "image size (defaults to the network default size)"

    group.add_argument(
        "--channels", type=int, default=settings.DEFAULT_NUM_CHANNELS, metavar="N", help="no. of image channels"
    )
    group.add_argument("--size", type=int, nargs="+", metavar=("H", "W"), help=size_help)


def add_detection_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Input parameters")
    group.add_argument(
        "--channels", type=int, default=settings.DEFAULT_NUM_CHANNELS, metavar="N", help="no. of image channels"
    )
    group.add_argument(
        "--size",
        type=int,
        nargs="+",
        metavar=("H", "W"),
        help=(
            "target image size as [height, width], if --dynamic-size is enabled, "
            "uses the smaller dimension as target size while preserving aspect ratio (defaults to model's signature)"
        ),
    )
    group.add_argument(
        "--max-size",
        type=int,
        help="maximum size for the longer edge of resized images, when specified, enables dynamic sizing",
    )
    group.add_argument(
        "--dynamic-size",
        default=False,
        action="store_true",
        help="allow variable image sizes while preserving aspect ratios",
    )
    group.add_argument("--multiscale", default=False, action="store_true", help="enable random scale per image")


def add_training_schedule_args(parser: argparse.ArgumentParser, default_epochs: int = 100) -> None:
    group = parser.add_argument_group("Training schedule parameters")
    group.add_argument("--epochs", type=int, default=default_epochs, metavar="N", help="number of training epochs")
    group.add_argument(
        "--stop-epoch", type=int, metavar="N", help="epoch to stop the training at (multi stage training)"
    )
    group.add_argument("--warmup-epochs", type=int, metavar="N", help="number of warmup epochs")
    group.add_argument("--warmup-iters", type=int, metavar="N", help="number of warmup iterations")
    group.add_argument("--cooldown-epochs", type=int, metavar="N", help="number of cooldown epochs (linear to zero)")
    group.add_argument("--cooldown-iters", type=int, metavar="N", help="number of cooldown iterations (linear to zero)")


def add_batch_norm_args(parser: argparse.ArgumentParser, backbone_freeze: bool = False) -> None:
    group = parser.add_argument_group("Batch normalization parameters")
    group.add_argument(
        "--freeze-bn",
        default=False,
        action="store_true",
        help="freeze all batch statistics and affine parameters of batchnorm2d layers",
    )
    if backbone_freeze is True:
        group.add_argument(
            "--freeze-backbone-bn",
            default=False,
            action="store_true",
            help="freeze all batch statistics and affine parameters of batchnorm2d layers (backbone only)",
        )

    group.add_argument("--sync-bn", default=False, action="store_true", help="use synchronized BatchNorm")


def add_data_aug_args(
    parser: argparse.ArgumentParser,
    default_level: int = 4,
    default_min_scale: Optional[float] = None,
    default_re_prob: Optional[float] = None,
    mixup_cutmix: bool = False,
) -> None:
    group = parser.add_argument_group("Data augmentation parameters")
    group.add_argument(
        "--aug-type", type=str, choices=list(get_args(AugType)), default="birder", help="augmentation type"
    )
    group.add_argument(
        "--aug-level",
        type=int,
        choices=list(range(10 + 1)),
        default=default_level,
        help="magnitude of birder augmentations (0 off -> 10 highest)",
    )
    group.add_argument(
        "--use-grayscale", default=False, action="store_true", help="use grayscale augmentation (birder aug only)"
    )
    group.add_argument(
        "--ra-num-ops",
        type=int,
        default=2,
        metavar="N",
        help="number of augmentation transformations to apply sequentially",
    )
    group.add_argument("--ra-magnitude", type=int, default=9, help="magnitude for all the RandAugment transformations")
    group.add_argument("--augmix-severity", type=int, default=3, help="severity of AugMix policy")
    group.add_argument("--resize-min-scale", type=float, default=default_min_scale, help="random resize min scale")
    group.add_argument(
        "--re-prob",
        type=float,
        default=default_re_prob,
        metavar="P",
        help="random erase probability (default according to aug-level)",
    )
    group.add_argument(
        "--simple-crop", default=False, action="store_true", help="use simple random crop (SRC) instead of RRC"
    )
    if mixup_cutmix is True:
        group.add_argument("--mixup-alpha", type=float, help="mixup alpha")
        group.add_argument("--cutmix", default=False, action="store_true", help="enable cutmix")

    group.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="birder",
        help="RGB mean and std to use for normalization",
    )
    group.add_argument(
        "--rgb-mean",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="set custom RGB mean values (overrides values from selected RGB mode)",
    )
    group.add_argument(
        "--rgb-std",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="set custom RGB std values (overrides values from selected RGB mode)",
    )


def add_detection_data_aug_args(parser: argparse.ArgumentParser, default_level: int = 4) -> None:
    group = parser.add_argument_group("Data augmentation parameters")
    group.add_argument(
        "--aug-type", type=str, choices=list(get_args(DetAugType)), default="birder", help="augmentation type"
    )
    group.add_argument(
        "--aug-level",
        type=int,
        choices=list(range(10 + 1)),
        default=default_level,
        help="magnitude of birder augmentations (0 off -> 10 highest)",
    )
    group.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="birder",
        help="RGB mean and std to use for normalization",
    )
    group.add_argument(
        "--rgb-mean",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="set custom RGB mean values (overrides values from selected RGB mode)",
    )
    group.add_argument(
        "--rgb-std",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="set custom RGB std values (overrides values from selected RGB mode)",
    )
    group.add_argument("--mosaic-prob", type=float, default=0.0, metavar="P", help="mosaic augmentation probability")
    group.add_argument(
        "--mosaic-type", type=str, choices=get_args(MosaicType), default="fixed_grid", help="mosaic augmentation type"
    )


def add_checkpoint_args(
    parser: argparse.ArgumentParser, default_save_frequency: int = 1, pretrained: bool = False
) -> None:
    group = parser.add_argument_group("Checkpoint parameters")
    group.add_argument(
        "--save-frequency", type=int, default=default_save_frequency, metavar="N", help="frequency of model saving"
    )
    group.add_argument(
        "--keep-last", type=int, metavar="N", help="number of recent checkpoints to keep (older ones are deleted)"
    )
    if pretrained is True:
        group.add_argument(
            "--pretrained",
            default=False,
            action="store_true",
            help="start with pretrained version of specified network (will download if not found locally)",
        )

    group.add_argument("--resume-epoch", type=int, metavar="N", help="epoch number to resume training from")
    group.add_argument(
        "--non-strict-weights",
        default=False,
        action="store_true",
        help="allow non-strict loading of model weights (missing or unexpected keys in state_dict)",
    )
    group.add_argument(
        "--load-states",
        default=False,
        action="store_true",
        help="load optimizer, scheduler and scaler states when resuming",
    )
    group.add_argument("--load-scheduler", default=False, action="store_true", help="load only scheduler when resuming")


def add_ema_args(
    parser: argparse.ArgumentParser, default_ema_steps: int = 32, default_ema_decay: float = 0.9999
) -> None:
    group = parser.add_argument_group("Exponential moving average parameters")
    group.add_argument(
        "--model-ema",
        default=False,
        action="store_true",
        help="enable tracking exponential moving average of model parameters",
    )
    group.add_argument(
        "--model-ema-steps",
        type=int,
        default=default_ema_steps,
        metavar="N",
        help="the number of iterations that controls how often to update the EMA model (adjusted to grad-accum-steps)",
    )
    group.add_argument(
        "--model-ema-decay",
        type=float,
        default=default_ema_decay,
        help="decay factor for exponential moving average of model parameters",
    )
    group.add_argument(
        "--model-ema-warmup",
        type=int,
        metavar="N",
        help="number of steps before EMA is applied (defaults to schedule warmup steps, pass 0 to disable warmup)",
    )


def add_dataloader_args(
    parser: argparse.ArgumentParser,
    no_img_loader: bool = False,
    default_num_workers: Optional[int] = None,
    default_drop_last: bool = False,
    ra_sampler: bool = False,
) -> None:
    group = parser.add_argument_group("Dataloader parameters")
    if no_img_loader is False:
        group.add_argument(
            "--img-loader", type=str, choices=["tv", "pil"], default="tv", help="backend to load and decode images"
        )

    if default_num_workers is None:
        default_num_workers = min(12, max(os.cpu_count() // 4, 4))  # type: ignore[operator]

    group.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=default_num_workers,
        metavar="N",
        help="number of preprocessing workers",
    )
    group.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
    )
    if default_drop_last is True:
        group.add_argument(
            "--no-drop-last",
            dest="drop_last",
            default=True,
            action="store_false",
            help="drop the last incomplete batch",
        )
    else:
        group.add_argument("--drop-last", default=False, action="store_true", help="drop the last incomplete batch")

    if ra_sampler is True:
        group.add_argument(
            "--ra-sampler", default=False, action="store_true", help="whether to use Repeated Augmentation in training"
        )
        group.add_argument(
            "--ra-reps", type=int, default=3, metavar="N", help="number of repetitions for Repeated Augmentation"
        )


def add_precision_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Precision parameters")
    group.add_argument(
        "--model-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="model dtype to use",
    )
    group.add_argument(
        "--amp",
        default=False,
        action="store_true",
        help="enable automatic mixed precision (AMP) training via torch.amp",
    )
    group.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    group.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )


def add_distributed_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Distributed training parameters")
    group.add_argument("--world-size", type=int, default=1, metavar="N", help="number of distributed processes")
    group.add_argument("--local-rank", type=int, metavar="N", help="local rank")
    group.add_argument("--dist-url", type=str, default="env://", help="URL used to initialize distributed training")
    group.add_argument("--dist-backend", type=str, default="nccl", help="distributed backend")
    group.add_argument(
        "--find-unused-parameters",
        default=False,
        action="store_true",
        help="enable searching for unused parameters in DistributedDataParallel (may impact performance)",
    )


def add_logging_and_debug_args(
    parser: argparse.ArgumentParser, default_log_interval: int = 50, fake_data: bool = True
) -> None:
    group = parser.add_argument_group("Logging and debugging parameters")
    group.add_argument(
        "--experiment",
        "--exp",
        type=str,
        metavar="NAME",
        help="experiment name for logging (creates dedicated directory for the run)",
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=default_log_interval,
        metavar="N",
        help="how many steps between summary writes",
    )
    group.add_argument(
        "--grad-anomaly-detection",
        default=False,
        action="store_true",
        help="enable the autograd anomaly detection (for debugging)",
    )
    group.add_argument(
        "--use-deterministic-algorithms", default=False, action="store_true", help="use only deterministic algorithms"
    )
    group.add_argument(
        "--plot-lr", default=False, action="store_true", help="plot learning rate and exit (skip training)"
    )
    group.add_argument("--no-summary", default=False, action="store_true", help="don't print model summary")
    group.add_argument(
        "--non-interactive",
        default=False,
        action="store_true",
        help="force non-interactive mode (disables progress bars)",
    )
    group.add_argument(
        "--seed", type=int, help="set random seed for better reproducibility (affects torch, numpy and random)"
    )
    group.add_argument("--cpu", default=False, action="store_true", help="use cpu (mostly for testing)")
    if fake_data is True:
        group.add_argument(
            "--use-fake-data",
            default=False,
            action="store_true",
            help="use fake data instead of real dataset (like torchvision.datasets.FakeData)",
        )


def add_training_data_args(parser: argparse.ArgumentParser, unsupervised: bool = False) -> None:
    group = parser.add_argument_group("Training data parameters", description="WebDataset")
    group.add_argument("--wds", default=False, action="store_true", help="use webdataset for training")
    group.add_argument("--wds-info", type=str, metavar="FILE", help="wds info file path")
    group.add_argument("--wds-cache-dir", type=str, metavar="DIR", help="webdataset cache directory")
    group.add_argument("--wds-train-size", type=int, metavar="N", help="size of the wds training set")
    if unsupervised is False:
        group.add_argument("--wds-class-file", type=str, metavar="FILE", help="class list file")
        group.add_argument("--wds-val-size", type=int, metavar="N", help="size of the wds validation set")
        group.add_argument(
            "--wds-training-split", type=str, default="training", metavar="NAME", help="wds dataset train split"
        )
        group.add_argument(
            "--wds-val-split", type=str, default="validation", metavar="NAME", help="wds dataset validation split"
        )
    else:
        group.add_argument(
            "--wds-split", type=str, default="training", metavar="NAME", help="wds dataset split to load"
        )

    group.add_argument(
        "--wds-extra-shuffle",
        default=False,
        action="store_true",
        help=(
            "enable cross-worker batch shuffling after batching. Provides maximum sample diversity but incurs a "
            "notable performance penalty. Use with caution"
        ),
    )

    group = parser.add_argument_group(description="Directory")
    if unsupervised is False:
        group.add_argument(
            "--hierarchical",
            default=False,
            action="store_true",
            help="use hierarchical directory structure for labels (e.g., 'dir1/subdir2' -> 'dir1_subdir2' label)",
        )
        group.add_argument(
            "--data-path",
            type=str,
            default=str(settings.TRAINING_DATA_PATH),
            metavar="DIR",
            help="training directory path",
        )
        group.add_argument(
            "--val-path",
            type=str,
            default=str(settings.VALIDATION_DATA_PATH),
            metavar="DIR",
            help="validation directory path",
        )
    else:
        group.add_argument(
            "--data-path", nargs="*", default=[], help="training directories paths (directories and files)"
        )


def add_detection_training_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Training data parameters")
    group.add_argument(
        "--data-path",
        type=str,
        default=str(settings.DETECTION_DATA_PATH),
        metavar="DIR",
        help="training base directory path",
    )
    group.add_argument(
        "--val-path",
        type=str,
        default=str(settings.DETECTION_DATA_PATH),
        metavar="DIR",
        help="validation base directory path",
    )
    group.add_argument(
        "--coco-json-path",
        type=str,
        default=f"{settings.TRAINING_DETECTION_ANNOTATIONS_PATH}_coco.json",
        metavar="FILE",
        help="training COCO json path",
    )
    group.add_argument(
        "--coco-val-json-path",
        type=str,
        default=f"{settings.VALIDATION_DETECTION_ANNOTATIONS_PATH}_coco.json",
        metavar="FILE",
        help="validation COCO json path",
    )
    group.add_argument("--class-file", type=str, metavar="FILE", help="class list file, overrides json categories")
    group.add_argument(
        "--ignore-file", type=str, metavar="FILE", help="ignore list file, list of samples to ignore in training"
    )


# pylint: disable=too-many-branches
def common_args_validation(args: argparse.Namespace) -> None:
    # Some scripts like train_kd do not have a network argument,
    # but if it exists, it's mandatory all around
    if hasattr(args, "network") is True and args.network is None:
        raise ValidationError("--network is required")

    # Training schedule args, shared by all scripts
    if args.stop_epoch is not None and args.stop_epoch > args.epochs:
        raise ValidationError(
            f"--stop-epoch must be smaller than the total number of epochs ({args.epochs}), got {args.stop_epoch}"
        )
    if args.warmup_epochs is not None and args.warmup_iters is not None:
        raise ValidationError("--warmup-epochs cannot be used with --warmup-iters")
    if args.cooldown_epochs is not None and args.cooldown_iters is not None:
        raise ValidationError("--cooldown-epochs cannot be used with --cooldown-iters")

    if hasattr(args, "lr_scheduler_update") is True:
        if args.lr_scheduler_update != "iter" and args.warmup_iters is not None:
            raise ValidationError(
                "--warmup-iters can only be used when --lr-scheduler-update is 'iter', "
                f"but it is set to '{args.lr_scheduler_update}'"
            )
        if args.lr_scheduler_update != "iter" and args.cooldown_iters is not None:
            raise ValidationError(
                "--cooldown-iters can only be used when --lr-scheduler-update is 'iter', "
                f"but it is set to '{args.lr_scheduler_update}'"
            )

    # Compile args, argument dependant
    if hasattr(args, "compile_teacher") is True:
        if args.compile is True and args.compile_teacher is True:
            raise ValidationError("--compile cannot be used with --compile-teacher")
    if hasattr(args, "compile_backbone") is True:
        if args.compile is True and args.compile_backbone is True:
            raise ValidationError("--compile cannot be used with --compile-backbone")

    # Checkpoint args, shared by all scripts
    if args.load_states is True and args.resume_epoch is None:
        raise ValidationError("--load-states requires --resume-epoch to be set")
    if args.load_scheduler is True and args.resume_epoch is None:
        raise ValidationError("--load-scheduler requires --resume-epoch to be set")
    if hasattr(args, "pretrained") is True and args.pretrained is True and args.resume_epoch is not None:
        raise ValidationError("--pretrained cannot be used with --resume-epoch")

    # Data augmentation args have standard and detection version. Apply only to standard
    if hasattr(args, "resize_min_scale") is True:
        if args.resize_min_scale is not None and (args.resize_min_scale <= 0.0 or args.resize_min_scale >= 1.0):
            raise ValidationError(f"--resize-min-scale must be in range of (0, 1.0), got {args.resize_min_scale}")

    # Training data args have a standard and a detection version. The detection version does not have WDS
    if hasattr(args, "wds") is True:
        # Supervised WDS
        if hasattr(args, "wds_class_file") is True and args.wds is True and args.wds_class_file is None:
            raise ValidationError("--wds requires --wds-class-file to be set")

        if hasattr(args, "hierarchical") is True and args.wds is True and args.hierarchical is True:
            raise ValidationError("--wds cannot be used with --hierarchical")

        # WDS with debug args
        if hasattr(args, "use_fake_data") is True and args.use_fake_data is True and args.wds is True:
            raise ValidationError("--use-fake-data cannot be used with --wds")

        # WDS with dataloader args
        if hasattr(args, "ra_sampler") is True and args.wds is True and args.ra_sampler is True:
            raise ValidationError("Repeated Augmentation (--ra-sampler) not supported with WebDataset (--wds)")

        # Unsupervised training data
        if isinstance(args.data_path, list):
            if args.wds is False and len(args.data_path) == 0 and args.use_fake_data is False:
                raise ValidationError("Must provide at least one data source, --data-path or --wds")
            if args.wds is True and len(args.data_path) > 1:
                raise ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")

    # BatchNorm args, shared by some scripts
    if hasattr(args, "freeze_bn") is True and hasattr(args, "sync_bn"):
        if args.freeze_bn is True and args.sync_bn is True:
            raise ValidationError("--freeze-bn cannot be used with --sync-bn")

    # Precision_args, shared by all scripts
    if args.amp is True and args.model_dtype != "float32":
        raise ValidationError("--amp can only be used with --model-dtype float32")
