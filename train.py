"""
Main training module, the idea is to be roughly compatible with Gluon CV
train_imagenet.py script.

To monitor using tensorboard run:
    tensorboard serve --logdir logs/
"""

import argparse
import logging
import os
import time
from typing import Any
from typing import Dict
from typing import Optional

import mxnet as mx
from mxboard import SummaryWriter

from birder.common import util
from birder.common.net import create_model
from birder.common.net import replace_top
from birder.common.preprocess import DEFAULT_RGB
from birder.common.preprocess import IMAGENET_RGB
from birder.conf import settings
from birder.core.iterators import ImageRecordIterLabelSmoothing
from birder.core.iterators import ImageRecordIterMixup
from birder.core.iterators import ImageRecordIterOneHot
from birder.core.metrics import OneHotAccuracy
from birder.core.metrics import OneHotCrossEntropy
from birder.core.metrics import OneHotTopKAccuracy
from birder.core.net.base import get_net_class
from birder.core.net.base import net_list


class BoardMetricWriter:
    """
    MXBoard callback class to be used as batch_end_callback and eval_end_callback.
    This callback functor logs all defined metrics by their name to the defined
    log directory in tensorboard format. Network graph is written upon initialization.
    """

    def __init__(
        self, model: mx.module.Module, suffix: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.suffix = f"{suffix}-{int(time.time())}"
        logging.info(f"Recording run '{self.suffix}'")

        log_path = os.path.join(settings.TRAINING_LOGS_DIR, suffix)

        self.summary_writer = SummaryWriter(
            logdir=log_path, filename_suffix=suffix, flush_secs=15, verbose=False
        )
        self.global_step = 0
        self.summary_writer.add_graph(model.symbol)
        if metadata is not None:
            util.write_logfile("", suffix, metadata)

    def __call__(self, param: mx.model.BatchEndParam) -> None:
        """
        In order to distinguish between training and validation callbacks we look for timing operations,
        as the training time is monitored (using tic-toc timer) and validation is not.
        """

        if "tic" in param.locals:
            for metric in param.eval_metric.metrics:
                self.summary_writer.add_scalar(
                    tag=metric.get()[0], value={self.suffix: metric.get()[1]}, global_step=self.global_step
                )

            self.global_step += 1

        else:
            for metric in param.eval_metric.metrics:
                self.summary_writer.add_scalar(
                    tag=metric.get()[0],
                    value={self.suffix + "_val": metric.get()[1]},
                    global_step=self.global_step,
                )

    def close(self) -> None:
        self.summary_writer.close()


def get_num_samples(rec_path: str) -> int:
    records = mx.recordio.MXIndexedRecordIO(rec_path[:-3] + "idx", rec_path, "r")
    num_samples = len(records.keys)
    records.close()

    return num_samples


def get_train_iterator(
    rec_path: str,
    size: int,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    smooth_alpha: float,
    mixup_beta: float,
    rgb_values: Dict[str, float],
) -> mx.io.DataIter:
    kwargs = {
        "path_imgrec": rec_path,
        "label_width": 1,
        "data_shape": (3, size, size),
        "preprocess_threads": num_workers,
        "verbose": False,
        "shuffle": True,
        "round_batch": True,
        "random_resized_crop": True,
        "dtype": "float32",
        "max_aspect_ratio": 4.0 / 3.0,
        "min_aspect_ratio": 3.0 / 4.0,
        "max_random_area": 1.0,
        "min_random_area": 0.7,
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.3,
        "pca_noise": 0.1,
        "fill_value": 127,
        "inter_method": 10,  # Random interpolation
        "rand_mirror": True,
        **rgb_values,
    }

    if mixup_beta > 0:
        return ImageRecordIterMixup(
            num_classes=num_classes,
            batch_size=batch_size,
            smoothing_alpha=smooth_alpha,
            mixup_beta=mixup_beta,
            **kwargs,
        )

    if smooth_alpha > 0:
        return ImageRecordIterLabelSmoothing(
            num_classes=num_classes, batch_size=batch_size, smoothing_alpha=smooth_alpha, **kwargs
        )

    return mx.io.ImageRecordIter(batch_size=batch_size, **kwargs)


def get_val_iterator(
    rec_path: str,
    size: int,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    smooth_alpha: float,
    mixup_beta: float,
    rgb_values: Dict[str, float],
) -> mx.io.DataIter:
    kwargs = {
        "path_imgrec": rec_path,
        "label_width": 1,
        "data_shape": (3, size, size),
        "preprocess_threads": num_workers,
        "verbose": False,
        "shuffle": False,
        "dtype": "float32",
        "round_batch": False,
        "resize": size,
        "inter_method": 2,
        **rgb_values,
    }

    if smooth_alpha > 0 or mixup_beta > 0:
        return ImageRecordIterOneHot(num_classes=num_classes, batch_size=batch_size, **kwargs)

    return mx.io.ImageRecordIter(batch_size=batch_size, **kwargs)


# pylint: disable=too-many-locals
def train(num_outputs: int, args: argparse.Namespace) -> None:
    # Initialize training variables
    begin_epoch = 0
    num_samples = get_num_samples(args.data_path)

    if args.rgb is True:
        rgb_values = util.read_rgb()

    elif args.rgb_imagenet is True:
        rgb_values = IMAGENET_RGB

    else:
        rgb_values = DEFAULT_RGB

    if args.transfer is True:
        # Transfer learning - replace the top of the network
        assert args.size is not None, "Must specify size when using transfer"

        # Load iterators
        train_iter = get_train_iterator(
            args.data_path,
            args.size,
            args.batch_size,
            args.num_workers,
            num_outputs,
            args.smooth_alpha,
            args.mixup_beta,
            rgb_values,
        )
        val_iter = get_val_iterator(
            args.val_path,
            args.size,
            args.batch_size,
            args.num_workers,
            num_outputs,
            args.smooth_alpha,
            args.mixup_beta,
            rgb_values,
        )

        # Create module
        network_name = args.network
        model_path = util.get_model_path(network_name)
        (net_symbol, arg_params, aux_params) = mx.model.load_checkpoint(model_path, args.transfer_epoch)
        (net_symbol, fixed_layers) = replace_top(
            net_symbol, num_outputs, feature_layer=args.features_layer, dropout=0
        )
        arg_params.pop("fullyconnectedtransfer0", None)
        model = create_model(
            net_symbol,
            arg_params,
            aux_params,
            mx.gpu(),
            (args.size, args.size),
            for_training=True,
            batch_size=args.batch_size,
            fixed_param_names=fixed_layers,
        )

        # Update network name and path (after loading)
        network_name = f"{network_name}_{num_outputs}"
        model_path = util.get_model_path(network_name)

    elif args.resume_epoch is not None:
        # Resume training
        begin_epoch = args.resume_epoch
        network_name = f"{args.network}_{num_outputs}"
        model_path = util.get_model_path(network_name)
        signature = util.read_signature(network_name)
        if args.size is None:
            args.size = util.get_signature_size(signature)

        # Load iterators
        train_iter = get_train_iterator(
            args.data_path,
            args.size,
            args.batch_size,
            args.num_workers,
            num_outputs,
            args.smooth_alpha,
            args.mixup_beta,
            rgb_values,
        )
        val_iter = get_val_iterator(
            args.val_path,
            args.size,
            args.batch_size,
            args.num_workers,
            num_outputs,
            args.smooth_alpha,
            args.mixup_beta,
            rgb_values,
        )

        # Create module
        model = mx.module.Module.load(
            model_path,
            args.resume_epoch,
            load_optimizer_states=args.load_states,
            context=mx.gpu(),
        )
        model.bind(
            data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label, for_training=True
        )

    else:
        # Create new model
        try:
            net_class = get_net_class(args.network, num_outputs, args.num_layers)

        except KeyError:
            logging.error(f"Could not find network named '{args.network}'")
            logging.info(f"Known networks are: {', '.join(net_list())}")
            raise SystemExit(1) from None

        if args.size is None:
            args.size = net_class.default_size

        # Load iterators
        train_iter = get_train_iterator(
            args.data_path,
            args.size,
            args.batch_size,
            args.num_workers,
            num_outputs,
            args.smooth_alpha,
            args.mixup_beta,
            rgb_values,
        )
        val_iter = get_val_iterator(
            args.val_path,
            args.size,
            args.batch_size,
            args.num_workers,
            num_outputs,
            args.smooth_alpha,
            args.mixup_beta,
            rgb_values,
        )

        # Create module
        network_name = net_class.get_name()
        model_path = util.get_model_path(network_name)
        net_symbol = net_class.get_symbol()
        model = mx.module.Module(symbol=net_symbol, context=mx.gpu())
        model.bind(
            data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label, for_training=True
        )

        # Initialize weights
        model.init_params(net_class.get_initializer())

    # Print model summary
    mx.visualization.print_summary(model.symbol, shape={"data": (1, 3, args.size, args.size)})

    # Write signature.json
    util.write_signature(network_name, args.size, rgb_values)

    # Initialize metrics and board metric writer
    if args.smooth_alpha > 0:
        eval_metrics = mx.metric.create(
            [OneHotCrossEntropy(), OneHotAccuracy(), OneHotTopKAccuracy(top_k=settings.TOP_K)]
        )

    else:
        eval_metrics = mx.metric.create(
            [mx.metric.CrossEntropy(), mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=settings.TOP_K)]
        )

    board_metric_writer = BoardMetricWriter(
        model,
        f"{args.log_prefix}{network_name}",
        {
            "batch_size": args.batch_size,
            "begin_epoch": begin_epoch,
            "epochs": args.num_epochs,
            "final_learning_rate": args.final_lr,
            "image_size": args.size,
            "learning_rate": args.lr,
            "load_states": args.load_states,
            "mixup_beta": args.mixup_beta,
            "momentum": args.momentum,
            "num_outputs": num_outputs,
            "rgb_imagenet": args.rgb_imagenet,
            "rgb": args.rgb,
            "smooth_alpha": args.smooth_alpha,
            "training_samples": num_samples,
            "transfer": args.transfer,
            "validation_samples": get_num_samples(args.val_path),
            "warmup_epochs": args.warmup_epochs,
            "weight_decay": args.wd,
        },
    )

    # Set optimizer
    if args.load_states is True:
        total_steps = args.num_epochs * num_samples // args.batch_size
        begin_num_update = begin_epoch * num_samples // args.batch_size

    else:
        total_steps = (args.num_epochs - begin_epoch) * num_samples // args.batch_size
        begin_num_update = 0

    lr_schedule = mx.lr_scheduler.CosineScheduler(
        total_steps,
        base_lr=args.lr,
        final_lr=args.final_lr,
        warmup_steps=args.warmup_epochs * num_samples // args.batch_size,
        warmup_begin_lr=0.001,
        warmup_mode="linear",
    )

    optimizer = mx.optimizer.SGD(
        rescale_grad=1.0 / args.batch_size,
        lr_scheduler=lr_schedule,
        momentum=args.momentum,
        wd=args.wd,
        begin_num_update=begin_num_update,
    )
    model.init_optimizer(optimizer=optimizer, force_init=True)

    # Training
    try:
        model.fit(
            train_data=train_iter,
            eval_data=val_iter,
            eval_metric=eval_metrics,
            batch_end_callback=board_metric_writer,
            eval_end_callback=board_metric_writer,
            epoch_end_callback=mx.callback.module_checkpoint(
                model, model_path, period=args.save_frequency, save_optimizer_states=True
            ),
            begin_epoch=begin_epoch,
            num_epoch=args.num_epochs,
        )

    except KeyboardInterrupt:
        model.save_checkpoint(model_path, 9999, save_optimizer_states=False)
        board_metric_writer.close()
        raise SystemExit(1) from None

    # Save model and close writer
    model.save_checkpoint(model_path, 0, save_optimizer_states=False)
    board_metric_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Train classification model",
        epilog=(
            "Usage examples:"
            "\n"
            "python3 train.py --network resnet_v2 --num-layers 50 --size 224 --batch-size 64"
            "\n"
            "python3 train.py --network inception_v4"
            "\n"
            "python3 train.py --network shufflenet_v2 --num-layers 0.5 --size 224 "
            "--resume-epoch 50 --val-path val_data/data.rec --data-path data/data.rec"
            "\n"
            "python3 train.py --network shufflenet_v2_2_75 --transfer --transfer-epoch 0 --size 224"
            "\n\n"
            "Transfer from imagenet trained model from modelzoo"
            "\n"
            "wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params"
            "\n"
            "wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json"
            "\n"
            "python3 train.py --network resnet_152 --transfer --features-layer flatten0 "
            "--size 224 --lr 0.01 --num-epochs 40"
        ),
        formatter_class=util.ArgumentHelpFormatter,
    )
    parser.add_argument("--network", type=str, required=True, help="the neural network to use")
    parser.add_argument(
        "--num-layers",
        type=float,
        help="number of layers or layer multiplayer, required by most networks",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=settings.NUM_EPOCHS, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="base learning rate")
    parser.add_argument(
        "--final-lr", type=float, default=0.01, help="final learning rate for the cosine schedular"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum value for optimizer")
    parser.add_argument("--wd", type=float, default=0.0001, help="weight decay rate")
    parser.add_argument(
        "--save-frequency", type=int, default=settings.SAVE_FREQUENCY, help="frequency of model saving"
    )
    parser.add_argument("--resume-epoch", type=int, help="epoch to resume training from")
    parser.add_argument(
        "--load-states",
        default=False,
        action="store_true",
        help="whether to load optimizer states when resuming",
    )
    parser.add_argument(
        "--transfer",
        default=False,
        action="store_true",
        help="transfer existing network to fit current classes (replace network top)",
    )
    parser.add_argument(
        "--features-layer", type=str, default="features", help="name of features layer (transfer cutoff)"
    )
    parser.add_argument("--transfer-epoch", type=int, default=0, help="model checkpoint to transfer")
    parser.add_argument(
        "--size", type=int, default=None, help="packed image size (defaults to network recommendation)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="the batch size")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="number of warmup epochs")
    parser.add_argument("--smooth-alpha", type=float, default=0.0, help="label smoothing alpha")
    parser.add_argument(
        "--mixup-beta", type=float, default=0.0, help="beta distribution parameter for mixup sampling"
    )
    parser.add_argument(
        "-j",
        "--num-data-workers",
        dest="num_workers",
        type=int,
        default=4,
        help="number of preprocessing workers",
    )
    parser.add_argument(
        "--rgb",
        default=False,
        action="store_true",
        help="use pre-calculated mean and std rgb values",
    )
    parser.add_argument(
        "--rgb-imagenet",
        default=False,
        action="store_true",
        help="use imagenet mean and std rgb values",
    )
    parser.add_argument("--val-path", type=str, default=settings.VAL_PATH, help="path to validation rec")
    parser.add_argument("--data-path", type=str, default=settings.DATA_PATH, help="rec file path")
    parser.add_argument("--log-prefix", type=str, default="", help="mxboard log prefix")
    args = parser.parse_args()

    assert 0.5 > args.smooth_alpha >= 0, "Smoothing alpha must be in range"

    # Get number of outputs
    num_outputs = len(util.read_synset())

    # Call train
    train(num_outputs, args)


if __name__ == "__main__":
    main()
