"""
MobileNet v2, adapted from
https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/mobilenetv2.py

Paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks", https://arxiv.org/abs/1801.04381
"""

from typing import List
from typing import Tuple

import mxnet as mx

from birder.common.net import relu6
from birder.core.net.base import BaseNet


def bn_convolution_relu6(
    data: mx.sym.Symbol,
    num_filter: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    pad: Tuple[int, int],
    num_group: int = 1,
) -> mx.sym.Symbol:
    x = mx.sym.Convolution(
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        num_group=num_group,
        no_bias=True,
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=1e-5, momentum=0.9)
    x = relu6(data=x)

    return x


def inverted_residual_unit(
    data: mx.sym.Symbol,
    num_in_filter: int,
    num_filter: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    pad: Tuple[int, int],
    expansion_factor: float,
    shortcut: bool,
) -> mx.sym.Symbol:
    num_expfilter = int(round(num_in_filter * expansion_factor))

    x = bn_convolution_relu6(data, num_filter=num_expfilter, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = bn_convolution_relu6(
        x, num_filter=num_expfilter, kernel=kernel, stride=stride, pad=pad, num_group=num_expfilter
    )
    x = mx.sym.Convolution(
        data=x,
        num_filter=num_filter,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        no_bias=True,
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=1e-5, momentum=0.9)

    if shortcut is True:
        x = x + data

    return x


def mobilenet_v2(
    num_classes: int, multiplier: float, inverted_residual_setting: List[List[int]]
) -> mx.sym.Symbol:
    data = mx.sym.Variable(name="data")

    in_c = int(round(32 * multiplier))
    x = bn_convolution_relu6(data, num_filter=in_c, kernel=(3, 3), stride=(2, 2), pad=(1, 1))

    # pylint: disable=invalid-name
    for t, c, n, s in inverted_residual_setting:
        c = int(round(c * multiplier))
        x = inverted_residual_unit(
            x,
            num_in_filter=in_c,
            num_filter=c,
            kernel=(3, 3),
            stride=(s, s),
            pad=(1, 1),
            expansion_factor=t,
            shortcut=False,
        )

        for _ in range(1, n):
            x = inverted_residual_unit(
                x,
                num_in_filter=c,
                num_filter=c,
                kernel=(3, 3),
                stride=(1, 1),
                pad=(1, 1),
                expansion_factor=t,
                shortcut=True,
            )

        in_c = int(round(c * multiplier))

    x = bn_convolution_relu6(
        x, num_filter=int(round(1280 * max(1.0, multiplier))), kernel=(1, 1), stride=(1, 1), pad=(0, 0)
    )

    # Classification block
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


# pylint: disable=invalid-name
class MobileNet_v2(BaseNet):
    def get_symbol(self) -> mx.sym.Symbol:
        multiplier = self.num_layers
        multiplier_values = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        assert multiplier in multiplier_values, f"alpha = {multiplier} not supported"

        # t - expension factor
        # c - num_filters (channels)
        # n - number of repetitions
        # s - stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        return mobilenet_v2(self.num_classes, multiplier, inverted_residual_setting)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
