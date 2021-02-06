"""
Xception, adapted from
https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py

Paper "Xception: Deep Learning with Depthwise Separable Convolutions", https://arxiv.org/abs/1610.02357
"""

from typing import Tuple

import mxnet as mx

from birder.common.net import bn_convolution
from birder.core.net.base import BaseNet


def separable_conv(
    data: mx.sym.Symbol,
    num_filter_a: int,
    num_filter_b: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    pad: Tuple[int, int],
    no_bias: bool = True,
) -> mx.sym.Symbol:
    x = mx.sym.Convolution(
        data=data,
        num_filter=num_filter_a,
        kernel=kernel,
        stride=stride,
        pad=pad,
        num_group=num_filter_a,
        no_bias=no_bias,
    )
    x = mx.sym.Convolution(
        data=x,
        num_filter=num_filter_b,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        num_group=1,
        no_bias=no_bias,
    )

    return x


def xception(num_classes: int) -> mx.sym.Symbol:
    data = mx.sym.Variable(name="data")

    # Entry
    x = bn_convolution(data, num_filter=32, kernel=(3, 3), stride=(2, 2), pad=(0, 0))
    x = bn_convolution(x, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(0, 0))

    residual = mx.sym.Convolution(
        data=x,
        num_filter=128,
        kernel=(1, 1),
        stride=(2, 2),
        pad=(0, 0),
        no_bias=True,
    )
    residual = mx.sym.BatchNorm(data=residual, fix_gamma=False, eps=2e-5, momentum=0.9)

    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=64, num_filter_b=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=128, num_filter_b=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max")
    x = x + residual

    residual = mx.sym.Convolution(
        data=x,
        num_filter=256,
        kernel=(1, 1),
        stride=(2, 2),
        pad=(0, 0),
        no_bias=True,
    )
    residual = mx.sym.BatchNorm(data=residual, fix_gamma=False, eps=2e-5, momentum=0.9)

    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=128, num_filter_b=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=256, num_filter_b=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max")
    x = x + residual

    residual = mx.sym.Convolution(
        data=x,
        num_filter=728,
        kernel=(1, 1),
        stride=(2, 2),
        pad=(0, 0),
        no_bias=True,
    )
    residual = mx.sym.BatchNorm(data=residual, fix_gamma=False, eps=2e-5, momentum=0.9)

    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=256, num_filter_b=728, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=728, num_filter_b=728, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max")
    x = x + residual

    # Middle
    for _ in range(8):
        residual = x

        x = mx.sym.Activation(data=x, act_type="relu")
        x = separable_conv(x, num_filter_a=728, num_filter_b=728, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
        x = mx.sym.Activation(data=x, act_type="relu")
        x = separable_conv(x, num_filter_a=728, num_filter_b=728, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
        x = mx.sym.Activation(data=x, act_type="relu")
        x = separable_conv(x, num_filter_a=728, num_filter_b=728, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)

        x = x + residual

    # Exit
    residual = mx.sym.Convolution(
        data=x,
        num_filter=1024,
        kernel=(1, 1),
        stride=(2, 2),
        pad=(0, 0),
        no_bias=True,
    )
    residual = mx.sym.BatchNorm(data=residual, fix_gamma=False, eps=2e-5, momentum=0.9)

    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=728, num_filter_b=728, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=728, num_filter_b=1024, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max")
    x = x + residual

    x = separable_conv(x, num_filter_a=1024, num_filter_b=1536, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = separable_conv(x, num_filter_a=1536, num_filter_b=2048, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")

    # Classification block
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


class Xception(BaseNet):
    default_size = 299

    def get_symbol(self) -> mx.sym.Symbol:
        return xception(num_classes=self.num_classes)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
