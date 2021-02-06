"""
SqueezeNext 23v5 version.

Paper "SqueezeNext: Hardware-Aware Neural Network Design",  https://arxiv.org/abs/1803.10615
"""

from typing import List

import mxnet as mx

from birder.common.net import bn_convolution
from birder.core.net.base import BaseNet


def sqnxt_unit(data: mx.sym.Symbol, in_channels: int, out_channels: int, stride: int) -> mx.sym.Symbol:
    if stride == 2:
        reduction = 1
        resize_identity = True

    elif in_channels > out_channels:
        reduction = 4
        resize_identity = True

    else:
        reduction = 2
        resize_identity = False

    if resize_identity is True:
        identity = bn_convolution(
            data, num_filter=out_channels, kernel=(1, 1), stride=(stride, stride), pad=(0, 0)
        )

    else:
        identity = data

    x = bn_convolution(
        data, num_filter=in_channels // reduction, kernel=(1, 1), stride=(stride, stride), pad=(0, 0)
    )
    x = bn_convolution(x, num_filter=in_channels // (2 * reduction), kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = bn_convolution(x, num_filter=in_channels // reduction, kernel=(1, 3), stride=(1, 1), pad=(0, 1))
    x = bn_convolution(x, num_filter=in_channels // reduction, kernel=(3, 1), stride=(1, 1), pad=(1, 0))
    x = bn_convolution(x, num_filter=out_channels, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    x = x + identity
    x = mx.sym.Activation(data=x, act_type="relu")

    return x


def squeezenext(
    num_classes: int, width_scale: float, channels_per_layers: List[int], layers_per_stage: List[int]
) -> mx.sym.Symbol:
    assert len(channels_per_layers) == len(layers_per_stage)
    num_layers_per_stage = len(layers_per_stage)

    data = mx.sym.Variable(name="data")

    x = mx.sym.Convolution(
        data=data, num_filter=int(64 * width_scale), kernel=(7, 7), stride=(2, 2), pad=(1, 1), no_bias=True
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    in_channels = int(64 * width_scale)
    for i in range(num_layers_per_stage):
        for j in range(layers_per_stage[i]):
            if j == 0 and i != 0:
                stride = 2

            else:
                stride = 1

            out_channels = int(channels_per_layers[i] * width_scale)
            x = sqnxt_unit(x, in_channels=in_channels, out_channels=out_channels, stride=stride)
            in_channels = out_channels

    # Top
    x = mx.sym.Convolution(
        data=x, num_filter=int(128 * width_scale), kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
    )
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


class SqueezeNext(BaseNet):
    default_size = 227

    def get_symbol(self) -> mx.sym.Symbol:
        width_scale = self.num_layers
        width_scale_values = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        assert width_scale in width_scale_values, f"width scale = {width_scale} not supported"

        channels_per_layers = [32, 64, 128, 256]
        layers_per_stage = [2, 4, 14, 1]

        return squeezenext(self.num_classes, width_scale, channels_per_layers, layers_per_stage)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed([".*"], [mx.init.MSRAPrelu()])
