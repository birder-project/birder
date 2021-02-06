"""
ShuffleNet v1, adapted from
https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV1/network.py

Paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices",
https://arxiv.org/abs/1707.01083
"""

from typing import List

import mxnet as mx

from birder.core.net.base import BaseNet


def _channel_shuffle(data: mx.sym.Symbol, groups: int) -> mx.sym.Symbol:
    x = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
    x = mx.sym.swapaxes(x, 1, 2)
    x = mx.sym.reshape(x, shape=(0, -3, -2))

    return x


def shuffle_unit(
    data: mx.sym.Symbol, in_channels: int, out_channels: int, groups: int, grouped_conv: bool
) -> mx.sym.Symbol:
    assert in_channels <= out_channels

    bottleneck_channels = out_channels // 4

    if in_channels == out_channels:
        dw_conv_stride = 1

    elif in_channels < out_channels:
        dw_conv_stride = 2
        out_channels -= in_channels

    if grouped_conv is True:
        first_groups = groups

    else:
        first_groups = 1

    x = mx.sym.Convolution(
        data=data,
        num_filter=bottleneck_channels,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        num_group=first_groups,
        no_bias=True,
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=True, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")

    if grouped_conv is True:
        x = _channel_shuffle(x, groups)

    x = mx.sym.Convolution(
        data=x,
        num_filter=bottleneck_channels,
        kernel=(3, 3),
        stride=(dw_conv_stride, dw_conv_stride),
        pad=(1, 1),
        num_group=bottleneck_channels,
        no_bias=True,
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=True, eps=2e-5, momentum=0.9)
    x = mx.sym.Convolution(
        data=x,
        num_filter=out_channels,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        num_group=groups,
        no_bias=True,
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=True, eps=2e-5, momentum=0.9)

    if dw_conv_stride == 1:
        x = x + data

    else:
        data = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="avg")
        x = mx.sym.concat(x, data, dim=1)

    x = mx.sym.Activation(data=x, act_type="relu")

    return x


def shufflenet_v1(
    num_classes: int, stage_repeats: List[int], out_channels: List[int], groups: int
) -> mx.sym.Symbol:
    assert len(stage_repeats) == 3
    assert len(stage_repeats) + 1 == len(out_channels)

    # Entry
    data = mx.sym.Variable(name="data")
    x = mx.sym.Convolution(data=data, num_filter=24, kernel=(3, 3), stride=(2, 2), pad=(1, 1), no_bias=True)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max")

    # Generate stages
    for i, _ in enumerate(stage_repeats):
        if i == 0:
            grouped_conv = False

        else:
            grouped_conv = True

        x = shuffle_unit(
            data=x,
            in_channels=out_channels[i],
            out_channels=out_channels[i + 1],
            groups=groups,
            grouped_conv=grouped_conv,
        )
        for _ in range(stage_repeats[i]):
            x = shuffle_unit(
                data=x,
                in_channels=out_channels[i + 1],
                out_channels=out_channels[i + 1],
                groups=groups,
                grouped_conv=True,
            )

    # Classification
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


# pylint: disable=invalid-name
class ShuffleNet_v1(BaseNet):
    def get_symbol(self) -> mx.sym.Symbol:
        groups = int(self.num_layers)

        if groups == 1:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 144, 288, 576]

        elif groups == 2:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 200, 400, 800]

        elif groups == 3:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 240, 480, 960]

        elif groups == 4:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 272, 544, 1088]

        elif groups == 8:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 384, 768, 1536]

        else:
            raise ValueError(f"groups = {groups} not supported")

        return shufflenet_v1(self.num_classes, stage_repeats, out_channels, groups)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
