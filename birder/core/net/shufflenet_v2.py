"""
ShuffleNet v2, adapted from
https://github.com/Tveek/mxnet-shufflenet/blob/master/model/shufflenet_v2.py

Paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design",
https://arxiv.org/abs/1807.11164
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
    data: mx.sym.Symbol, in_channels: int, out_channels: int, dw_conv_stride: int
) -> mx.sym.Symbol:
    branch_channels = out_channels // 2

    if dw_conv_stride == 1:
        branch1 = mx.sym.slice_axis(data, axis=1, begin=0, end=in_channels // 2)
        branch2 = mx.sym.slice_axis(data, axis=1, begin=in_channels // 2, end=in_channels)

    else:
        branch1 = data
        branch2 = data

        branch1 = mx.sym.Convolution(
            data=branch1,
            num_filter=in_channels,
            kernel=(3, 3),
            stride=(dw_conv_stride, dw_conv_stride),
            pad=(1, 1),
            num_group=in_channels,
            no_bias=True,
        )
        branch1 = mx.sym.BatchNorm(data=branch1, fix_gamma=True, eps=2e-5, momentum=0.9)
        branch1 = mx.sym.Convolution(
            data=branch1, num_filter=branch_channels, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
        )
        branch1 = mx.sym.BatchNorm(data=branch1, fix_gamma=True, eps=2e-5, momentum=0.9)
        branch1 = mx.sym.Activation(data=branch1, act_type="relu")

    branch2 = mx.sym.Convolution(
        data=branch2, num_filter=branch_channels, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
    )
    branch2 = mx.sym.BatchNorm(data=branch2, fix_gamma=True, eps=2e-5, momentum=0.9)
    branch2 = mx.sym.Activation(data=branch2, act_type="relu")

    branch2 = mx.sym.Convolution(
        data=branch2,
        num_filter=branch_channels,
        kernel=(3, 3),
        stride=(dw_conv_stride, dw_conv_stride),
        pad=(1, 1),
        num_group=branch_channels,
        no_bias=True,
    )
    branch2 = mx.sym.BatchNorm(data=branch2, fix_gamma=True, eps=2e-5, momentum=0.9)
    branch2 = mx.sym.Convolution(
        data=branch2, num_filter=branch_channels, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
    )
    branch2 = mx.sym.BatchNorm(data=branch2, fix_gamma=True, eps=2e-5, momentum=0.9)
    branch2 = mx.sym.Activation(data=branch2, act_type="relu")

    x = mx.sym.concat(branch1, branch2, dim=1)
    x = _channel_shuffle(x, groups=2)

    return x


def shufflenet_v2(num_classes: int, stage_repeats: List[int], out_channels: List[int]) -> mx.sym.Symbol:
    assert len(stage_repeats) == 3
    assert len(stage_repeats) + 2 == len(out_channels)

    # Entry
    data = mx.sym.Variable(name="data")
    x = mx.sym.Convolution(data=data, num_filter=24, kernel=(3, 3), stride=(2, 2), pad=(1, 1), no_bias=True)
    x = mx.sym.BatchNorm(data=x, fix_gamma=True, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max")

    # Generate stages
    for i, _ in enumerate(stage_repeats):
        x = shuffle_unit(
            data=x, in_channels=out_channels[i], out_channels=out_channels[i + 1], dw_conv_stride=2
        )
        for _ in range(stage_repeats[i]):
            x = shuffle_unit(
                data=x, in_channels=out_channels[i + 1], out_channels=out_channels[i + 1], dw_conv_stride=1
            )

    # Classification block
    x = mx.sym.Convolution(
        data=x, num_filter=out_channels[-1], kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=True, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


# pylint: disable=invalid-name
class ShuffleNet_v2(BaseNet):
    def get_symbol(self) -> mx.sym.Symbol:
        multiplier = self.num_layers
        if multiplier == 0.5:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 48, 96, 192, 1024]

        elif multiplier == 1:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 116, 232, 464, 1024]

        elif multiplier == 1.5:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 176, 352, 704, 1024]

        elif multiplier == 2:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 244, 488, 976, 2048]

        else:
            raise ValueError(f"multiplier = {multiplier} not supported")

        return shufflenet_v2(self.num_classes, stage_repeats, out_channels)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
