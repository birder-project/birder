"""
ResNeXt, adapted from
https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/resnext.py

Paper "Aggregated Residual Transformations for Deep Neural Networks", https://arxiv.org/abs/1611.05431
"""

from typing import List
from typing import Tuple

import mxnet as mx

from birder.common.net import bn_convolution
from birder.core.net.base import BaseNet


def residual_unit(
    data: mx.sym.Variable, num_filter: int, stride: Tuple[int, int], dim_match: bool, bottle_neck: bool = True
) -> mx.sym.Symbol:
    if bottle_neck is True:
        x = bn_convolution(
            data, num_filter=num_filter // 2, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
        )
        x = bn_convolution(
            x,
            num_filter=num_filter // 2,
            kernel=(3, 3),
            stride=stride,
            pad=(1, 1),
            num_group=32,
            no_bias=True,
        )
        x = mx.sym.Convolution(
            data=x, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
        )
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)

    else:
        x = bn_convolution(
            data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True
        )
        x = mx.sym.Convolution(
            data=x, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True
        )
        x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)

    if dim_match:
        shortcut = data

    else:
        shortcut = mx.sym.Convolution(
            data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), no_bias=True
        )
        shortcut = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)

    x = x + shortcut
    x = mx.sym.Activation(data=x, act_type="relu")

    return x


def resnext(
    units: List[int], filter_list: List[int], num_classes: int, bottle_neck: bool = True
) -> mx.sym.Symbol:
    assert len(units) + 1 == len(filter_list)
    num_unit = len(units)

    data = mx.sym.Variable(name="data")
    x = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9)

    x = bn_convolution(x, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max")

    for i in range(num_unit):
        if i == 0:
            stride = (1, 1)

        else:
            stride = (2, 2)

        x = residual_unit(x, filter_list[i + 1], stride=stride, dim_match=False, bottle_neck=bottle_neck)
        for _ in range(units[i] - 1):
            x = residual_unit(x, filter_list[i + 1], stride=(1, 1), dim_match=True, bottle_neck=bottle_neck)

    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


class ResNeXt(BaseNet):
    def get_symbol(self) -> mx.sym.Symbol:
        num_layers = int(self.num_layers)

        if num_layers == 18:
            bottle_neck = False
            units = [2, 2, 2, 2]
            filter_list = [64, 64, 128, 256, 512]

        elif num_layers == 34:
            bottle_neck = False
            units = [3, 4, 6, 3]
            filter_list = [64, 64, 128, 256, 512]

        elif num_layers == 50:
            bottle_neck = True
            units = [3, 4, 6, 3]
            filter_list = [64, 256, 512, 1024, 2048]

        elif num_layers == 101:
            bottle_neck = True
            units = [3, 4, 23, 3]
            filter_list = [64, 256, 512, 1024, 2048]

        elif num_layers == 152:
            bottle_neck = True
            units = [3, 8, 36, 3]
            filter_list = [64, 256, 512, 1024, 2048]

        elif num_layers == 200:
            bottle_neck = True
            units = [3, 24, 36, 3]
            filter_list = [64, 256, 512, 1024, 2048]

        elif num_layers == 269:
            bottle_neck = True
            units = [3, 30, 48, 8]
            filter_list = [64, 256, 512, 1024, 2048]

        else:
            raise ValueError(f"num_layers = {num_layers} not supported")

        return resnext(
            units=units, filter_list=filter_list, num_classes=self.num_classes, bottle_neck=bottle_neck
        )

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
