"""
MobileNet v1, adapted from
https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/mobilenet.py

Paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications",
https://arxiv.org/abs/1704.04861
"""

from typing import Tuple

import mxnet as mx

from birder.common.net import bn_convolution
from birder.core.net.base import BaseNet


def dpw_bn_convolution(data: mx.sym.Symbol, num_filter: int, stride: Tuple[int, int]) -> mx.sym.Symbol:
    x = bn_convolution(
        data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), num_group=num_filter
    )
    x = bn_convolution(x, num_filter=num_filter * stride[0], kernel=(1, 1), stride=stride, pad=(0, 0))

    return x


def mobilenet_v1(num_classes: int, alpha: float) -> mx.sym.Symbol:
    base = int(32 * alpha)

    data = mx.sym.Variable(name="data")

    x = bn_convolution(data, num_filter=base, kernel=(3, 3), stride=(2, 2), pad=(1, 1))

    x = bn_convolution(x, num_filter=base, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=base)
    x = bn_convolution(x, num_filter=base * 2, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = dpw_bn_convolution(x, num_filter=base * 2, stride=(2, 2))

    x = dpw_bn_convolution(x, num_filter=base * 4, stride=(1, 1))
    x = dpw_bn_convolution(x, num_filter=base * 4, stride=(2, 2))

    x = dpw_bn_convolution(x, num_filter=base * 8, stride=(1, 1))
    x = dpw_bn_convolution(x, num_filter=base * 8, stride=(2, 2))

    x = dpw_bn_convolution(x, num_filter=base * 16, stride=(1, 1))
    x = dpw_bn_convolution(x, num_filter=base * 16, stride=(1, 1))
    x = dpw_bn_convolution(x, num_filter=base * 16, stride=(1, 1))
    x = dpw_bn_convolution(x, num_filter=base * 16, stride=(1, 1))
    x = dpw_bn_convolution(x, num_filter=base * 16, stride=(1, 1))
    x = dpw_bn_convolution(x, num_filter=base * 16, stride=(2, 2))

    x = dpw_bn_convolution(x, num_filter=base * 32, stride=(1, 1))

    # Classification block
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


# pylint: disable=invalid-name
class MobileNet_v1(BaseNet):
    def get_symbol(self) -> mx.sym.Symbol:
        alpha = self.num_layers
        alpha_values = [0.25, 0.50, 0.75, 1.0]
        assert alpha in alpha_values, f"alpha = {alpha} not supported"

        return mobilenet_v1(self.num_classes, alpha)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
