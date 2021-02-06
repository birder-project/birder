"""
SqueezeNet v1.1, adapted from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/squeezenet.py

Paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size",
https://arxiv.org/abs/1602.07360
"""

import mxnet as mx

from birder.core.net.base import BaseNet


def fire_module(data: mx.sym.Symbol, squeeze: int, expand: int) -> mx.sym.Symbol:
    x = mx.sym.Convolution(
        data=data, num_filter=squeeze, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
    )
    x = mx.sym.Activation(data=x, act_type="relu")

    left = mx.sym.Convolution(
        data=x, num_filter=expand, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
    )
    left = mx.sym.Activation(data=left, act_type="relu")

    right = mx.sym.Convolution(
        data=x, num_filter=expand, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True
    )
    right = mx.sym.Activation(data=right, act_type="relu")

    x = mx.sym.concat(left, right, dim=1)

    return x


def squeezenet(num_classes: int) -> mx.sym.Symbol:
    data = mx.sym.Variable(name="data")

    x = mx.sym.Convolution(data=data, num_filter=64, kernel=(3, 3), stride=(2, 2), pad=(0, 0), no_bias=True)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    x = fire_module(x, squeeze=16, expand=64)
    x = fire_module(x, squeeze=16, expand=64)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    x = fire_module(x, squeeze=32, expand=128)
    x = fire_module(x, squeeze=32, expand=128)
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    x = fire_module(x, squeeze=48, expand=192)
    x = fire_module(x, squeeze=48, expand=192)
    x = fire_module(x, squeeze=64, expand=256)
    x = fire_module(x, squeeze=64, expand=256)

    # Top
    x = mx.sym.Dropout(data=x, p=0.5)
    x = mx.sym.Convolution(
        data=x, num_filter=num_classes, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
    )
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


class SqueezeNet(BaseNet):
    def get_symbol(self) -> mx.sym.Symbol:
        return squeezenet(self.num_classes)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed([".*"], [mx.init.MSRAPrelu()])
