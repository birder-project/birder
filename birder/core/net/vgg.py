"""
Paper "Very Deep Convolutional Networks for Large-Scale Image Recognition", https://arxiv.org/abs/1409.1556
"""

from typing import List

import mxnet as mx

from birder.core.net.base import BaseNet


def vgg(num_classes: int, layers: List[int], filters: List[int]) -> mx.sym.Symbol:
    data = mx.sym.Variable(name="data")

    x = data

    for i, num in enumerate(layers):
        for _ in range(num):
            x = mx.sym.Convolution(data=x, num_filter=filters[i], kernel=(3, 3), stride=(1, 1), pad=(1, 1))
            x = mx.sym.Activation(data=x, act_type="relu")

        x = mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max")

    x = mx.sym.Flatten(data=x)
    x = mx.sym.FullyConnected(data=x, num_hidden=4096)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Dropout(data=x, p=0.5)
    x = mx.sym.FullyConnected(data=x, num_hidden=4096)
    x = mx.sym.Activation(data=x, act_type="relu", name="features")
    x = mx.sym.Dropout(data=x, p=0.5)
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


class Vgg(BaseNet):
    def get_symbol(self) -> mx.sym.Symbol:
        num_layers = int(self.num_layers)

        if num_layers == 11:
            layers = [1, 1, 2, 2, 2]
            filters = [64, 128, 256, 512, 512]

        elif num_layers == 13:
            layers = [2, 2, 2, 2, 2]
            filters = [64, 128, 256, 512, 512]

        elif num_layers == 16:
            layers = [2, 2, 3, 3, 3]
            filters = [64, 128, 256, 512, 512]

        elif num_layers == 19:
            layers = [2, 2, 4, 4, 4]
            filters = [64, 128, 256, 512, 512]

        else:
            raise ValueError(f"num_layers = {num_layers} not supported")

        return vgg(self.num_classes, layers, filters)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="uniform", factor_type="avg", magnitude=3)]
        )
