"""
Densenet, adapted from
https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/densenet.py

Paper "Densely Connected Convolutional Networks", https://arxiv.org/abs/1608.06993
"""

from typing import List

import mxnet as mx

from birder.core.net.base import BaseNet


def dense_block(data: mx.sym.Variable, num_layers: int, growth_rate: int) -> mx.sym.Symbol:
    x = data
    for _ in range(num_layers):
        dense_branch = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
        dense_branch = mx.sym.Activation(data=dense_branch, act_type="relu")
        dense_branch = mx.sym.Convolution(
            data=dense_branch,
            num_filter=4 * growth_rate,
            kernel=(1, 1),
            stride=(1, 1),
            pad=(0, 0),
            no_bias=True,
        )
        dense_branch = mx.sym.BatchNorm(data=dense_branch, fix_gamma=False, eps=2e-5, momentum=0.9)
        dense_branch = mx.sym.Activation(data=dense_branch, act_type="relu")
        dense_branch = mx.sym.Convolution(
            data=dense_branch, num_filter=growth_rate, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True
        )
        x = mx.sym.concat(x, dense_branch, dim=1)

    return x


def transition_block(data: mx.sym.Variable, num_features) -> mx.sym.Symbol:
    x = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Convolution(
        data=x, num_filter=num_features, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True
    )
    x = mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="avg")

    return x


def densenet(
    num_classes: int, growth_rate: int, num_init_features: int, layer_list: List[int]
) -> mx.sym.Symbol:
    data = mx.sym.Variable(name="data")

    x = mx.sym.Convolution(
        data=data, num_filter=num_init_features, kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max")

    # Add dense blocks
    num_features = num_init_features
    for i, num_layers in enumerate(layer_list):
        x = dense_block(x, num_layers, growth_rate)
        num_features = num_features + (num_layers * growth_rate)

        # Last block does not require transition
        if i != len(layer_list) - 1:
            num_features = num_features // 2
            x = transition_block(x, num_features)

    # Classification block
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


class Densenet(BaseNet):
    def get_symbol(self) -> mx.sym.Symbol:
        num_layers = int(self.num_layers)
        if num_layers == 121:
            growth_rate = 32
            num_init_features = 64
            layer_list = [6, 12, 24, 16]

        elif num_layers == 161:
            growth_rate = 48
            num_init_features = 96
            layer_list = [6, 12, 36, 24]

        elif num_layers == 169:
            growth_rate = 32
            num_init_features = 64
            layer_list = [6, 12, 32, 32]

        elif num_layers == 201:
            growth_rate = 32
            num_init_features = 64
            layer_list = [6, 12, 48, 32]

        else:
            raise ValueError(f"num_layers = {num_layers} not supported")

        return densenet(
            num_classes=self.num_classes,
            growth_rate=growth_rate,
            num_init_features=num_init_features,
            layer_list=layer_list,
        )

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
