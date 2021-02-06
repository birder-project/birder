"""
A more "modern" approach to VGG, with batch normalization and without the
double fullyconnected layers.

This is also an example on how to use Gluon API as symbol provider for birder.

Original paper "Very Deep Convolutional Networks for Large-Scale Image Recognition",
https://arxiv.org/abs/1409.1556
"""

from typing import List

import mxnet as mx
from mxnet import gluon

from birder.core.net.base import BaseNet
from birder.core.net.base import gluon_to_symbol


class VggReduced(gluon.nn.HybridBlock):
    def __init__(self, num_classes: int, layers: List[int], filters: List[int], **kwargs):
        super().__init__(**kwargs)

        self.features = gluon.nn.HybridSequential()
        for i, num in enumerate(layers):
            for _ in range(num):
                self.features.add(
                    gluon.nn.Conv2D(filters[i], kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
                )
                self.features.add(gluon.nn.BatchNorm())
                self.features.add(gluon.nn.Activation("relu"))

            self.features.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding=(0, 0)))

        self.features.add(gluon.nn.Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0)))
        self.features.add(gluon.nn.BatchNorm())
        self.features.add(gluon.nn.Activation("relu"))
        self.features.add(gluon.nn.GlobalAvgPool2D())
        self.features.add(gluon.nn.Flatten(prefix="features_"))

        self.output = gluon.nn.Dense(num_classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)

        return x


# pylint: disable=invalid-name
class Vgg_Reduced(BaseNet):
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

        vgg_reduced_gluon = VggReduced(self.num_classes, layers, filters)
        return gluon_to_symbol(vgg_reduced_gluon)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="uniform", factor_type="avg", magnitude=3)]
        )
