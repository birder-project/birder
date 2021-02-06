"""
Inception v4, adapted from
https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/inception-v4.py

Paper "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning",
https://arxiv.org/abs/1602.07261
"""

import mxnet as mx

from birder.common.net import bn_convolution
from birder.core.net.base import BaseNet


def stem_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    x = bn_convolution(data, num_filter=32, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    x = bn_convolution(x, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(0, 0))
    x = bn_convolution(x, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    pooling = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")
    branch = bn_convolution(x, num_filter=96, kernel=(3, 3), stride=(2, 2), pad=(0, 0))
    x = mx.sym.concat(pooling, branch, dim=1)

    branch_1 = bn_convolution(x, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_1 = bn_convolution(branch_1, num_filter=96, kernel=(3, 3), stride=(1, 1), pad=(0, 0))
    branch_2 = bn_convolution(x, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_2 = bn_convolution(branch_2, num_filter=64, kernel=(7, 1), stride=(1, 1), pad=(3, 0))
    branch_2 = bn_convolution(branch_2, num_filter=64, kernel=(1, 7), stride=(1, 1), pad=(0, 3))
    branch_2 = bn_convolution(branch_2, num_filter=96, kernel=(3, 3), stride=(1, 1), pad=(0, 0))
    x = mx.sym.concat(branch_1, branch_2, dim=1)

    pooling = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")
    branch = bn_convolution(x, num_filter=192, kernel=(3, 3), stride=(2, 2), pad=(0, 0))
    x = mx.sym.concat(pooling, branch, dim=1)

    return x


def inception_a_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # 1x1 branch
    branch_1x1 = bn_convolution(data, num_filter=96, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # 3x3 branch
    branch_3x3 = bn_convolution(data, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3x3 = bn_convolution(branch_3x3, num_filter=96, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    # Double 3x3 branch
    branch_dbl_3x3 = bn_convolution(data, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_dbl_3x3 = bn_convolution(branch_dbl_3x3, num_filter=96, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    branch_dbl_3x3 = bn_convolution(branch_dbl_3x3, num_filter=96, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    # Pool branch
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type="avg")
    pooling = bn_convolution(pooling, num_filter=96, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # Concat
    x = mx.sym.concat(branch_1x1, branch_3x3, branch_dbl_3x3, pooling, dim=1)

    return x


def inception_a_reduction_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # Branch 1
    branch_1 = bn_convolution(data, num_filter=384, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # Branch 2
    branch_2 = bn_convolution(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_2 = bn_convolution(branch_2, num_filter=224, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    branch_2 = bn_convolution(branch_2, num_filter=256, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # Pool branch
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    # Concat
    x = mx.sym.concat(branch_1, branch_2, pooling, dim=1)

    return x


def inception_b_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # 1x1 branch
    branch_1x1 = bn_convolution(data, num_filter=384, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # 7x7 branch
    branch_7x7 = bn_convolution(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_7x7 = bn_convolution(branch_7x7, num_filter=224, kernel=(1, 7), stride=(1, 1), pad=(0, 3))
    branch_7x7 = bn_convolution(branch_7x7, num_filter=256, kernel=(7, 1), stride=(1, 1), pad=(3, 0))

    # Double 7x7 branch
    branch_dbl_7x7 = bn_convolution(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_dbl_7x7 = bn_convolution(branch_dbl_7x7, num_filter=192, kernel=(7, 1), stride=(1, 1), pad=(3, 0))
    branch_dbl_7x7 = bn_convolution(branch_dbl_7x7, num_filter=224, kernel=(1, 7), stride=(1, 1), pad=(0, 3))
    branch_dbl_7x7 = bn_convolution(branch_dbl_7x7, num_filter=224, kernel=(7, 1), stride=(1, 1), pad=(3, 0))
    branch_dbl_7x7 = bn_convolution(branch_dbl_7x7, num_filter=256, kernel=(1, 7), stride=(1, 1), pad=(0, 3))

    # Pool branch
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type="avg")
    pooling = bn_convolution(pooling, num_filter=128, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # Concat
    x = mx.sym.concat(branch_1x1, branch_7x7, branch_dbl_7x7, pooling, dim=1)

    return x


def inception_b_reduction_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # 3x3 branch
    branch_3x3 = bn_convolution(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3x3 = bn_convolution(branch_3x3, num_filter=192, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # 7x7 branch
    branch_7x7 = bn_convolution(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_7x7 = bn_convolution(branch_7x7, num_filter=256, kernel=(1, 7), stride=(1, 1), pad=(0, 3))
    branch_7x7 = bn_convolution(branch_7x7, num_filter=320, kernel=(7, 1), stride=(1, 1), pad=(3, 0))
    branch_7x7 = bn_convolution(branch_7x7, num_filter=320, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # Pool branch
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    # Concat
    x = mx.sym.concat(branch_3x3, branch_7x7, pooling, dim=1)

    return x


def inception_c_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # 1x1 branch
    branch_1x1 = bn_convolution(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # 3x3 branch
    branch_3x3 = bn_convolution(data, num_filter=384, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3x3a = bn_convolution(branch_3x3, num_filter=256, kernel=(1, 3), stride=(1, 1), pad=(0, 1))
    branch_3x3b = bn_convolution(branch_3x3, num_filter=256, kernel=(3, 1), stride=(1, 1), pad=(1, 0))

    # Double 3x3 branch
    branch_dbl_3x3 = bn_convolution(data, num_filter=384, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_dbl_3x3 = bn_convolution(branch_dbl_3x3, num_filter=448, kernel=(1, 3), stride=(1, 1), pad=(0, 1))
    branch_dbl_3x3 = bn_convolution(branch_dbl_3x3, num_filter=512, kernel=(3, 1), stride=(1, 1), pad=(1, 0))
    branch_dbl_3x3a = bn_convolution(branch_dbl_3x3, num_filter=256, kernel=(1, 3), stride=(1, 1), pad=(0, 1))
    branch_dbl_3x3b = bn_convolution(branch_dbl_3x3, num_filter=256, kernel=(3, 1), stride=(1, 1), pad=(1, 0))

    # Pool branch
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type="avg")
    pooling = bn_convolution(pooling, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # Concat
    x = mx.sym.concat(branch_1x1, branch_3x3a, branch_3x3b, branch_dbl_3x3a, branch_dbl_3x3b, pooling, dim=1)

    return x


def inception_v4(num_classes: int) -> mx.sym.Symbol:
    data = mx.sym.Variable(name="data")

    x = stem_block(data)

    x = inception_a_block(x)
    x = inception_a_block(x)
    x = inception_a_block(x)
    x = inception_a_block(x)
    x = inception_a_reduction_block(x)

    x = inception_b_block(x)
    x = inception_b_block(x)
    x = inception_b_block(x)
    x = inception_b_block(x)
    x = inception_b_block(x)
    x = inception_b_block(x)
    x = inception_b_block(x)
    x = inception_b_reduction_block(x)

    x = inception_c_block(x)
    x = inception_c_block(x)
    x = inception_c_block(x)

    # Classification block
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.Dropout(data=x, p=0.2)
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


# pylint: disable=invalid-name
class Inception_v4(BaseNet):
    default_size = 299

    def get_symbol(self) -> mx.sym.Symbol:
        return inception_v4(num_classes=self.num_classes)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
