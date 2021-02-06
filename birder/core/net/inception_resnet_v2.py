"""
Inception-ResNet v2, adapted from
https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/inception-resnet-v2.py

Paper "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning",
https://arxiv.org/abs/1602.07261
"""

import mxnet as mx

from birder.common.net import bn_conv_bias
from birder.core.net.base import BaseNet


def stem_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # Branch 1
    branch_1 = bn_conv_bias(data, num_filter=96, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # Branch 2
    branch_2 = bn_conv_bias(data, num_filter=48, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_2 = bn_conv_bias(branch_2, num_filter=64, kernel=(5, 5), stride=(1, 1), pad=(2, 2))

    # Branch 3
    branch_3 = bn_conv_bias(data, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3 = bn_conv_bias(branch_3, num_filter=96, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    branch_3 = bn_conv_bias(branch_3, num_filter=96, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    # Pool branch
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type="avg")
    pooling = bn_conv_bias(pooling, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # Concat
    x = mx.sym.concat(branch_1, branch_2, branch_3, pooling, dim=1)

    return x


def inception_a_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # 1x1 branch
    branch_1x1 = bn_conv_bias(data, num_filter=32, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # 3x3 branch
    branch_3x3 = bn_conv_bias(data, num_filter=32, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3x3 = bn_conv_bias(branch_3x3, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    # Double 3x3 branch
    branch_dbl_3x3 = bn_conv_bias(data, num_filter=32, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_dbl_3x3 = bn_conv_bias(branch_dbl_3x3, num_filter=48, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    branch_dbl_3x3 = bn_conv_bias(branch_dbl_3x3, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    # Concat
    x = mx.sym.concat(branch_1x1, branch_3x3, branch_dbl_3x3, dim=1)

    # Residual scaling
    x = bn_conv_bias(x, num_filter=320, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = data + x * 0.17

    return x


def inception_a_reduction_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # Branch 1
    branch_1 = bn_conv_bias(data, num_filter=384, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # Branch 2
    branch_2 = bn_conv_bias(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_2 = bn_conv_bias(branch_2, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    branch_2 = bn_conv_bias(branch_2, num_filter=384, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # Pool branch
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    # Concat
    x = mx.sym.concat(branch_1, branch_2, pooling, dim=1)

    return x


def inception_b_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # 1x1 branch
    branch_1x1 = bn_conv_bias(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # 7x7 branch
    branch_7x7 = bn_conv_bias(data, num_filter=128, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_7x7 = bn_conv_bias(branch_7x7, num_filter=160, kernel=(1, 7), stride=(1, 1), pad=(0, 3))
    branch_7x7 = bn_conv_bias(branch_7x7, num_filter=192, kernel=(7, 1), stride=(1, 1), pad=(3, 0))

    # Concat
    x = mx.sym.concat(branch_1x1, branch_7x7, dim=1)

    # Residual scaling
    x = bn_conv_bias(x, num_filter=1088, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = data + x * 0.1

    return x


def inception_b_reduction_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # Branch 1
    branch_1 = bn_conv_bias(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_1 = bn_conv_bias(branch_1, num_filter=384, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # Branch 2
    branch_2 = bn_conv_bias(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_2 = bn_conv_bias(branch_2, num_filter=288, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # Branch 3
    branch_3 = bn_conv_bias(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3 = bn_conv_bias(branch_3, num_filter=288, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    branch_3 = bn_conv_bias(branch_3, num_filter=320, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    # Pool branch
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    # Concat
    x = mx.sym.concat(branch_1, branch_2, branch_3, pooling, dim=1)

    return x


def inception_c_block(data: mx.sym.Symbol) -> mx.sym.Symbol:
    # 1x1 branch
    branch_1x1 = bn_conv_bias(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # 3x3 branch
    branch_3x3 = bn_conv_bias(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3x3 = bn_conv_bias(branch_3x3, num_filter=224, kernel=(1, 3), stride=(1, 1), pad=(0, 1))
    branch_3x3 = bn_conv_bias(branch_3x3, num_filter=256, kernel=(3, 1), stride=(1, 1), pad=(1, 0))

    # Concat
    x = mx.sym.concat(branch_1x1, branch_3x3, dim=1)

    # Residual scaling
    x = bn_conv_bias(x, num_filter=2080, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = data + x * 0.2

    return x


def inception_resnet_v2(num_classes: int) -> mx.sym.Symbol:
    data = mx.sym.Variable(name="data")

    x = bn_conv_bias(data, num_filter=32, kernel=(3, 3), stride=(2, 2), pad=(0, 0))

    x = bn_conv_bias(x, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(0, 0))
    x = bn_conv_bias(x, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    x = bn_conv_bias(x, num_filter=80, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = bn_conv_bias(x, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(0, 0))
    x = mx.sym.Pooling(data=x, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max")

    x = stem_block(x)

    # Most implementations use 10 repetitions
    for _ in range(5):
        x = inception_a_block(x)

    x = inception_a_reduction_block(x)

    # Most implementations use 20 repetitions
    for _ in range(10):
        x = inception_b_block(x)

    x = inception_b_reduction_block(x)

    # Most implementations use 9 repetitions
    for _ in range(4):
        x = inception_c_block(x)

    # Modified C block
    block_input = x
    branch_1x1 = bn_conv_bias(block_input, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    branch_3x3 = bn_conv_bias(block_input, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3x3 = bn_conv_bias(branch_3x3, num_filter=224, kernel=(1, 3), stride=(1, 1), pad=(0, 1))
    branch_3x3 = bn_conv_bias(branch_3x3, num_filter=256, kernel=(3, 1), stride=(1, 1), pad=(1, 0))

    x = mx.sym.concat(branch_1x1, branch_3x3, dim=1)
    x = mx.sym.Convolution(data=x, num_filter=2080, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=False)
    x = block_input + x
    x = bn_conv_bias(x, num_filter=1536, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # Classification block
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.Dropout(data=x, p=0.2)
    x = mx.sym.FullyConnected(data=x, num_hidden=num_classes)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


# pylint: disable=invalid-name
class Inception_ResNet_v2(BaseNet):
    default_size = 299

    def get_symbol(self) -> mx.sym.Symbol:
        return inception_resnet_v2(num_classes=self.num_classes)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )
