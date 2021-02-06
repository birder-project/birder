"""
Custom network for bird classification
"""

import mxnet as mx

from birder.common.net import bn_convolution
from birder.core.net import BaseNet
from birder.core.net.shufflenet_v2 import shuffle_unit


def stem_block(data: mx.sym.Symbol, alpha: float = 1.0) -> mx.sym.Symbol:
    # Branch 1
    branch_1 = bn_convolution(data, num_filter=int(96 * alpha), kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # Branch 2
    branch_2 = bn_convolution(data, num_filter=int(48 * alpha), kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_2 = bn_convolution(branch_2, num_filter=int(64 * alpha), kernel=(5, 5), stride=(1, 1), pad=(2, 2))

    # Branch 3
    branch_3 = bn_convolution(data, num_filter=int(64 * alpha), kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3 = bn_convolution(branch_3, num_filter=int(96 * alpha), kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    branch_3 = bn_convolution(branch_3, num_filter=int(96 * alpha), kernel=(3, 3), stride=(1, 1), pad=(1, 1))

    # Concat
    x = mx.sym.concat(branch_1, branch_2, branch_3, dim=1)

    return x


def conv_block(data: mx.sym.Symbol, alpha: float = 1.0) -> mx.sym.Symbol:
    # 1x1 branch
    branch_1x1 = bn_convolution(data, num_filter=int(256 * alpha), kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    # 3x3 branch
    branch_3x3 = bn_convolution(data, num_filter=int(384 * alpha), kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    branch_3x3a = bn_convolution(
        branch_3x3, num_filter=int(256 * alpha), kernel=(1, 3), stride=(1, 1), pad=(0, 1)
    )
    branch_3x3b = bn_convolution(
        branch_3x3, num_filter=int(256 * alpha), kernel=(3, 1), stride=(1, 1), pad=(1, 0)
    )

    # Double 3x3 branch
    branch_dbl_3x3 = bn_convolution(
        data, num_filter=int(384 * alpha), kernel=(1, 1), stride=(1, 1), pad=(0, 0)
    )
    branch_dbl_3x3 = bn_convolution(
        branch_dbl_3x3, num_filter=int(448 * alpha), kernel=(1, 3), stride=(1, 1), pad=(0, 1)
    )
    branch_dbl_3x3 = bn_convolution(
        branch_dbl_3x3, num_filter=int(512 * alpha), kernel=(3, 1), stride=(1, 1), pad=(1, 0)
    )
    branch_dbl_3x3a = bn_convolution(
        branch_dbl_3x3, num_filter=int(256 * alpha), kernel=(1, 3), stride=(1, 1), pad=(0, 1)
    )
    branch_dbl_3x3b = bn_convolution(
        branch_dbl_3x3, num_filter=int(256 * alpha), kernel=(3, 1), stride=(1, 1), pad=(1, 0)
    )

    # Concat
    x = mx.sym.concat(branch_1x1, branch_3x3a, branch_3x3b, branch_dbl_3x3a, branch_dbl_3x3b, dim=1)

    return x


def birdnet(num_outputs: int) -> mx.sym.Symbol:
    data = mx.sym.Variable(name="data")

    x = bn_convolution(data, num_filter=32, kernel=(3, 3), stride=(2, 2), pad=(1, 1))

    x = bn_convolution(x, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = bn_convolution(x, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max")

    x = bn_convolution(x, num_filter=80, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = bn_convolution(x, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    x = mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max")

    # Separate high and low frequencies (https://arxiv.org/abs/1904.05049)
    lf_branch = mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="avg")
    lf_branch = stem_block(lf_branch, alpha=0.875)
    lf_branch = shuffle_unit(lf_branch, 224, 224, 1)
    lf_branch = mx.sym.UpSampling(lf_branch, scale=2, sample_type="nearest")

    hf_branch = stem_block(x, alpha=0.125)
    hf_branch = bn_convolution(hf_branch, num_filter=224, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    hf_branch = shuffle_unit(hf_branch, 224, 224, 1)
    hf_branch = shuffle_unit(hf_branch, 224, 224, 1)

    x = lf_branch + hf_branch
    x = mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max")

    # Separate high and low frequencies (https://arxiv.org/abs/1904.05049)
    lf_branch = mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="avg")
    lf_branch = conv_block(lf_branch, alpha=0.875)
    lf_branch = conv_block(lf_branch, alpha=0.875)
    lf_branch = mx.sym.UpSampling(lf_branch, scale=2, sample_type="nearest")

    hf_branch = conv_block(x, alpha=0.125)
    hf_branch = conv_block(hf_branch, alpha=0.125)
    hf_branch = conv_block(hf_branch, alpha=0.125)
    hf_branch = conv_block(hf_branch, alpha=0.125)
    hf_branch = bn_convolution(hf_branch, num_filter=1120, kernel=(1, 1), stride=(1, 1), pad=(0, 0))

    x = lf_branch + hf_branch
    x = mx.sym.Pooling(data=x, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max")

    x = bn_convolution(x, num_filter=1024, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    x = shuffle_unit(x, 1024, 1024, 1)
    x = shuffle_unit(x, 1024, 1024, 1)
    x = shuffle_unit(x, 1024, 1024, 1)

    # Classification block
    x = mx.sym.Pooling(data=x, global_pool=True, pool_type="avg")
    x = mx.sym.Flatten(data=x, name="features")
    x = mx.sym.Dropout(data=x, p=0.2)
    x = mx.sym.FullyConnected(data=x, num_hidden=num_outputs)
    x = mx.sym.SoftmaxOutput(data=x, name="softmax")

    return x


class Birdnet(BaseNet):
    default_size = 256

    def get_symbol(self) -> mx.sym.Symbol:
        return birdnet(self.num_classes)

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="uniform", factor_type="avg", magnitude=3)]
        )
