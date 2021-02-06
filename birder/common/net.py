from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import mxnet as mx


def create_model(
    net_symbol: mx.sym.Symbol,
    arg_params: Dict[str, mx.nd.NDArray],
    aux_params: Dict[str, mx.nd.NDArray],
    context: mx.Context,
    input_size: Tuple[int, int],
    for_training: bool,
    inputs_need_grad: bool = False,
    batch_size: int = 1,
    features: bool = False,
    features_layer: str = "features",
    fixed_param_names: Optional[List[str]] = None,
) -> mx.module.Module:
    """
    Create MXNet module from load_checkpoint arguments.

    Parameters
    ----------
    net_symbol
        The symbol configuration of computation network (as returned from load_checkpoint).
    arg_params
        Network weights (as returned from load_checkpoint).
    aux_params
        Network auxiliary states (as returned from load_checkpoint).
    context
        The model context.
    input_size
        The network input size.
    for_training
        Whether the executors should be bind for training.
    inputs_need_grad
        Whether the gradients to the input data need to be computed (passed directly to module bind).
    batch_size
        Defines the data shape for the model.
    features
        Whether or not to set the features layer as an output.
    features_layer
        Name of the features layer.
    fixed_param_names
        Layer names to freeze parameters during training.

    Returns
    -------
    Module
        MXNet bound module, ready for training or inference.
    """

    label_shapes: Optional[List[Tuple[str, List[int]]]] = None
    label_names = None
    if for_training is True:
        label_shapes = [("softmax_label", [batch_size])]
        label_names = ("softmax_label",)

    if features is True:
        features_symbol = net_symbol.get_internals()[f"{features_layer}_output"]
        net_symbol = mx.symbol.Group([net_symbol, features_symbol])

    model = mx.module.Module(
        symbol=net_symbol, context=context, label_names=label_names, fixed_param_names=fixed_param_names
    )
    model.bind(
        for_training=for_training,
        data_shapes=[("data", (batch_size, 3, input_size[0], input_size[1]))],
        label_shapes=label_shapes,
        inputs_need_grad=inputs_need_grad,
    )
    model.set_params(arg_params, aux_params, allow_missing=True, force_init=True, allow_extra=True)

    return model


def replace_top(
    net_symbol: mx.sym.Symbol, num_outputs: int, feature_layer: str = "features", dropout: float = 0
) -> Tuple[mx.sym.Symbol, List[str]]:
    """
    Replace network top with new fully connected top size 'num_outputs' followed by softmax
    Only applicable to networks with fully connected top (unlike squeezenet)
    """

    assert 1.0 > dropout >= 0

    net_symbol = net_symbol.get_internals()[f"{feature_layer}_output"]
    fixed_layers = net_symbol.list_arguments()
    if dropout != 0:
        net_symbol = mx.sym.Dropout(data=net_symbol, p=dropout)

    # Set static name, allow us to drop future weights after multiple transfers
    net_symbol = mx.symbol.FullyConnected(
        data=net_symbol, num_hidden=num_outputs, name="fullyconnectedtransfer0"
    )
    net_symbol = mx.symbol.SoftmaxOutput(data=net_symbol, name="softmax")

    return (net_symbol, fixed_layers)


def bn_convolution(
    data: mx.sym.Symbol,
    num_filter: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    pad: Tuple[int, int],
    num_group: int = 1,
    no_bias: bool = True,
) -> mx.sym.Symbol:
    x = mx.sym.Convolution(
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        num_group=num_group,
        no_bias=no_bias,
    )
    x = mx.sym.BatchNorm(data=x, fix_gamma=False, eps=2e-5, momentum=0.9)
    x = mx.sym.Activation(data=x, act_type="relu")

    return x


def bn_conv_bias(
    data: mx.sym.Symbol,
    num_filter: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    pad: Tuple[int, int],
) -> mx.sym.Symbol:
    return bn_convolution(
        data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=False
    )


def relu6(data: mx.sym.Symbol) -> mx.sym.Symbol:
    return mx.sym.clip(data, 0, 6)
