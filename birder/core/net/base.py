from typing import List

import mxnet as mx

_REGISTERED_NETS = {}


class BaseNet:
    default_size = 224

    def __init_subclass__(cls):
        super().__init_subclass__()
        _REGISTERED_NETS[cls.__name__.lower()] = cls

    def __init__(self, num_classes: int, num_layers: float):
        self.num_classes = num_classes
        self.num_layers = num_layers

    def get_symbol(self) -> mx.sym.Symbol:
        raise NotImplementedError

    @staticmethod
    def get_initializer() -> mx.initializer.Initializer:
        return mx.initializer.Mixed(
            [".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)]
        )

    def get_name(self) -> str:
        if self.num_layers is not None:
            if int(self.num_layers) == self.num_layers:
                self.num_layers = int(self.num_layers)

            return f"{type(self).__name__.lower()}_{self.num_layers}_{self.num_classes}"

        return f"{type(self).__name__.lower()}_{self.num_classes}"


def get_signature(size: int):
    return {"inputs": [{"data_name": "data", "data_shape": [0, 3, size, size]}]}


def gluon_to_symbol(gluon_model: mx.gluon.nn.HybridBlock) -> mx.sym.Symbol:
    gluon_model.hybridize()

    data = mx.sym.Variable(name="data")
    symbol = gluon_model(data)
    symbol = mx.sym.SoftmaxOutput(data=symbol, name="softmax")

    return symbol


def get_net_class(name: str, num_classes: int, num_layers: float) -> BaseNet:
    return _REGISTERED_NETS[name](num_classes, num_layers)


def net_list() -> List[str]:
    return sorted(list(_REGISTERED_NETS.keys()))
