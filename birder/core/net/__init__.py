import importlib

from birder.conf.settings import NET_MODULE
from birder.core.net import densenet
from birder.core.net import inception_resnet_v2
from birder.core.net import inception_v3
from birder.core.net import inception_v4
from birder.core.net import mobilenet_v1
from birder.core.net import mobilenet_v2
from birder.core.net import resnet_v2
from birder.core.net import resnext
from birder.core.net import shufflenet_v1
from birder.core.net import shufflenet_v2
from birder.core.net import squeezenet
from birder.core.net import squeezenext
from birder.core.net import vgg
from birder.core.net import vgg_reduced
from birder.core.net import xception
from birder.core.net.base import BaseNet

if NET_MODULE:
    importlib.import_module(NET_MODULE)

__all__ = [
    "BaseNet",
    "densenet",
    "inception_resnet_v2",
    "inception_v3",
    "inception_v4",
    "mobilenet_v1",
    "mobilenet_v2",
    "resnet_v2",
    "resnext",
    "shufflenet_v1",
    "shufflenet_v2",
    "squeezenet",
    "squeezenext",
    "vgg",
    "vgg_reduced",
    "xception",
]
