from collections import namedtuple

import mxnet as mx

from birder.common.preprocess import deprocess_image
from birder.common.preprocess import preprocess_image
from birder.core.adversarial.base import Adversarial

Batch = namedtuple("Batch", ["data", "label"])


class FGSM(Adversarial):
    def __init__(
        self, rgb_mean: mx.nd.NDArray, rgb_std: mx.nd.NDArray, rgb_scale: float, factor: float = 0.04
    ):
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.rgb_scale = rgb_scale
        self.factor = factor

    def __call__(
        self,
        model: mx.module.Module,
        img: mx.nd.NDArray,
        label: int,
        size: int,
    ) -> mx.nd.NDArray:
        (height, width) = img.shape[:2]
        img = preprocess_image(img, (size, size), self.rgb_mean, self.rgb_std, self.rgb_scale)

        batch = Batch([img], label)
        model.forward(batch, is_train=False)
        model.backward()
        grads = model.get_input_grads()[0]
        perturbation = mx.nd.sign(grads)
        adv_img = img + perturbation * self.factor

        adv_img = deprocess_image(adv_img, self.rgb_mean, self.rgb_std, self.rgb_scale)
        adv_img = mx.image.imresize(adv_img, width, height, interp=4)

        return adv_img
