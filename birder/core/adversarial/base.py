import matplotlib.pyplot as plt
import mxnet as mx

_ADVERSARIAL_REGISTERED_CLASSES = []


class Adversarial:
    def __init_subclass__(cls):
        super().__init_subclass__()
        _ADVERSARIAL_REGISTERED_CLASSES.append(cls)

    def __call__(self, model: mx.module.Module, img: mx.nd.NDArray, label: int, size: int) -> mx.nd.NDArray:
        raise NotImplementedError

    def visualize(self, img: mx.nd.NDArray, adv_img: mx.nd.NDArray) -> None:
        img = img.asnumpy()
        adv_img = adv_img.asnumpy()

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img)
        ax.axis("off")

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(adv_img)
        ax.axis("off")

        plt.show()
