import cv2
import matplotlib.pyplot as plt
import numpy as np

_AUGMENTATION_REGISTERED_CLASSES = []
_POST_AUGMENTATION_REGISTERED_CLASSES = []


class BaseAugmentation:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def visualize(self, img: np.ndarray) -> None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(self(img))
        plt.axis("off")
        plt.title(type(self).__name__)
        plt.show()


class Augmentation(BaseAugmentation):
    def __init_subclass__(cls):
        super().__init_subclass__()
        _AUGMENTATION_REGISTERED_CLASSES.append(cls)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PostAugmentation(BaseAugmentation):
    priority = 0

    def __init_subclass__(cls):
        super().__init_subclass__()
        _POST_AUGMENTATION_REGISTERED_CLASSES.append(cls)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError
