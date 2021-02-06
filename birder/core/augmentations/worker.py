import logging
import os
import random

import cv2

from birder.conf import settings
from birder.core.augmentations.base import _AUGMENTATION_REGISTERED_CLASSES
from birder.core.augmentations.base import _POST_AUGMENTATION_REGISTERED_CLASSES


def worker(queue, visualize: bool) -> None:
    augmentations = []
    for aug_cls in _AUGMENTATION_REGISTERED_CLASSES:
        if aug_cls.__name__ not in settings.AUG_EXCLUDE:
            augmentations.append(aug_cls())

    post_augmentations = []
    for post_aug_cls in sorted(_POST_AUGMENTATION_REGISTERED_CLASSES, key=lambda x: x.priority, reverse=True):
        if post_aug_cls.__name__ not in settings.POST_AUG_EXCLUDE:
            post_augmentations.append(post_aug_cls())

    while True:
        deq = queue.get()
        if deq is None:
            break

        (aug_count, image_path) = deq
        image_name = os.path.basename(image_path)
        image_name = f"aug_{image_name}"
        aug_path = os.path.join(os.path.dirname(image_path), image_name)

        aug_function = random.choice(augmentations)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = aug_function(img)
        for post_aug_function in post_augmentations:
            img = post_aug_function(img)

        if visualize is True:
            aug_function.visualize(img)

        else:
            jpeg_quality = random.randint(65, 80)
            cv2.imwrite(aug_path, img, (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality))

        if aug_count % 1000 == 999:
            logging.info(f"Augmented image no. {aug_count + 1}...")
