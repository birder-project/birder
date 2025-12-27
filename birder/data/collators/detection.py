import math
import random
from typing import Any

import torch
from torchvision.transforms import v2

BATCH_MULTISCALE_SIZES = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)


def collate_fn(batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    return tuple(zip(*batch))


def batch_images(images: list[torch.Tensor], size_divisible: int) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """
    Batch list of image tensors of different sizes into a single batch.
    Pad with zeros all images to the shape of the largest image in the list.
    """

    size_list = [list(img.shape) for img in images]
    max_size = size_list[0]
    for sublist in size_list[1:]:
        for index, item in enumerate(sublist):
            max_size[index] = max(max_size[index], item)

    image_sizes = [img.shape[-2:] for img in images]
    stride = float(size_divisible)
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

    batch_shape = [len(images)] + max_size
    (B, _, H, W) = batch_shape
    batched_imgs = images[0].new_full(batch_shape, 0)
    masks = images[0].new_full((B, H, W), 1).to(torch.bool)
    for img, pad_img, mask in zip(images, batched_imgs, masks):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        mask[: img.shape[1], : img.shape[2]] = False

    return (batched_imgs, masks, image_sizes)


class DetectionCollator:
    def __init__(self, input_offset: int, size_divisible: int = 32) -> None:
        self.offset = input_offset
        self.size_divisible = size_divisible

    def __call__(self, batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
        # In case that the "transforms" returning a tuple, flatten it
        # if isinstance(batch[0][self.offset], tuple):
        #     batch = [(*sample[: self.offset], *sample[self.offset], *sample[self.offset + 1 :]) for sample in batch]

        data = collate_fn(batch)
        images: list[torch.Tensor] = data[self.offset]
        assert images[0].ndim == 3

        (batched_imgs, masks, size_list) = batch_images(images, self.size_divisible)

        return data[: self.offset] + (batched_imgs,) + data[self.offset + 1 :] + (masks, size_list)


class BatchRandomResizeCollator(DetectionCollator):
    def __init__(self, input_offset: int, size: tuple[int, int], size_divisible: int = 32) -> None:
        super().__init__(input_offset, size_divisible=size_divisible)
        if size is None:
            raise ValueError("size must be provided for batch multiscale")

        max_side = max(size)
        sizes = [side for side in BATCH_MULTISCALE_SIZES if side <= max_side]
        if len(sizes) == 0:
            sizes = [max_side]

        self.sizes = [int(side) for side in sizes]

    def __call__(self, batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
        data = list(collate_fn(batch))
        images: list[torch.Tensor] = data[self.offset]
        targets: list[dict[str, Any]] = data[self.offset + 1]

        target_size = random.choice(self.sizes)
        resize = v2.Resize((target_size, target_size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)

        resized_images = []
        resized_targets = []
        for image, target in zip(images, targets):
            (image, target) = resize(image, target)
            resized_images.append(image)
            resized_targets.append(target)

        data[self.offset] = resized_images
        data[self.offset + 1] = resized_targets

        (batched_imgs, masks, size_list) = batch_images(resized_images, self.size_divisible)

        return tuple(data[: self.offset]) + (batched_imgs,) + tuple(data[self.offset + 1 :]) + (masks, size_list)


# inputs, targets
training_collate_fn = DetectionCollator(0)

# file_paths, inputs, targets
inference_collate_fn = DetectionCollator(1)
