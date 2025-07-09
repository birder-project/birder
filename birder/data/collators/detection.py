import math
from typing import Any

import torch


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


# inputs, targets
training_collate_fn = DetectionCollator(0)

# file_paths, inputs, targets
inference_collate_fn = DetectionCollator(1)
