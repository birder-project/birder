import math
import random
from typing import Optional

import numpy as np
import torch


# Unused, keeping as a reference
def _mask_token_omission(
    x: torch.Tensor, mask_ratio: float, kept_mask_ratio: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a 1D mask to the input tensor using the MAE (Masked Autoencoder) style masking.

    Parameters
    ----------
    x
        Tensor of shape (N, L, D), where N is the batch size, L is the sequence length, and D is the feature dimension.
    mask_ratio
        The ratio of the sequence length to be masked. This value should be between 0 and 1.
    kept_mask_ratio
        The ratio of the masked tokens to be kept. If None, it defaults to the value of mask_ratio.
        This value should be between 0 and mask_ratio.

    Returns
    -------
    A tuple containing four elements:
    - The masked input tensor of shape (N, len_keep, D), where len_keep is the length of the sequence after masking.
    - The binary mask tensor of shape (N, L), where 0 indicates kept tokens and 1 indicates masked tokens.
    - The indices of kept tokens.
    - The indices to restore the original order of the sequence after masking.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(2, 10, 5)  # Example input tensor
    >>> mask_ratio = 0.5
    >>> (x_masked, mask, ids_keep, ids_restore) = _mask_token_omission(x, mask_ratio)
    >>> print(x_masked.size())  # Should print torch.Size([2, 5, 5])
    >>> print(mask.size())  # Should print torch.Size([2, 10])
    >>> print(ids_restore.size())  # Should print torch.Size([2, 10])
    """

    if kept_mask_ratio is None:
        kept_mask_ratio = mask_ratio

    # Masking: length -> length * mask_ratio
    # Perform per-sample random masking by per-sample shuffling.
    # Per-sample shuffling is done by argsort random noise.
    (N, L, D) = x.size()  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    len_masked = int(L * (mask_ratio - kept_mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # Noise in [0, 1]

    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # Generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, : len_keep + len_masked] = 0

    # Un-shuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return (x_masked, mask, ids_keep, ids_restore)


def mask_tensor(
    x: torch.Tensor,
    mask: torch.Tensor,
    channels_last: bool = False,
    patch_factor: int = 1,
    mask_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if channels_last is False:
        x = x.permute(0, 2, 3, 1)

    (B, H, W, _) = x.size()

    shaped_mask = mask.reshape(-1, H // patch_factor, W // patch_factor)
    shaped_mask = shaped_mask.repeat_interleave(patch_factor, axis=1).repeat_interleave(patch_factor, axis=2)
    shaped_mask = shaped_mask.unsqueeze(3).type_as(x)

    if mask_token is not None:
        mask_tokens = mask_token.expand(B, H, W, -1)
        x_masked = x * (1.0 - shaped_mask) + (mask_tokens * shaped_mask)
    else:
        x_masked = x * (1.0 - shaped_mask)

    if channels_last is False:
        x_masked = x_masked.permute(0, 3, 1, 2)

    return x_masked


def uniform_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    kept_mask_ratio: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kept_mask_ratio is None:
        kept_mask_ratio = mask_ratio

    # Masking: length -> length * mask_ratio
    # Perform per-sample random masking by per-sample shuffling.
    # Per-sample shuffling is done by argsort random noise.
    len_keep = int(seq_len * (1 - mask_ratio))
    len_masked = int(seq_len * (mask_ratio - kept_mask_ratio))

    noise = torch.rand(batch_size, seq_len, device=device)  # Noise in [0, 1]

    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    # Generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, seq_len], device=device)
    mask[:, : len_keep + len_masked] = 0

    # Un-shuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # assert mask.ndim == 2

    return (mask, ids_keep, ids_restore)


def get_ids_keep(mask: torch.Tensor) -> torch.Tensor:
    B = mask.size(0)
    return (1 - mask).nonzero(as_tuple=True)[1].reshape(B, -1)


def get_random_masked_indices(mask: torch.Tensor, n: int) -> torch.Tensor:
    B = mask.size(0)
    num_masked = mask.count_nonzero().item() // B
    mask_indices_abs = mask.nonzero(as_tuple=True)[1].reshape(B, -1)
    randperm = torch.argsort(torch.rand(B, num_masked, device=mask.device))[:, :n]

    return torch.gather(mask_indices_abs, index=randperm, dim=1)


def mask_from_indices(indices: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Indices return as 1's
    """

    B = indices.size(0)
    row_indices = torch.arange(B, device=indices.device).unsqueeze(1).expand_as(indices)

    mask = torch.zeros([B, seq_len], device=indices.device)
    mask[row_indices.flatten(), indices.flatten()] = 1

    return mask


class Masking:
    def __call__(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError


class UniformMasking(Masking):
    def __init__(self, input_size: tuple[int, int], mask_ratio: float, device: Optional[torch.device] = None) -> None:
        self.seq_len = input_size[0] * input_size[1]
        self.mask_ratio = mask_ratio
        self.device = device

    def __call__(self, batch_size: int) -> torch.Tensor:
        return uniform_mask(batch_size, self.seq_len, self.mask_ratio, self.device)[0]


class BlockMasking(Masking):
    # Adapted from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/masking.py

    def __init__(
        self,
        input_size: tuple[int, int],
        min_num_patches: int,
        max_num_patches: int,
        min_aspect: float,
        max_aspect: float,
    ) -> None:
        self.height = input_size[0]
        self.width = input_size[1]

        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches

        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def _mask(self, mask: torch.Tensor, max_mask_patches: int) -> int:
        # 0 is keep, 1 is remove
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()

                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break

        return delta

    def __call__(self, batch_size: int) -> torch.Tensor:
        num_masking_patches = random.randint(self.min_num_patches, self.max_num_patches)
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(*self.get_shape())
            mask_count = 0
            while mask_count < num_masking_patches:
                max_mask_patches = num_masking_patches - mask_count
                max_mask_patches = min(max_mask_patches, self.max_num_patches)

                delta = self._mask(mask, max_mask_patches)
                if delta == 0:
                    break

                mask_count += delta

            masks.append(mask.flatten())

        return torch.stack(masks, dim=0)


class RollBlockMasking(Masking):
    # Adapted from: https://github.com/facebookresearch/capi/blob/main/data.py

    def __init__(
        self,
        input_size: tuple[int, int],
        num_masking_patches: int,
        min_aspect: float = 0.5,
        max_aspect: float = 2.0,
    ):
        self.height = input_size[0]
        self.width = input_size[1]
        self.num_masking_patches = num_masking_patches
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __call__(self, batch_size: int) -> torch.Tensor:
        masks = []
        for _ in range(batch_size):
            if self.num_masking_patches == 0:
                masks.append(torch.zeros(self.height * self.width))
                continue
            if self.num_masking_patches == self.height * self.width:
                masks.append(torch.ones(self.height * self.width))
                continue

            # Sample aspect ratio, not too large or too small for image
            min_lar = max(self.log_aspect_ratio[0], np.log(self.num_masking_patches / (self.width**2)))
            max_lar = min(
                self.log_aspect_ratio[1],
                np.log(self.height**2 / (self.num_masking_patches + 1e-5)),
            )
            aspect_ratio = math.exp(random.uniform(min_lar, max_lar))

            # Use ceil so mask is >= num_masking_patches
            h = int(np.ceil(math.sqrt(self.num_masking_patches * aspect_ratio)))
            w = int(np.ceil(math.sqrt(self.num_masking_patches / aspect_ratio)))
            top = random.randint(0, self.height - h)
            left = random.randint(0, self.width - w)
            b_mask = np.zeros((self.height, self.width))
            b_mask[top : top + h, left : left + w] = 1

            # truncate ids to get exactly num_masking_patches
            ids = np.where(b_mask.flatten())[0][: self.num_masking_patches]
            mask = np.zeros((self.height, self.width)).flatten()
            mask[ids] = 1
            mask_2d = mask.reshape((self.height, self.width))

            # Roll
            shift_x = random.randint(0, mask_2d.shape[0] - 1)
            shift_y = random.randint(0, mask_2d.shape[1] - 1)
            mask = np.roll(mask_2d, (shift_x, shift_y), (0, 1))
            masks.append(torch.from_numpy(mask.flatten()))

        return torch.stack(masks, dim=0)


class InverseRollBlockMasking(RollBlockMasking):
    def __init__(
        self,
        input_size: tuple[int, int],
        num_masking_patches: int,
        min_aspect: float = 0.5,
        max_aspect: float = 2.0,
    ):
        num_masking_patches = input_size[0] * input_size[1] - num_masking_patches
        super().__init__(input_size, num_masking_patches, min_aspect, max_aspect)

    def __call__(self, batch_size: int) -> torch.Tensor:
        return 1 - super().__call__(batch_size)
