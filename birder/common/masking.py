import math
import random
from collections.abc import Callable
from typing import Optional

import numpy as np
import torch


# Unused, keeping as a reference
def _mask_token_omission(
    x: torch.Tensor, mask_ratio: float, kept_mask_ratio: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a 1D mask to the input tensor using MAE-style masking

    Parameters
    ----------
    x
        Tensor of shape (N, L, D), where N is the batch size, L is the sequence length and D is the feature dimension.
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
    >>> x_masked, mask, ids_keep, ids_restore = _mask_token_omission(x, mask_ratio)
    >>> print(x_masked.size())  # Should print torch.Size([2, 5, 5])
    >>> print(mask.size())  # Should print torch.Size([2, 10])
    >>> print(ids_restore.size())  # Should print torch.Size([2, 10])
    """

    if kept_mask_ratio is None:
        kept_mask_ratio = mask_ratio

    # Masking: length -> length * mask_ratio
    # Perform per-sample random masking by per-sample shuffling.
    # Per-sample shuffling is done by argsort random noise.
    N, L, D = x.size()  # batch, length, dim
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

    B, H, W, _ = x.size()

    shaped_mask = mask.reshape(B, H // patch_factor, W // patch_factor)
    shaped_mask = shaped_mask.repeat_interleave(patch_factor, dim=1).repeat_interleave(patch_factor, dim=2)
    shaped_mask = shaped_mask.unsqueeze(3).type_as(x)

    if mask_token is not None:
        expanded_mask_token = mask_token.expand(B, H, W, -1)
        x_masked = x * (1.0 - shaped_mask) + (expanded_mask_token * shaped_mask)
    else:
        x_masked = x * (1.0 - shaped_mask)

    if channels_last is False:
        x_masked = x_masked.permute(0, 3, 1, 2)

    return x_masked


def mask_tokens(x: torch.Tensor, mask: torch.Tensor, mask_token: Optional[torch.Tensor] = None) -> torch.Tensor:
    shaped_mask = mask.unsqueeze(-1).type_as(x)
    if mask_token is not None:
        expanded_mask_token = mask_token.reshape(1, 1, -1).expand_as(x)
        return x * (1.0 - shaped_mask) + (expanded_mask_token * shaped_mask)

    return x * (1.0 - shaped_mask)


def uniform_mask(
    batch_size: int,
    h: int,
    w: int,
    mask_ratio: float,
    kept_mask_ratio: Optional[float] = None,
    min_mask_size: int = 1,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a uniform random mask for a batch of sequences

    Performs per-sample random masking by shuffling random noise. The mask can optionally
    keep a portion of the masked tokens (useful for certain training strategies).

    Parameters
    ----------
    batch_size
        Number of samples in the batch.
    h
        Height of the 2D grid (number of patches along height).
    w
        Width of the 2D grid (number of patches along width).
    mask_ratio
        The ratio of the sequence length to be masked. This value should be between 0 and 1.
    kept_mask_ratio
        The ratio of the masked tokens to be kept. If None, it defaults to the value of mask_ratio.
        This value should be between 0 and mask_ratio.
    min_mask_size
        The minimum masking unit size. When greater than 1, masking is performed at a coarser
        granularity where each mask unit covers a block of min_mask_size x min_mask_size patches.
        Both h and w must be divisible by min_mask_size.
    device
        The device on which to create the tensors.

    Returns
    -------
    A tuple containing three elements:
    - The binary mask tensor of shape (batch_size, h * w), where 0 indicates kept tokens and 1 indicates
      masked tokens.
    - The indices of kept tokens of shape (batch_size, len_keep).
    - The indices to restore the original order of the sequence of shape (batch_size, h * w).
    """

    if kept_mask_ratio is None:
        kept_mask_ratio = mask_ratio

    seq_len = h * w
    h_coarse = h // min_mask_size
    w_coarse = w // min_mask_size
    seq_len_coarse = h_coarse * w_coarse

    len_keep_coarse = int(seq_len_coarse * (1 - mask_ratio))
    len_masked_coarse = int(seq_len_coarse * (mask_ratio - kept_mask_ratio))

    noise = torch.rand(batch_size, seq_len_coarse, device=device)
    ids_shuffle_coarse = torch.argsort(noise, dim=1)
    ids_restore_coarse = torch.argsort(ids_shuffle_coarse, dim=1)

    mask_coarse = torch.ones([batch_size, seq_len_coarse], device=device)
    mask_coarse[:, : len_keep_coarse + len_masked_coarse] = 0
    mask_coarse = torch.gather(mask_coarse, dim=1, index=ids_restore_coarse)

    if min_mask_size > 1:
        # Expand coarse mask to fine resolution
        mask = (
            mask_coarse.reshape(batch_size, h_coarse, w_coarse)
            .repeat_interleave(min_mask_size, dim=1)
            .repeat_interleave(min_mask_size, dim=2)
            .reshape(batch_size, seq_len)
        )

        # Derive ids_shuffle from mask using expanded noise as tie-breaker
        noise_fine = (
            noise.reshape(batch_size, h_coarse, w_coarse)
            .repeat_interleave(min_mask_size, dim=1)
            .repeat_interleave(min_mask_size, dim=2)
            .reshape(batch_size, seq_len)
        )
        sort_key = mask * 2 + noise_fine  # kept: [0,1), masked: [2,3)
        ids_shuffle = torch.argsort(sort_key, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
    else:
        # Coarse is already fine, no expansion needed
        mask = mask_coarse
        ids_shuffle = ids_shuffle_coarse
        ids_restore = ids_restore_coarse

    len_keep = len_keep_coarse * (min_mask_size**2)
    ids_keep = ids_shuffle[:, :len_keep]

    return (mask, ids_keep, ids_restore)


def fixed_size_block_mask(
    batch_size: int,
    h: int,
    w: int,
    mask_ratio: float,
    block_size: int,
    mask_ratio_adjust: float = 0.0,
    inverse_mask: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate random fixed-size block masks for a batch of patch grids

    Block centers are sampled independently and expanded to square blocks. Overlapping blocks
    are merged, then each mask is randomly adjusted to the exact target number of patches.

    Parameters
    ----------
    batch_size
        Number of masks to generate.
    h
        Height of the patch grid.
    w
        Width of the patch grid.
    mask_ratio
        The ratio of patches to mask. This value should be between 0 and 1.
    block_size
        Side length of each sampled block in patches. This value must be greater than 1.
    mask_ratio_adjust
        Adjustment added to the ratio used to determine the number of sampled blocks. This
        affects the mask distribution but not the final number of masked patches.
    inverse_mask
        Sample blocks for the visible patches before inverting the mask.
    device
        The device on which to create the masks.

    Returns
    -------
    The binary mask tensor of shape (batch_size, h * w), where 0 indicates kept tokens and 1
    indicates masked tokens.
    """

    # Adapted from: https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/data_utils.py

    if mask_ratio < 0.0 or mask_ratio > 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if block_size <= 1:
        raise ValueError("block_size must be greater than 1")

    seq_len = h * w
    block_ratio = mask_ratio
    if inverse_mask is True:
        block_ratio = 1.0 - block_ratio

    num_blocks = int(seq_len * ((block_ratio + mask_ratio_adjust) / block_size**2))
    if num_blocks < 0:
        raise ValueError("mask_ratio_adjust results in a negative number of blocks")

    mask = torch.zeros((batch_size, h, w), device=device)
    block_centers = torch.randint(0, seq_len, size=(batch_size, num_blocks), device=device)
    mask.view(batch_size, -1).scatter_(1, block_centers, 1)
    batch_indices, center_y, center_x = mask.nonzero(as_tuple=True)

    offset = block_size // 2
    for i in range(block_size):
        for j in range(block_size):
            y = (center_y + i - offset).clamp_(min=0, max=h - 1)
            x = (center_x + j - offset).clamp_(min=0, max=w - 1)
            mask[(batch_indices, y, x)] = 1

    mask = mask.reshape(batch_size, -1)
    target_len = int(seq_len * block_ratio)
    for sample_mask in mask:
        current_len = int(sample_mask.sum().item())
        if current_len > target_len:
            indices = torch.multinomial(sample_mask, current_len - target_len, replacement=False)
            sample_mask[indices] = 0
        elif current_len < target_len:
            indices = torch.multinomial(1 - sample_mask, target_len - current_len, replacement=False)
            sample_mask[indices] = 1

    if inverse_mask is True:
        mask = 1 - mask

    return mask


def generate_naflex_masks(
    batch_size: int, grid_sizes: torch.Tensor, mask_generator: Callable[[int, int, int], torch.Tensor]
) -> torch.Tensor:
    """
    Generate and right-pad masks after grouping samples with identical patch grids
    """

    if grid_sizes.ndim != 2 or grid_sizes.size(1) != 2:
        raise ValueError(f"Grid sizes must have shape (batch_size, 2), got {tuple(grid_sizes.size())}")
    if grid_sizes.size(0) != batch_size:
        raise ValueError(f"Grid sizes batch dimension must match batch size {batch_size}, got {grid_sizes.size(0)}")

    seq_lens = grid_sizes.prod(dim=1)
    max_seq_len = int(seq_lens.max().item())
    unique_grid_sizes, group_indices = torch.unique(grid_sizes, dim=0, return_inverse=True)
    masks: Optional[torch.Tensor] = None
    for group_idx, (grid_h, grid_w) in enumerate(unique_grid_sizes.tolist()):
        batch_indices = (group_indices == group_idx).nonzero(as_tuple=True)[0]
        group_masks = mask_generator(batch_indices.numel(), grid_h, grid_w)
        if masks is None:
            masks = group_masks.new_zeros((batch_size, max_seq_len))

        masks[batch_indices.to(device=masks.device), : grid_h * grid_w] = group_masks

    return masks


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
    Return a mask with indices set to 1
    """

    B = indices.size(0)
    row_indices = torch.arange(B, device=indices.device).unsqueeze(1).expand_as(indices)

    mask = torch.zeros([B, seq_len], device=indices.device)
    mask[row_indices.flatten(), indices.flatten()] = 1

    return mask


class Masking:
    def __call__(self, batch_size: int, *, grid_sizes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if grid_sizes is None:
            return self._generate(batch_size)

        return self._generate_naflex(batch_size, grid_sizes)

    def _generate(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError

    def _generate_naflex(self, batch_size: int, grid_sizes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UniformMasking(Masking):
    def __init__(
        self,
        input_size: tuple[int, int],
        mask_ratio: float,
        min_mask_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.h = input_size[0]
        self.w = input_size[1]
        self.mask_ratio = mask_ratio
        self.min_mask_size = min_mask_size
        self.device = device

    def _generate_for_grid(self, batch_size: int, grid_h: int, grid_w: int) -> torch.Tensor:
        padded_h = math.ceil(grid_h / self.min_mask_size) * self.min_mask_size
        padded_w = math.ceil(grid_w / self.min_mask_size) * self.min_mask_size
        mask = uniform_mask(
            batch_size, padded_h, padded_w, self.mask_ratio, min_mask_size=self.min_mask_size, device=self.device
        )[0]
        if (padded_h, padded_w) == (grid_h, grid_w):
            return mask

        return mask.reshape(batch_size, padded_h, padded_w)[:, :grid_h, :grid_w].reshape(batch_size, -1)

    def _generate(self, batch_size: int) -> torch.Tensor:
        return self._generate_for_grid(batch_size, self.h, self.w)

    def _generate_naflex(self, batch_size: int, grid_sizes: torch.Tensor) -> torch.Tensor:
        return generate_naflex_masks(batch_size, grid_sizes, self._generate_for_grid)


class BlockMasking(Masking):
    """
    Generate random block masks for a batch of patch grids

    Samples a total number of masked patches between min_masking_patches and max_num_patches,
    then fills that budget with one or more rectangular blocks. min_num_patches controls the
    minimum area of an individual sampled block. For NaFlex grids, these patch counts are
    scaled by the grid area relative to input_size.

    Parameters
    ----------
    input_size
        The patch grid size as (height, width).
    min_num_patches
        The minimum number of patches in an individual sampled block.
    max_num_patches
        The maximum total number of masked patches per sample.
        This also caps the size of an individual sampled block.
    min_aspect
        The minimum aspect ratio for sampled blocks.
    max_aspect
        The maximum aspect ratio for sampled blocks.
    min_masking_patches
        The minimum total number of masked patches per sample. If None, defaults to min_num_patches.
    """

    # Adapted from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/masking.py

    def __init__(
        self,
        input_size: tuple[int, int],
        min_num_patches: int,
        max_num_patches: int,
        min_aspect: float,
        max_aspect: float,
        min_masking_patches: Optional[int] = None,
    ) -> None:
        self.height = input_size[0]
        self.width = input_size[1]

        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        if min_masking_patches is None:
            min_masking_patches = min_num_patches

        self.min_masking_patches = min_masking_patches
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def get_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def _mask(self, mask: torch.Tensor, max_mask_patches: int) -> int:
        # 0 is keep, 1 is remove
        height, width = mask.shape
        max_target_area = min(
            max_mask_patches,
            mask.numel(),
            width**2 * self.max_aspect,
            height**2 / self.min_aspect,
        )
        if max_target_area < 1:
            return 0

        min_num_patches = self.min_num_patches * mask.numel() // self.num_patches
        min_target_area = min(max(min_num_patches, 1), max_target_area)
        for _ in range(10):
            target_area = random.uniform(min_target_area, max_target_area)
            min_aspect = max(self.min_aspect, target_area / (width**2))
            max_aspect = min(self.max_aspect, height**2 / target_area)
            aspect_ratio = math.exp(random.uniform(math.log(min_aspect), math.log(max_aspect)))
            h = max(1, int(round(math.sqrt(target_area * aspect_ratio))))
            w = max(1, int(round(math.sqrt(target_area / aspect_ratio))))
            if w <= width and h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                block = mask[top : top + h, left : left + w]
                delta = int((block == 0).sum().item())
                if 0 < delta <= max_mask_patches:
                    block.fill_(1)
                    return delta

        return 0

    def _generate_for_grid(self, batch_size: int, grid_h: int, grid_w: int) -> torch.Tensor:
        grid_num_patches = grid_h * grid_w
        max_num_patches, min_masking_patches = (
            0 if num_patches == 0 else max(1, num_patches * grid_num_patches // self.num_patches)
            for num_patches in (self.max_num_patches, self.min_masking_patches)
        )
        max_num_patches = min(grid_num_patches, max_num_patches)
        min_masking_patches = min(max_num_patches, min_masking_patches)
        num_masking_patches = random.randint(min_masking_patches, max_num_patches)

        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(grid_h, grid_w)
            mask_count = 0
            while mask_count < num_masking_patches:
                max_mask_patches = num_masking_patches - mask_count
                delta = self._mask(mask, max_mask_patches)
                if delta == 0:
                    break

                mask_count += delta

            masks.append(mask.flatten())

        return torch.stack(masks, dim=0)

    def _generate(self, batch_size: int) -> torch.Tensor:
        return self._generate_for_grid(batch_size, self.height, self.width)

    def _generate_naflex(self, batch_size: int, grid_sizes: torch.Tensor) -> torch.Tensor:
        return generate_naflex_masks(batch_size, grid_sizes, self._generate_for_grid)


class FixedSizeBlockMasking(Masking):
    def __init__(
        self,
        input_size: tuple[int, int],
        mask_ratio: float,
        block_size: int,
        mask_ratio_adjust: float = 0.0,
        inverse_mask: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.h = input_size[0]
        self.w = input_size[1]
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.mask_ratio_adjust = mask_ratio_adjust
        self.inverse_mask = inverse_mask
        self.device = device

    def _generate_for_grid(self, batch_size: int, grid_h: int, grid_w: int) -> torch.Tensor:
        return fixed_size_block_mask(
            batch_size,
            grid_h,
            grid_w,
            self.mask_ratio,
            self.block_size,
            mask_ratio_adjust=self.mask_ratio_adjust,
            inverse_mask=self.inverse_mask,
            device=self.device,
        )

    def _generate(self, batch_size: int) -> torch.Tensor:
        return self._generate_for_grid(batch_size, self.h, self.w)

    def _generate_naflex(self, batch_size: int, grid_sizes: torch.Tensor) -> torch.Tensor:
        return generate_naflex_masks(batch_size, grid_sizes, self._generate_for_grid)


class RollBlockMasking(Masking):
    # Adapted from: https://github.com/facebookresearch/capi/blob/main/data.py

    def __init__(
        self, input_size: tuple[int, int], num_masking_patches: int, min_aspect: float = 0.5, max_aspect: float = 2.0
    ) -> None:
        self.height = input_size[0]
        self.width = input_size[1]
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _scale_patch_count(self, patch_count: int, grid_num_patches: int) -> int:
        if patch_count == 0:
            return 0

        return min(grid_num_patches, max(1, patch_count * grid_num_patches // self.num_patches))

    def _generate_block_masks(self, batch_size: int, grid_h: int, grid_w: int, num_block_patches: int) -> torch.Tensor:
        grid_num_patches = grid_h * grid_w
        masks = []
        for _ in range(batch_size):
            if num_block_patches == 0:
                masks.append(torch.zeros(grid_num_patches))
                continue
            if num_block_patches == grid_num_patches:
                masks.append(torch.ones(grid_num_patches))
                continue

            # Sample aspect ratio, not too large or too small for image
            grid_min_lar = math.log(num_block_patches / (grid_w**2))
            grid_max_lar = math.log(grid_h**2 / num_block_patches)
            min_lar = max(self.log_aspect_ratio[0], grid_min_lar)
            max_lar = min(self.log_aspect_ratio[1], grid_max_lar)
            if min_lar <= max_lar:
                aspect_ratio = math.exp(random.uniform(min_lar, max_lar))
            elif self.log_aspect_ratio[0] > grid_max_lar:
                aspect_ratio = math.exp(grid_max_lar)
            else:
                aspect_ratio = math.exp(grid_min_lar)

            # Use ceil so mask is >= num_block_patches
            h = min(grid_h, int(np.ceil(math.sqrt(num_block_patches * aspect_ratio))))
            w = min(grid_w, int(np.ceil(math.sqrt(num_block_patches / aspect_ratio))))
            top = random.randint(0, grid_h - h)
            left = random.randint(0, grid_w - w)
            b_mask = np.zeros((grid_h, grid_w), dtype=np.float32)
            b_mask[top : top + h, left : left + w] = 1

            # Truncate ids to get exactly num_block_patches
            ids = np.where(b_mask.flatten())[0][:num_block_patches]
            mask = np.zeros((grid_h, grid_w), dtype=np.float32).flatten()
            mask[ids] = 1
            mask_2d = mask.reshape((grid_h, grid_w))

            # Roll
            shift_x = random.randint(0, mask_2d.shape[0] - 1)
            shift_y = random.randint(0, mask_2d.shape[1] - 1)
            mask = np.roll(mask_2d, (shift_x, shift_y), (0, 1))
            masks.append(torch.from_numpy(mask.flatten()))

        return torch.stack(masks, dim=0)

    def _generate_for_grid(self, batch_size: int, grid_h: int, grid_w: int) -> torch.Tensor:
        grid_num_patches = grid_h * grid_w
        num_masking_patches = self._scale_patch_count(self.num_masking_patches, grid_num_patches)
        return self._generate_block_masks(batch_size, grid_h, grid_w, num_masking_patches)

    def _generate(self, batch_size: int) -> torch.Tensor:
        return self._generate_for_grid(batch_size, self.height, self.width)

    def _generate_naflex(self, batch_size: int, grid_sizes: torch.Tensor) -> torch.Tensor:
        return generate_naflex_masks(batch_size, grid_sizes, self._generate_for_grid)


class InverseRollBlockMasking(RollBlockMasking):
    def _generate_for_grid(self, batch_size: int, grid_h: int, grid_w: int) -> torch.Tensor:
        grid_num_patches = grid_h * grid_w
        num_masking_patches = self._scale_patch_count(self.num_masking_patches, grid_num_patches)
        num_visible_patches = grid_num_patches - num_masking_patches
        return 1 - self._generate_block_masks(batch_size, grid_h, grid_w, num_visible_patches)
