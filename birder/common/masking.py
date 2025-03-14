from typing import Optional

import torch


def mask1d(
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
    >>> (x_masked, mask, ids_keep, ids_restore) = mask1d(x, mask_ratio)
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


def mask2d(
    x: torch.Tensor,
    mask_ratio: float,
    kept_mask_ratio: Optional[float] = None,
    channels_last: bool = False,
    patch_factor: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if kept_mask_ratio is None:
        kept_mask_ratio = mask_ratio

    if channels_last is False:
        x = x.permute(0, 2, 3, 1)

    (B, H, W, _) = x.size()

    L = (H // patch_factor) * (W // patch_factor)
    len_keep = int(L * (1 - mask_ratio))
    len_masked = int(L * (mask_ratio - kept_mask_ratio))

    noise = torch.randn(B, L, device=x.device)

    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([B, L], device=x.device)
    mask[:, : len_keep + len_masked] = 0

    # Un-shuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # Reshape mask
    assert len(mask.shape) == 2

    shaped_mask = mask.reshape(-1, H // patch_factor, W // patch_factor)
    shaped_mask = shaped_mask.repeat_interleave(patch_factor, axis=1).repeat_interleave(patch_factor, axis=2)
    shaped_mask = shaped_mask.unsqueeze(3).type_as(x)

    x_masked = x * (1.0 - shaped_mask)

    if channels_last is False:
        x_masked = x_masked.permute(0, 3, 1, 2)
        shaped_mask = shaped_mask.permute(0, 3, 1, 2)

    return (x_masked, mask, shaped_mask, ids_restore)
