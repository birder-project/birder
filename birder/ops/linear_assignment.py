import torch
from scipy.optimize import linear_sum_assignment

from birder.kernels.load_kernel import load_linear_assignment

LINEAR_ASSIGNMENT = None


class LinearAssignment:
    """
    Batched linear assignment with optional CUDA kernel acceleration

    The custom kernel is loaded on first instantiation, not at import time.
    Falls back to SciPy's Hungarian algorithm if kernel loading fails.
    """

    def __init__(self) -> None:
        global LINEAR_ASSIGNMENT  # pylint: disable=global-statement
        if LINEAR_ASSIGNMENT is None and not torch.jit.is_tracing() and not torch.jit.is_scripting():
            LINEAR_ASSIGNMENT = load_linear_assignment()

        self.is_available = LINEAR_ASSIGNMENT is not None

    def __call__(self, cost: torch.Tensor, min_batch_for_cuda: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_available is True and cost.is_cuda is True:
            squeeze = False
            if cost.ndim == 2:
                cost = cost.unsqueeze(0)
                squeeze = True
            elif cost.ndim != 3:
                raise ValueError(f"Cost matrix must have shape (B, W, T) or (W, T), got {tuple(cost.shape)}")

            batch_size = cost.size(0)
            if batch_size >= min_batch_for_cuda:
                col4row, row4col = LINEAR_ASSIGNMENT.batch_linear_assignment(cost.contiguous())  # type: ignore
                if squeeze is True:
                    return (col4row[0], row4col[0])

                return (col4row, row4col)

        return batch_linear_assignment(cost)


def batch_linear_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch wrapper for the Hungarian assignment algorithm

    Args:
        cost (Tensor[B, W, T] | Tensor[W, T]): Cost matrix for each batch item.

    Returns:
        col4row (Tensor[B, W] | Tensor[W]): Column assignment for each row.
        row4col (Tensor[B, T] | Tensor[T]): Row assignment for each column.
    """

    squeeze = False
    if cost.ndim == 2:
        cost = cost.unsqueeze(0)
        squeeze = True
    elif cost.ndim != 3:
        raise ValueError(f"Cost matrix must have shape (B, W, T) or (W, T), got {tuple(cost.shape)}")

    batch_size, num_workers, num_tasks = cost.shape
    device = cost.device
    col4row = torch.full((batch_size, num_workers), -1, dtype=torch.int64, device=device)
    row4col = torch.full((batch_size, num_tasks), -1, dtype=torch.int64, device=device)

    # Bulk transfer to CPU once, then solve all items on CPU
    cost_cpu = cost.detach().float().cpu().numpy()
    for idx in range(batch_size):
        cost_matrix = cost_cpu[idx]
        if cost_matrix.size == 0:
            continue

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        row_indices = torch.from_numpy(row_indices).to(device=device, dtype=torch.int64)
        col_indices = torch.from_numpy(col_indices).to(device=device, dtype=torch.int64)

        col4row[idx, row_indices] = col_indices
        row4col[idx, col_indices] = row_indices

    if squeeze is True:
        return (col4row[0], row4col[0])

    return (col4row, row4col)
