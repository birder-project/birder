import math

from torch.optim.lr_scheduler import LRScheduler


class ReciprocalSquareRootLR(LRScheduler):
    def get_lr(self) -> list[float]:
        decay_factor = 1 / math.sqrt(self.last_epoch + 1)
        return [base_lr * decay_factor for base_lr in self.base_lrs]
