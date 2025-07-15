import torch
from torch.optim.lr_scheduler import LRScheduler


class CooldownLR(LRScheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, total_steps: int, eps: float = 1e-8, last_epoch: int = -1
    ) -> None:
        self.total_steps = total_steps
        self.eps = eps
        self.last_last_epoch = last_epoch
        self._first_call = True
        self._base_lr_updated = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self._first_call is True:
            # First call is just initialization
            self._first_call = False
        elif self._base_lr_updated is False:
            # In case this is part of a SequentialLR or similar, adjust to the current learning rate
            # instead of the learning rate defined at init
            self.base_lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]
            self._base_lr_updated = True
            if self.last_epoch == self.last_last_epoch:
                # Handle the step(0) crazy-ness of SequentialLR
                self.last_epoch += 1

        self.last_last_epoch = self.last_epoch
        if self.last_epoch > self.total_steps:
            return [0.0 for _ in self.base_lrs]

        decay_factor = 1.0 - (1.0 - self.eps) * (self.last_epoch / self.total_steps)
        return [base_lr * decay_factor for base_lr in self.base_lrs]
