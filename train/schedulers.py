import math
from torch.optim.lr_scheduler import LambdaLR


class CosineAnnealingWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.num_cycles = num_cycles
        super().__init__(optimizer, lr_lambda=self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_steps:
            return self.max_lr * float(epoch) / float(max(1, self.warmup_steps))
        if epoch > self.training_steps:
            return self.min_lr
        progress = float(epoch - self.warmup_steps) / float(
            max(1, self.training_steps - self.warmup_steps)
        )
        coeff = max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)),
        )
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
