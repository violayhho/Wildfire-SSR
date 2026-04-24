import torch.optim as optim
 
class WarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, gamma = 5e-3, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return self.base_lrs
 