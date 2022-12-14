import warnings
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, step_size: list, gamma: list, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        assert len(step_size) == len(gamma), "len of step_size must be same as the gamma"
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        for step_size, gamma in zip(self.step_size, self.gamma):
            if (self.last_epoch == 0) or (self.last_epoch != step_size):
                lr = [group['lr'] for group in self.optimizer.param_groups]
            else:
                lr = [group['lr'] * gamma for group in self.optimizer.param_groups]
                break
        return lr
