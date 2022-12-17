import numpy as np
import pdb
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import wandb
from utils import to_img


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer,
                         config, data_loader, valid_data_loader, len_epoch, lr_scheduler)
        self.config = config
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _training_step(self, batch, batch_idx):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        data, _ = batch
        data = data.to(self.device)
        _, decode = self.model(data)
        loss = self.criterion(decode, data)
        self.writer.log("train/loss", loss.item(), on_step=True, on_epoch=True)
        decode_imgs = make_grid(decode, nrow=8, normalize=True)
        target_imgs = make_grid(data, nrow=8, normalize=True)
        self.writer.log_img("train/decode_img", wandb.Image(torch.cat([target_imgs, decode_imgs], dim=-1), caption="Training"))
        if self.metric_ftns:
            for met in self.metric_ftns:
                self.writer.log(f"train/{met.__name__}", met(decode, data))

        return {
            "loss": loss,
        }

    def _validation_step(self, batch, batch_idx):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        data, _ = batch
        data = data.to(self.device)
        _, decode = self.model(data)
        loss = self.criterion(decode, data)
        self.writer.log("valid/loss", loss.item())
        decode_imgs = make_grid(decode, nrow=8, normalize=True)
        target_imgs = make_grid(data, nrow=8, normalize=True)
        self.writer.log_img("valid/decode_img", wandb.Image(torch.cat([target_imgs, decode_imgs], dim=-1), caption="Validation"))
        if self.metric_ftns:
            for met in self.metric_ftns:
                self.writer.log(f"valid/{met.__name__}", met(decode, data))

        return {
            "loss": loss,
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
