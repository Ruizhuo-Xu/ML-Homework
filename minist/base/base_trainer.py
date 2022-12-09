import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter, WandbWriter
import os
from utils import inf_loop
from tqdm import tqdm


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns,
                 optimizer, config, data_loader, valid_data_loader, len_epoch):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        if len_epoch is None:
            # epoch-based training + validating
            self.len_epoch = ((len(self.data_loader) + len(self.valid_data_loader))
                                if self.do_validation else len(self.data_loader))
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        # self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.writer = WandbWriter(config, self.logger, cfg_trainer['wandb'])
        self.steps = 0

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, batch_idx, batch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, batch_idx, batch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # self.writer.mode = 'global'
            # self.writer.log('Epoch', epoch)
            with tqdm(total=self.len_epoch, desc="Progress:", leave=False) as pbar:
                self.model.train()
                for batch_idx, batch in enumerate(self.data_loader):
                    self.writer.set_step()
                    self.optimizer.zero_grad()
                    train_result = self._train_epoch(batch_idx, batch)
                    if isinstance(train_result, dict):
                        assert "loss" in train_result.keys(), "return dict must contain 'loss'"
                        loss = train_result["loss"]
                    else:
                        loss = train_result
                    loss.backward()
                    self.optimizer.step()
                    self.steps += 1
                    pbar.update()

                self.model.eval()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(self.valid_data_loader):
                        self.writer.set_step()
                        valid_result = self._valid_epoch(batch_idx, batch)
                        self.steps += 1
                        pbar.update()
                        
                    
                    
                    
            # result = self._train_epoch(epoch)

            # save logged informations into log dict
            # log = {'epoch': epoch}
            # log.update(result)

            # print logged informations to the screen
            # for key, value in log.items():
            #     self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            # best = False
            # if self.mnt_mode != 'off':
            #     try:
            #         # check whether model performance improved or not, according to specified metric(mnt_metric)
            #         improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
            #                    (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
            #     except KeyError:
            #         self.logger.warning("Warning: Metric '{}' is not found. "
            #                             "Model performance monitoring is disabled.".format(self.mnt_metric))
            #         self.mnt_mode = 'off'
            #         improved = False

            #     if improved:
            #         self.mnt_best = log[self.mnt_metric]
            #         not_improved_count = 0
            #         best = True
            #     else:
            #         not_improved_count += 1

            #     if not_improved_count > self.early_stop:
            #         self.logger.info("Validation performance didn\'t improve for {} epochs. "
            #                          "Training stops.".format(self.early_stop))
            #         break

            # if epoch % self.save_period == 0:
            #     self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        for file in os.listdir(self.checkpoint_dir):
            """remove the old checkpoint"""
            if "checkpoint" in file:
                os.remove(self.checkpoint_dir/file)

        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def over(self):
        self.writer.finish()
