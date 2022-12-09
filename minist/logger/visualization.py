import importlib
from datetime import datetime


class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        """计算每秒的step数"""
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class WandbWriter():
    def __init__(self, config, logger, enabled):
        self.writer = None
        """config of wandb"""
        self.wandb_config = config.config.get("wandb", dict())
        self.wandb_config.update({"dir": config.log_dir})
        # self.wandb_config.update({"config": config.config})

        if enabled:
            # Retrieve vizualization writer.
            succeeded = False
            try:
                self.writer = importlib.import_module("wandb").init(**self.wandb_config)
                succeeded = True
            except ImportError:
                succeeded = False

            if not succeeded:
                message = "Warning: visualization (wandb) is configured to use, but currently not installed on " \
                    "this machine. Please install wandb with 'pip install wandb'"
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.timer = datetime.now()

    def set_step(self):
        """计算每秒的step数"""
        self.step += 1
        duration = datetime.now() - self.timer
        self.log('steps_per_sec', 1 / duration.total_seconds())
        self.timer = datetime.now()

    def log(self, tag, value):
        """
        if writer is not None,
        log data with some additional info
        """
        if self.writer is None:
            return None
        # key = f"{self.mode}/{tag}"
        data = {tag: value}
        self.writer.log(data, self.step) 
        # self.writer.log(data) 

    def finish(exit_code=0, quiet=None):
        importlib.import_module("wandb").finish(exit_code, quiet)
        