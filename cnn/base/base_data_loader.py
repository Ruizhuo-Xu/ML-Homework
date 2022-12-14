import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate, k_folders=None, k=0):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.k_folders = k_folders
        if k_folders:
            assert isinstance(k_folders, int) and k_folders > 0
            self.k = k
            self.sampler, self.valid_sampler = self._k_folder_sampler(self.k_folders, self.k)
        else:
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        # 返回训练集
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def _k_folder_sampler(self, folders=5, k=0):
        idx_full = np.arange(self.n_samples)
        # 将所有的数据划分为K份
        len_valid = self.n_samples // folders
        # 根据目前的折数计算验证集的起始索引
        valid_start = len_valid * k
        # 选取验证集索引
        valid_idx = idx_full[valid_start: valid_start+len_valid]
        # 选取训练集索引
        train_idx = np.delete(idx_full, np.arange(valid_start, valid_start+len_valid))
        # 根据索引实例化Sampler
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler



    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            # 返回验证集
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
