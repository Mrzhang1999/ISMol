import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchsampler import ImbalancedDatasetSampler
from . import _datamodules

# 数据模块，设置并行
class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=True):
        datamodule_keys = _config["datasets"]   # ["pre_train"]
        assert len(datamodule_keys) > 0
        super().__init__()

        self.dm_keys = datamodule_keys  # [ "pretrain"]
        # self.dm_dicts = {
        #     "pretrain": pretrainDataModule,
        # }
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        # dms = [pretrainDataModule]

        self.dms = [v for k, v in self.dm_dicts.items()]
        self.batch_size = self.dms[0].batch_size
        self.num_workers = self.dms[0].num_workers
        self.dist = dist
        self.imbsampler = _config['imbsampler']

    def setup(self, stage):
        # 划分数据集（训练、验证、测试）,实例化，train_dataset、val_dataset、test_dataset
        for dm in self.dms:
            dm.setup(stage)
        # 将不同数据集中的数据进行拼接（训练、验证、测试）
        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
        # 分布式划分
        if self.imbsampler:
            self.train_sampler = ImbalancedDatasetSampler(self.train_dataset)
            self.val_sampler = ImbalancedDatasetSampler(self.val_dataset)
            self.test_sampler = ImbalancedDatasetSampler(self.test_dataset)
        elif self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
        )
        return loader