import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_smiles_len = _config["max_smiles_len"]
        self.setup_flag = False
        self._config = _config
        self.is_pretrain = _config['is_pretrain']

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            split="train",
            image_size=self.image_size,
            max_smiles_len=self.max_smiles_len,
            is_pretrain=self.is_pretrain,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            split="val",
            image_size=self.image_size,
            max_smiles_len=self.max_smiles_len,
            is_pretrain=self.is_pretrain,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            split="test",
            image_size=self.image_size,
            max_smiles_len=self.max_smiles_len,
            is_pretrain=self.is_pretrain,
        )

    def setup(self, stage):
        # setup_flag： set only once
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()
            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader
