from datasets import Pretrain_dataset
from .datamodules_base import BaseDataModule

class pretrainDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Pretrain_dataset

    @property
    def dataset_name(self):
        return "pretrain"
