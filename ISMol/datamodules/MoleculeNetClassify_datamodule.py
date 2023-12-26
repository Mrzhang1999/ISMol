from datasets import MoleculeNet_classify_dataset
from .datamodules_base import BaseDataModule

class MoleculeNetClassifyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MoleculeNet_classify_dataset

    @property
    def dataset_name(self):
        return "MoleculeNet_classify"