from .MoleculeNetClassify_datamodule import MoleculeNetClassifyDataModule

from .pre_datamodule import pretrainDataModule

_datamodules = {
    "pretrain": pretrainDataModule,
    'MoleculeNet_classify': MoleculeNetClassifyDataModule
}