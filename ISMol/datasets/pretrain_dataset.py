from .augment_transform import load_norm_transform
from .base_dataset import BaseDataset

class Pretrain_dataset(BaseDataset):
    def __init__(self, *args, image_size,max_smiles_len, split="",is_pretrain=True, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.smiles_column_name = 'Smiles'
        self.labelled = (split == 'train')

        self.transform = load_norm_transform(image_size)
        self.max_smiles_len = max_smiles_len
        self.is_pretrain = is_pretrain

        if split == "train":
            names = ["test"]
        elif split == "val":
            names = ["test"]
        elif split == "test":
            names = ['test']

        super().__init__(*args, **kwargs, transforms=self.transform, names=names, image_size=image_size, smiles_column_name = self.smiles_column_name,split = self.split,is_pretrain=self.is_pretrain)

    def __getitem__(self, index):
        return self.getItem(index)
