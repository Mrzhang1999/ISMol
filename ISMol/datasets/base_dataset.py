import json
import os.path
import random
import re
# from xeger import Xeger
import pandas as pd
import torch
import torchvision.transforms as transforms
from rdkit import Chem
from torch.utils.data import Dataset
from PIL import Image,ImageFile
import numpy as np
import pyarrow as pa
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import io

class BaseDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 image_size: int,
                 names: list,
                 transforms,
                 smiles_column_name: str = "Smiles",
                 fingerprint_column_name: str = "Fingerprint",
                 k_100_column_name: str = 'k_100',
                 k_1000_column_name: str =  'k_1000',#'idd2k_1000',
                 k_10000_column_name: str = 'k_10000',# 'idd2k_10000',
                 label_name = None,
                 split = 'train',
                 is_pretrain=True,
                 MoleculeNet = True,
                 ocsr = False
            ):
        self.image_size = image_size
        self.smiles_column_name = smiles_column_name
        self.names = names
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms
        self.is_pretrain = is_pretrain
        # self._x = Xeger()
        self.regex = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        if len(names) != 0:
            print(f"{data_dir}/{names[0]}.arrow")
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table = pa.concat_tables(tables, promote=True)
            # Pretraining
            if self.is_pretrain and smiles_column_name != "" and k_100_column_name!="":
                self.smiles_column_name = smiles_column_name
                self.fingerprint_column_name = fingerprint_column_name
                self.k_100_column_name = k_100_column_name
                self.all_smiles = self.table[smiles_column_name].to_pandas().tolist()
                self.all_k_100 = self.table[k_100_column_name].to_pandas().tolist()
                self.all_k_1000 = self.table[k_1000_column_name].to_pandas().tolist()
                self.all_k_10000 = self.table[k_10000_column_name].to_pandas().tolist()

            # downstream
            elif not self.is_pretrain and smiles_column_name != "" and MoleculeNet == True:
                self.smiles_column_name = smiles_column_name
                self.all_smiles = self.table[smiles_column_name].to_pandas().tolist()
                self.labels = self.table[label_name].to_pandas().tolist()
            # error
            else:
                self.all_smiles = list()
                self.all_fingerprint = list()
                self.all_k_100 = list()
                self.all_k_1000 = list()
                self.all_k_10000 = list()
        # error
        else:
            self.all_smiles = list()
            self.all_fingerprint = list()
            self.all_k_100 = list()
            self.all_k_1000 = list()
            self.all_k_10000 = list()

    def get_raw_image(self, index, image_key="images"):
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="images"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = self.transforms(image)
        return image_tensor

    def get_false_image(self, image_key="images"):
        random_index = random.randint(0, len(self.all_smiles) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = self.transforms(image)
        return image_tensor

    def get_smiles(self, index):
        smiles = self.all_smiles[index]
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, doRandom=True)  # 不同表示
        return smiles

    def get_false_smiles(self,index):
        raw_smiles = self.all_smiles[index]
        smiles_false = self.change_smiles(raw_smiles)
        return smiles_false

    def get_k_100(self,index):
        k_100 = self.all_k_100[index] # self.all_k_100[index][0]
        return k_100

    def get_k_1000(self,index):
        k_1000 = self.all_k_1000[index] # self.all_k_1000[index][0]
        return k_1000

    def get_k_10000(self,index):
        k_1000 = self.all_k_10000[index] # self.all_k_10000[index][0]
        return k_1000

    def get_label(self,index):
        assert not self.is_pretrain,"now,pretraining!!just download stream tasks have labels."
        return int(self.labels[index]) # self.labels[index][0] | self.labels[index]

    def get_labels(self):
        # labels = np.array(self.labels)
        # return labels[:,0].tolist()
        # return labels.tolist()
        return self.labels

    def get_oscr_smiles(self, index):
        smiles = self.all_smiles[index]
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, doRandom=True)
        return smiles

    def __len__(self):
        return len(self.all_smiles)

    def getItem(self, index):
        image = self.get_image(index)
        image_false = self.get_false_image()
        smiles = self.get_smiles(index)
        k_100 = self.get_k_100(index)
        k_1000 = self.get_k_1000(index)
        k_10000 = self.get_k_10000(index)
        smiles_false = self.get_false_smiles(index)
        return image,image_false,smiles,smiles_false,k_100,k_1000,k_10000

    def ocsr_predict_getItem(self,index):
        image = self.get_image(index)
        smiles = self.get_smiles(index)
        image_id = self.all_image_id[index]
        return smiles,image,int(image_id)

    def ocsr_getItem(self,index):
        image = self.get_image(index)
        smiles = self.get_oscr_smiles(index)
        image_id = self.all_image_id[index]
        image_false = self.get_false_image()
        return image,image_false,smiles,int(image_id)

    def get_num_to_modify(self, length, pn0=0.1, pn1=0.3, pn2=0.5,pn3=0.8):
        prob = random.random()
        if prob < pn0: num_to_modify = 1
        elif prob < pn1: num_to_modify = 2
        elif prob < pn2: num_to_modify = 3
        elif prob < pn3: num_to_modify = 4
        else: num_to_modify = 5

        if length <= 4: num_to_modify = min(num_to_modify, 1)
        else: num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify

    def change_smiles(self, smiles,pt0=0.25, pt1=0.5, pt2=0.75):

        length = len(smiles)
        num_to_modify = self.get_num_to_modify(length)

        raw_chars = re.findall(self.regex,smiles)
        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]

        for i, t in enumerate(raw_chars):
            if i not in index: chars.append(t)
            else:
                prob = random.random()
                randomAotm = raw_chars[random.randint(0,len(raw_chars)-1)]
                # randomAotm = self._x.xeger(self.regex)
                if prob < pt0: # replace
                    chars.append(randomAotm)
                elif prob < pt1: # insert
                    chars.append(randomAotm)
                    chars.append(t)
                elif prob < pt2: # insert
                    chars.append(t)
                    chars.append(t)
                else: # delete
                    continue

        new_smiles = ''.join(chars[: 202-1])
        return new_smiles