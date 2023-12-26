import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import MACCSkeys


def path2rest(path, iid2smiles,idd2k_100,idd2k_1000,idd2k_10000):
    name = path.split("/")[-1].split('.png')[0]
    with open(path, "rb") as fp:
        binary = fp.read()
    smiles = iid2smiles[name]
    k_100 = idd2k_100[name]
    k_1000 =idd2k_1000[name]
    k_10000 = idd2k_1000[name]
    return [binary, smiles, name, k_100,k_1000,k_10000]
# smiles_image_pairs/5w_data.json

def get_fingerprint(smiles_String):
    molecule = Chem.MolFromSmiles(smiles_String)
    fingerprint = MACCSkeys.GenMACCSKeys(molecule)
    return fingerprint

def make_arrow(root,data_split_path, dataset_root,split = 'train'):
    with open(f"{root}/{data_split_path}", "r") as fp:
        pairs = json.load(fp)

    iid2smiles = defaultdict(list)
    idd2k_100 = defaultdict(list)
    idd2k_1000 = defaultdict(list)
    idd2k_10000 = defaultdict(list)

    for cap in tqdm(pairs):
        # filename = cap['image_id'].split("/")[-1]
        filename = str(cap['image_id'])
        iid2smiles[filename].append(cap['Smiles'])
        idd2k_100[filename].append(cap['k_100'])
        idd2k_1000[filename].append(cap['k_1000'])
        idd2k_10000[filename].append(cap['k_10000'])

    paths = list(glob(f"{root}/images_30w7w/{split}/*.png"))
    random.shuffle(paths)

    smiles_paths = [path for path in paths if path.split("/")[-1].split('.png')[0] in iid2smiles]

    if len(paths) == len(smiles_paths):
        print("all images have smiles pair")
    else:
        print("not all images have smiles pair")
    print(
        len(paths), len(smiles_paths),
    )

    bs = [path2rest(path, iid2smiles,idd2k_100,idd2k_1000,idd2k_10000) for path in tqdm(smiles_paths)]

    dataframe = pd.DataFrame(
        bs, columns=["images", "Smiles", "image_id", "k_100","idd2k_1000","idd2k_10000"],
    )

    table = pa.Table.from_pandas(dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
        f"{dataset_root}/smiles_image_pairs_30w7w{split}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

make_arrow(root='.',
           data_split_path='train_30w.json',
           dataset_root = './dataset_pyarray',
           split = 'train'
           )

make_arrow(root='.',
           data_split_path='test_7w.json',
           dataset_root = './dataset_pyarray',
           split = 'test'
           )
print('done!')