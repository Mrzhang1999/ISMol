import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict

def path2rest(path, iid2smiles, iid2labels):
    name = path.split("/")[-1].split('.png')[0]
    with open(path, "rb") as fp:
        binary = fp.read()
    smiles = iid2smiles[name]
    label = iid2labels[name]
    return [binary, smiles, label]


def make_arrow(root,data_split_path, dataset_root,split = 'train'):
    with open(f"{root}/{data_split_path}", "r") as fp:
        pairs = json.load(fp)
    print(len(pairs))
    iid2smiles = defaultdict(list)
    idd2labels = defaultdict(list)

    for cap in tqdm(pairs):
        filename = str(cap['image_id'])
        iid2smiles[filename].append(cap['smiles'])
        # idd2labels[filename].append(eval(cap['labels'])[0])
        idd2labels[filename].append(cap['labels'])

    paths = list(glob(f"{root}/images/{split}/*.png"))
    random.shuffle(paths)

    smiles_paths = [path for path in paths if path.split("/")[-1].split('.png')[0] in iid2smiles]

    if len(paths) == len(smiles_paths):
        print("all images have smiles pair")
    else:
        print("not all images have smiles pair")
    print(
        len(paths), len(smiles_paths),
    )

    bs = [path2rest(path, iid2smiles, idd2labels) for path in tqdm(smiles_paths)]

    dataframe = pd.DataFrame(
        bs, columns=["images", "Smiles", 'labels'],
    )

    table = pa.Table.from_pandas(dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
        f"{dataset_root}/{split}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
