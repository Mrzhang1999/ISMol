import os
from collections import Counter

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Draw
import json
from data.preprocessing import preprocess_list

def smiles2image_save_as_json_by_one_file(path, image_out_dir, json_out_dir):
    out_path = os.path.join(os.getcwd(), image_out_dir)
    os.makedirs(out_path, exist_ok=True)
    if path.split('.')[-1] not in ['csv', 'txt', 'tsv']:
        raise NotImplementedError
    if path.split('.')[-1] in ['tsv']:  # moleculeNet
        df = pd.read_csv(path, sep='\t')
    else:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
    print(df.columns)
    df = preprocess_list(df,columns='smiles')
    df = filter(df,columns='canonical_smiles')
    print(df.columns)
    df.columns = ['group', 'labels', 'smiles']
    df_train = df[df['group'] == 'training'].reset_index(drop=True)
    df_dev = df[df['group'] == 'val'].reset_index(drop=True)
    df_test = df[df['group'] == 'test'].reset_index(drop=True)
    generate_images_json(df_train,image_out_dir,json_out_dir,'train')
    generate_images_json(df_dev,image_out_dir,json_out_dir,'dev')
    generate_images_json(df_test,image_out_dir,json_out_dir,'test')

def generate_images_json(df,image_out_dir,json_out_dir,split):
    s = 0
    df['image_id'] = df.index
    smi_list = df['smiles']
    print(Counter(df['labels']))
    os.makedirs(f'{image_out_dir}/{split}',exist_ok=True)
    for index, smi in enumerate(smi_list):

        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            continue

        Draw.MolToFile(mol, f'{image_out_dir}/{split}/{df["image_id"][index]}.png')
        s += 1
        if index % 1000 == 0:
            print(f'{index // 1000}k is done!')

    print(f"smiles2image is done!! Total: {s}")
    json_obj = eval(df.to_json(orient='records'))

    with open(f'{json_out_dir}/{split}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_obj, indent=2, ensure_ascii=False))

def filter(smiles,columns):
    df = pd.DataFrame(smiles)
    df = df.drop([columns], axis=1)
    df = df.dropna(subset=["smiles"])
    df = df.reset_index(drop=True)
    return df

