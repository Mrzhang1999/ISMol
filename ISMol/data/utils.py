import pandas
import os
import re
import pandas as pd
import torchvision.transforms
import csv
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import Draw
import json

'''
    smiles2images from csv
    path:csv path
    out_dir:output dir
'''
def smiles2image_save_as_json(path,image_out_dir,json_out_dir,json_name,start_id=0):
    out_path = os.path.join(os.getcwd(),image_out_dir)
    os.makedirs(out_path, exist_ok=True)

    if path.split('.')[-1] not in ['csv','txt']:
        raise NotImplementedError

    df = pd.read_csv(path)
    smi_list = df['Smiles']
    df['image_id'] = range(start_id,start_id + len(smi_list.values))

    for index, smi in enumerate(smi_list):
        mol = Chem.MolFromSmiles(smi)
        Draw.MolToFile(mol, f'{out_path}/{df["image_id"][index]}.png')
        if index % 10000 == 0:
            print(f'{index // 10000}w is done!')
    # update csv
    print("smiles2image is done!!")
    json_obj = eval(df.to_json(orient='records'))
    with open(f'{json_out_dir}/{json_name}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_obj, indent=2, ensure_ascii=False))

'''
    MW : 12-600
    Heavy Atom Count : 3-50
    logP: -7 - 5
    
    :params
    df:DataFrame Object,
    output_dir:path
    output_file_name:str
'''
def evaluate(df,output_dir,output_file_name):
    df_ok = df[
        df.MolWeight.between(*[12, 600]) &  # MW
        df.logP.between(*[-7, 5]) &  # LogP
        df.heavyAtomCount.between(*[3, 50])  # HeavyAtomCount
    ]
    return df_ok

'''
csv_file->json_file
'''
def csv2json(df,out_dir,json_name):
    json_obj = eval(df.to_json(orient='records'))
    with open(f'{out_dir}/{json_name}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_obj, indent=2, ensure_ascii=False))
    return json_obj


def smiles2image_save_as_json_by_one_file(path,image_out_dir,json_out_dir,json_name,start_id=0):
    out_path = os.path.join(os.getcwd(),image_out_dir)
    os.makedirs(out_path, exist_ok=True)

    if path.split('.')[-1] not in ['csv','txt','tsv']:
        raise NotImplementedError
    if path.split('.')[-1] in ['tsv']:  # moleculeNet
        df = pd.read_csv(path,sep = '\t')
        smi_list = df['text_a']
    else:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        smi_list = df['smiles']
    df['image_id'] = df.index

    s = 0
    for index, smi in enumerate(smi_list):
        mol = Chem.MolFromSmiles(smi)
        if mol==None:
            continue
        Draw.MolToFile(mol, f'{out_path}/{df["image_id"][index]}.png')
        s += 1
        if index % 1000 == 0:
            print(f'{index // 1000}k is done!')

    print(f"smiles2image is done!! Total: {s}")
    json_obj = eval(df.to_json(orient='records'))
    with open(f'{json_out_dir}/{json_name}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_obj, indent=2, ensure_ascii=False))
