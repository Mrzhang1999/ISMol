import pandas as pd
import pyarrow as pa

def write_table(df,dataset_root,split):
    table = pa.Table.from_pandas(df)
    with pa.OSFile(
            f"{dataset_root}/{split}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    print('donw!!')

def read_image_Oom(id,image_folder_path):
    path = image_folder_path + str('/') + str(id) + str('.png')
    with open(path, "rb") as fp:
        binary = fp.read()
    return binary
from collections import Counter
def make_arrow_Oom(csv_path, image_folder_path,dataset_root,split = 'train',path=None):
    df = pd.read_csv(csv_path)

    df.columns = ['Unnamed: 0', 'molecule chembl id', 'cidcdate', 'molecular weight',
       'standard type', 'standard relation', 'standard value',
       'standard units', 'labels', 'smiles', 'image_id']
    df = df.drop(["smiles"], axis=1)
    df = df.dropna(subset=["Smiles"])
    save_df = df[['Smiles','labels']]
    save_df.columns = ['Smiles','label']
    # save_df.to_csv(path + '.csv',index=False)
    print(Counter(save_df['label']))

    # df['images'] = df['image_id'].apply(read_image_Oom,args=(image_folder_path,))
    # write_table(df, dataset_root, split)

make_arrow_Oom(csv_path='/data1/zx/myCLIP/data/EGFR/Feature/train.csv',
           image_folder_path = '/data1/zx/data/EGFR/Feature/images/train',
           dataset_root='/data1/zx/data/EGFR/Feature',
           split = 'train',
            path='/home/hndx/CD-MVGNN/data/EGFR/Feature/train'
           )
make_arrow_Oom(csv_path='/data1/zx/myCLIP/data/EGFR/Feature/valid.csv',
           image_folder_path = '/data1/zx/data/EGFR/Feature/images/valid',
           dataset_root='/data1/zx/data/EGFR/Feature',
           split = 'dev',
            path='/home/hndx/CD-MVGNN/data/EGFR/Feature/valid'
           )
make_arrow_Oom(csv_path='/data1/zx/myCLIP/data/EGFR/Feature/test.csv',
           image_folder_path = '/data1/zx/data/EGFR/Feature/images/test',
           dataset_root='/data1/zx/data/EGFR/Feature',
           split = 'test',
            path='/home/hndx/CD-MVGNN/data/EGFR/Feature/test'
           )


make_arrow_Oom(csv_path='/data1/zx/myCLIP/data/EGFR/Past/train.csv',
           image_folder_path = '/data1/zx/data/EGFR/Past/images/train',
           dataset_root='/data1/zx/data/EGFR/Past',
           split = 'train',
            path='/home/hndx/CD-MVGNN/data/EGFR/Past/train'
           )
make_arrow_Oom(csv_path='/data1/zx/myCLIP/data/EGFR/Past/valid.csv',
           image_folder_path = '/data1/zx/data/EGFR/Past/images/valid',
           dataset_root='/data1/zx/data/EGFR/Past',
           split = 'dev',
            path='/home/hndx/CD-MVGNN/data/EGFR/Past/valid'
           )
make_arrow_Oom(csv_path='/data1/zx/myCLIP/data/EGFR/Past/test.csv',
           image_folder_path = '/data1/zx/data/EGFR/Past/images/test',
           dataset_root='/data1/zx/data/EGFR/Past',
           split = 'test',
            path='/home/hndx/CD-MVGNN/data/EGFR/Past/test'
           )