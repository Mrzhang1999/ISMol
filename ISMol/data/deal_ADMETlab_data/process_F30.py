from data.deal_ADMETlab_data.utils import smiles2image_save_as_json_by_one_file

if __name__ == '__main__':
    smiles2image_save_as_json_by_one_file(path='/data1/zx/myCLIP/data/ADMETlab_data/F(30%)_canonical.csv',
                                          image_out_dir='/data1/zx/myCLIP/data/downstreamDataset/ADMETlab_data/F30/images',
                                          json_out_dir='/data1/zx/myCLIP/data/downstreamDataset/ADMETlab_data/F30')