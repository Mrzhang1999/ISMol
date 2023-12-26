from data.write_arrow.utils import make_arrow

make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/ADMETlab_data/pgb_sub',
           data_split_path='train.json',
           dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/ADMETlab_data/pgb_sub',
           split = 'train'
           )

make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/ADMETlab_data/pgb_sub',
           data_split_path='test.json',
           dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/ADMETlab_data/pgb_sub',
           split = 'test'
           )

make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/ADMETlab_data/pgb_sub',
           data_split_path='dev.json',
           dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/ADMETlab_data/pgb_sub',
           split = 'dev'
           )
print('done!')