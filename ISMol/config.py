from sacred import Experiment
ex = Experiment("myCLIP",save_git_info=False)

def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "fpp": 0,
        # downstream task
        "MoleculeNet_classify": 0,
    }
    ret.update(d)
    return ret

@ex.config
def config():
    exp_name = "pretrain"
    seed = 0
    datasets = ["pretrain"]
    loss_names = _loss_names({"mlm": 1 , "fpp": 1,'itm': 1})

    batch_size = 2048  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    max_image_len = -1
    image_size = 224
    patch_size = 16
    resolution_before = 224
    encoder_width = 256
    mask_ratio = 0.25

    # Text Setting
    max_smiles_len = 202
    tokenizer = "DeepChem/ChemBERTa-77M-MLM"
    vocab_size = 591
    mlm_probability = 0.15

    # Transformer Setting
    num_top_layer = 3
    input_image_embed_size = 768
    input_smiles_embed_size = 768
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1.5e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 60
    max_steps = 10000
    warmup_steps = 0.1
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = './data/dataset_pyarray'
    log_dir = "result"
    checkpoint_save_path = "./cpkt_path"
    mode = 'max'
    per_gpu_batchsize = 12   # you should define this manually with per_gpu_batch_size=#
    num_gpus = [0]
    num_nodes = 1
    load_path = ""  # path to load pretraining checkpoint
    num_workers = 4
    precision = 32
    is_pretrain = True
    imbsampler = False  # imbalance sampler
    drop_rate = 0.2
    log_project = 'myCLIP'

@ex.named_config
def task_finetune_MoleculeNet_classify():
    exp_name = "finetune_classifier"
    datasets = ['MoleculeNet_classify']     # 直接映射到分类数据加载模块
    loss_names = _loss_names({"MoleculeNet_classify": 1})
    warmup_steps = 0.1
    is_pretrain = False
    load_path = '/data1/zx/myCLIP/cpkt_path/total_train/p1-epoch=25-global_step=0-025.ckpt'
    # load_path = ""
    per_gpu_batchsize = 16
    seed = 0
    test_only = False
    mode = 'max'   # checkpoint

############################## downstream ###################################
@ex.named_config
def task_finetune_MoleculeNet_classify_HIA():
    data_root = './data/downstreamDataset/ADMETlab_data/HIA'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/HIA'
    log_dir = 'downstream_result/ADMETlab_data/HIA'
    log_project = 'hia'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_pgb_sub():
    data_root = './data/downstreamDataset/ADMETlab_data/pgb_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/pgb_sub'
    log_dir = 'downstream_result/ADMETlab_data/pgb_sub'
    log_project = 'pgb_sub'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_F20():
    data_root = './data/downstreamDataset/ADMETlab_data/F20'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/F20'
    log_dir = 'downstream_result/ADMETlab_data/F20'
    log_project = 'F20'

@ex.named_config
def task_finetune_MoleculeNet_classify_F30():
    data_root = './data/downstreamDataset/ADMETlab_data/F30'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/F30'
    log_dir = 'downstream_result/ADMETlab_data/F30'
    log_project = 'F30'

@ex.named_config
def task_finetune_MoleculeNet_classify_FDAMDD():
    data_root = './data/downstreamDataset/ADMETlab_data/FDAMDD'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/FDAMDD'
    log_dir = 'downstream_result/ADMETlab_data/FDAMDD'
    log_project = 'FDAMDD'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP1A2():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP1A2_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP1A2_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP1A2_sub'
    log_project = 'CYP1A2'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C19():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C19_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2C19_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C19_sub'
    log_project = 'CYP2C19'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C9():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C9_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2C9_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C9_sub'
    log_project = 'CYP2C9'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2D6():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2D6_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2D6_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP2D6_sub'
    log_project = 'CYP2D6'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP3A4():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP3A4_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP3A4_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP3A4_sub'
    log_project = 'CYP3A4'


@ex.named_config
def task_finetune_MoleculeNet_classify_T12():
    data_root = './data/downstreamDataset/ADMETlab_data/T12'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/T12'
    log_dir = 'downstream_result/ADMETlab_data/T12'
    log_project = 'T12'

@ex.named_config
def task_finetune_MoleculeNet_classify_DILI():
    data_root = './data/downstreamDataset/ADMETlab_data/DILI'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/DILI'
    log_dir = 'downstream_result/ADMETlab_data/DILI'
    log_project = 'DILI'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_SkinSen():
    data_root = './data/downstreamDataset/ADMETlab_data/SkinSen'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/SkinSen'
    log_dir = 'downstream_result/ADMETlab_data/SkinSen'
    log_project = 'SkinSen'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_Respiratory():
    data_root = './data/downstreamDataset/ADMETlab_data/Respiratory'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/Respiratory'
    log_dir = 'downstream_result/ADMETlab_data/Respiratory'
    log_project = 'Respiratory'

#################################  EGFR ####################################
@ex.named_config
def task_finetune_EGFR_classify():
    exp_name = "task_egfr_feature"
    datasets = ['MoleculeNet_classify']
    loss_names = _loss_names({"MoleculeNet_classify": 1})
    is_pretrain = False
    load_path = '/data1/zx/myCLIP/cpkt_path/total_train/p1-epoch=25-global_step=0-025.ckpt'
    # load_path = ""
    per_gpu_batchsize = 16
    seed = 0
    test_only = False
    mode = 'max'

@ex.named_config
def task_finetune_EGFR_classify_feature():
    data_root = '/data1/zx/data/EGFR/Feature'
    checkpoint_save_path = './cpkt_path/EGFR/Feature'
    log_dir = 'downstream_result/EGFR/Feature'
    log_project = 'EGFR_Feature'
    imbsampler = True

@ex.named_config
def task_finetune_EGFR_classify_past():
    data_root = '/data1/zx/data/EGFR/Past'
    checkpoint_save_path = './cpkt_path/EGFR/Past'
    log_dir = 'downstream_result/EGFR/Past'
    log_project = 'EGFR_Past'