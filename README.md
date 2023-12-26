# ISMol

## Data
The pre-training initial database is a molecular dataset collected from two large-scale drug databases (ChEMBL and ZINC), which are publicly available[1].
[1] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec, “Strategies for pre-training graph neural networks,” arXiv preprint arXiv:1905.12265, 2019.

## Pre-training
'''
python pretrain.py
'''
## Fine-turning stage
'''
### training
python pretrain.py with task_finetune_MoleculeNet_classify task_finetune_MoleculeNet_classify_HIA num_gpus=[1] max_steps=2000 learning_rate=1e-5 batch_size=32

### testing
python pretrain.py with task_finetune_MoleculeNet_classify task_finetune_MoleculeNet_classify_HIA num_gpus=[1] test_only=True load_path='./ADMETlab_data/HIA/***.ckpt'
'''

# Conda environment
The environment can be created by
'''
conda create --name <env_name> --file requirements.txt
'''
