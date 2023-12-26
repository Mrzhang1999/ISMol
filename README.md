# ISMol

## Data
The pre-training initial database is a molecular dataset collected from two large-scale drug databases (ChEMBL and ZINC), which are publicly available[1].

Downstream mission dataset selected from ADMET 2.0 [2].

The EGFR downstream task dataset is selected from ChemBL and PubChem.

[1] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec, “Strategies for pre-training graph neural networks,” arXiv preprint arXiv:1905.12265, 2019.
[2] G. Xiong, Z. Wu, J. Yi, L. Fu, Z. Yang, C. Hsieh, M. Yin, X. Zeng, C. Wu, A. Lu et al., “Admetlab 2.0: an integrated online platform for accurate and comprehensive predictions of admet properties,” Nucleic Acids Research, vol. 49, no. W1, pp. W5–W14, 2021.


## Pre-training stage
```
python pretrain.py
```
## Fine-turning stage
```
### training
python pretrain.py with task_finetune_MoleculeNet_classify task_finetune_MoleculeNet_classify_HIA num_gpus=[1] max_steps=2000 learning_rate=1e-5 batch_size=32

### testing
python pretrain.py with task_finetune_MoleculeNet_classify task_finetune_MoleculeNet_classify_HIA num_gpus=[1] test_only=True load_path='./ADMETlab_data/HIA/***.ckpt'
```

# Conda environment
The environment can be created by
```
conda create --name <env_name> --file requirements.txt
```
