"""Functions that can be used to preprocess SMILES sequnces in the form used in the publication."""
import numpy as np
import pandas as pd
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors
REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])

def randomize_smile(sml):
    """Function that randomizes a SMILES sequnce. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequnce to randomize.
    Return:
        randomized SMILES sequnce or
        nan if SMILES is not interpretable.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=False)
    except:
        return float('nan')

def canonical_smile(sml):
    """Helper Function that returns the RDKit canonical SMILES for a input SMILES sequnce.
    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce."""
    return Chem.MolToSmiles(sml, canonical=True)

def keep_largest_fragment(sml):
    """Function that returns the SMILES sequence of the largest fragment for a input
    SMILES sequnce.

    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce of the largest fragment.
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)

def remove_salt_stereo(sml, remover):
    """Function that strips salts and removes stereochemistry information from a SMILES.
    Args:
        sml: SMILES sequence.
        remover: RDKit's SaltRemover object.
    Returns:
        canonical SMILES sequnce without salts and stereochemistry information.
    """
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml),
                                                dontRemoveEverything=True),
                               isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        sml = np.float("nan")
    return(sml)

def organic_filter(sml):
    """Function that filters for organic molecules.
    Args:
        sml: SMILES sequence.
    Returns:
        True if sml can be interpreted by RDKit and is organic.
        False if sml cannot interpreted by RDKIT or is inorganic.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        if is_organic:
            return True
        else:
            return False
    except:
        return False

def filter_smiles(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET
        if ((logp > -5) & (logp < 7) &
            (mol_weight > 12) & (mol_weight < 600) &
            (num_heavy_atoms > 3) & (num_heavy_atoms < 50) &
            is_organic ):
            return Chem.MolToSmiles(m)
        else:
            return float('nan')
    except:
        return float('nan')
    
def get_descriptors(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        descriptor_list = []
        descriptor_list.append(Descriptors.MolLogP(m))
        descriptor_list.append(Descriptors.MolMR(m)) #ok
        descriptor_list.append(Descriptors.BalabanJ(m))
        descriptor_list.append(Descriptors.NumHAcceptors(m)) #ok
        descriptor_list.append(Descriptors.NumHDonors(m)) #ok
        descriptor_list.append(Descriptors.NumValenceElectrons(m))
        descriptor_list.append(Descriptors.TPSA(m)) # nice
        return descriptor_list
    except:
        return [np.float("nan")] * 7
def create_feature_df(smiles_df):
    temp = list(zip(*smiles_df['canonical_smiles'].map(get_descriptors)))
    columns = ["MolLogP", "MolMR", "BalabanJ", "NumHAcceptors", "NumHDonors", "NumValenceElectrons", "TPSA"]
    df = pd.DataFrame(columns=columns)
    for i, c in enumerate(columns):
        df.loc[:, c] = temp[i]
    df = (df - df.mean(axis=0, numeric_only=True)) / df.std(axis=0, numeric_only=True)
    df = smiles_df.join(df)
    return df

def preprocess_smiles(sml):
    """Function that preprocesses a SMILES string such that it is in the same format as
    the translation model was trained on. It removes salts and stereochemistry from the
    SMILES sequnce. If the sequnce correspond to an inorganic molecule or cannot be
    interpreted by RDKit nan is returned.

    Args:
        sml: SMILES sequence.
    Returns:
        preprocessd SMILES sequnces or nan.
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    new_sml = filter_smiles(new_sml)
    return new_sml


def preprocess_list(smiles,columns = 'text_a'):
    df = pd.DataFrame(smiles)
    df["canonical_smiles"] = df[columns].map(preprocess_smiles)
    df = df.drop([columns], axis=1)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.reset_index(drop=True)
    # df["random_smiles"] = df["canonical_smiles"].map(randomize_smile)
    # df = create_feature_df(df)
    return df
    