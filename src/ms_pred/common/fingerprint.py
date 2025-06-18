"""fingerprint.py """

import numpy as np

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import AllChem, DataStructs


def get_morgan_fp(mol: Chem.Mol, nbits: int = 2048, radius=3, isbool = False) -> np.ndarray:
    """get_morgan_fp."""

    if mol is None:
        return None

    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    fingerprint = np.zeros((0,), dtype=np.uint8) if not isbool else  np.zeros((0,), dtype=np.bool_)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint


def get_morgan_fp_wt(mol: Chem.Mol, nbits: int = 2048, radius=3, isbool = False) -> np.ndarray:
    """get_morgan_fp."""

    if mol is None:
        return None, None

    weight = MolWt(mol)
    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    fingerprint = np.zeros((0,), dtype=np.uint8) if not isbool else np.zeros((0,), dtype=np.bool_)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint, weight


def get_morgan_fp_smi(smi: str, nbits: int = 2048, radius=3, isbool = False) -> np.ndarray:
    return get_morgan_fp(Chem.MolFromSmiles(smi), nbits=nbits, radius=radius, isbool = isbool)

def get_morgan_fp_inchi(inchi: str, nbits: int = 2048, radius=3, isbool = False) -> np.ndarray:
    return get_morgan_fp(Chem.MolFromInchi(inchi), nbits=nbits, radius=radius, isbool = isbool)

def get_morgan_fp_smi_wt(smi: str, nbits: int = 2048, radius=3, isbool = False) -> np.ndarray:
    return get_morgan_fp_wt(Chem.MolFromSmiles(smi), nbits=nbits, radius=radius, isbool=isbool)
