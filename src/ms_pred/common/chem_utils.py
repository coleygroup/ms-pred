"""chem_utils.py"""

import re
import numpy as np
import pandas as pd
from functools import reduce
from typing import List
import logging
import torch
from rdkit import Chem
from rdkit.Chem import Atom
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import ExactMolWt
try:
    from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer, TautomerTransform
    _RD_TAUTOMER_CANONICALIZER = 'v1'
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform('1,3 heteroatom H shift',
                          '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]'),
    )
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator  # newer rdkit
    _RD_TAUTOMER_CANONICALIZER = 'v2'

P_TBL = Chem.GetPeriodicTable()

ROUND_FACTOR = 4

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "N",
    "P",
    "O",
    "S",
    "Si",
    "I",
    "H",
    "Cl",
    "F",
    "Br",
    "B",
    "Se",
    "Fe",
    "Co",
    "As",
    "Na",
    "K",
]

ELEMENT_TO_GROUP = {
    "C": 4,  # group 5
    "N": 3,  # group 4
    "P": 3,
    "O": 5,  # group 6
    "S": 5,
    "Si": 4,
    "I": 6,  # group 7 / halogens
    "H": 0,
    "Cl": 6,
    "F": 6,
    "Br": 6,
    "B": 2,  # group 3
    "Se": 5,
    "Fe": 7,  # transition metals
    "Co": 7,
    "As": 3,
    "Na": 1,  # alkali metals
    "K": 1,
}
ELEMENT_GROUP_DIM = len(set(ELEMENT_TO_GROUP.values()))
ELEMENT_GROUP_VECTORS = np.eye(ELEMENT_GROUP_DIM)

# Set the exact molecular weight?
# Use this to define an element priority queue
VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]
CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)


ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))

# Convert to onehot
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

# Use Monoisotopic
# VALID_MASSES = np.array([Atom(i).GetMass() for i in VALID_ELEMENTS])
VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

ELEMENT_DIM_MASS = len(ELEMENT_VECTORS_MASS[0])
ELEMENT_DIM = len(ELEMENT_VECTORS[0])

COLLISION_PE_DIM = 64
COLLISION_PE_SCALAR = 10000

<<<<<<< HEAD
=======
SIM_PE_DIM = 64
SIM_PE_SCALAR = 10

GRAPHTYPE_LEN = 32
>>>>>>> dev-local/main
# Reasonable normalization vector for elements
# Estimated by max counts (+ 1 when zero)
NORM_VEC_MASS = np.array(
    [81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1, 1471]
)

NORM_VEC = np.array([81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1])
MAX_ATOM_CT = 160


# For helping binary conversions
BINARY_BITS = 8

# Assume 64 is the highest repeat of any 1 atom
MAX_ELEMENT_NUM = 64

# Hydrogen featurizer
MAX_H = 6

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS))
element_to_group = {k: ELEMENT_GROUP_VECTORS[v] for k, v in ELEMENT_TO_GROUP.items()}

# Map ion to adduct mass, don't use electron
ion2mass = {
    # positive mode
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H-H2O]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M+NH4]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M+H-2H2O]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    # negative mode
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M+Cl]-": ELEMENT_TO_MASS["Cl"] + ELECTRON_MASS,
    "[M-H2O-H]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H-H2O]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H-CO2]-": -ELEMENT_TO_MASS["C"] - ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
}
    # More high probability adducts:
    #    '[M+H-NH3]+', '[M-H+2Na]+', '[M+CHO2]-', '[M+HCOOH-H]-', '[M+CH3COOH-H]-', '[M+CH3OH-H]-'
    #    '[M-H+2i]-', '[M+H-CH2O2]+', '[M+H-C2H4O2]+', '[M+H-C4H8]+',
    #    '[M-2H]2-', '[M+H-3H2O]+', '[M+H-CH4O]+', '[M+H-C2H4]+',
    #    '[M+2Na-H]+', '[M+H-H2O+2i]+', '[M+H-C2H2O]+', '[M-H-CH3]-',
    #    '[M+H-CO]+', '[M+H-C3H6]+', '[M+H-C2H6O]+', '[M+H-CH3]+',
    #    '[M+H-HF]+', '[Cat]+', '[2M-H+2i]-', '[M+H-C6H10O5]+',
    #    '[M+H-CH5N]+', '[M-H-C6H10O5]-', '[M+H+K]2+', '[M+OH]-',
    #    '[M+H-Br]+', '[M+H-C2H7N]+', '[M+H-CO2]+', '[M+H+Na]2+',
    #    '[2M+H+2i]+', '[M+H-HCl]+', '[M+H+4i]+', '[2M-H+4i]-', '[M+2Na]2+',
    #    '[M+H-CH4O3]+', '[M+H-C6H10]+', '[M+H-HCN]+', '[M-H-CO2+2i]-',
    #    '[M+K]+', '[M+H-CHNO]+', '[3M+H]+', '[M+H-C2H5N]+', '[M+Na+2i]+',
    #    '[M+3Na-2H]+'

# Valid adducts
ion2onehot_pos = {
    "[M+H]+": 0,
    "[M+Na]+": 1,
    "[M+K]+": 2,
    "[M-H2O+H]+": 3,
    "[M+H-H2O]+": 3,
    "[M+H3N+H]+": 4,
    "[M+NH4]+": 4,
    "[M]+": 5,
    "[M-H4O2+H]+": 6,
    "[M+H-2H2O]+": 6,
    "[M-H]-": 7,
    "[M+Cl]-": 8,
    "[M-H2O-H]-": 9,
    "[M-H-H2O]-": 9,
    "[M-H-CO2]-": 10,
}

ion_pos2extra_multihot = {v: set() for v in ion2onehot_pos.values()}
for k, v in ion2onehot_pos.items():
    _ion_mode = k[-1]  # '+' or '-'
    k = k.strip(_ion_mode).strip('[M').strip(']')

    # split string into a list formula differences
    _ions = []
    for _, i in enumerate(k.split('+')):
        if i:
            if _ != 0:
                i = '+' + i
            for __, j in enumerate(i.split('-')):
                if j:
                    if __ == 0:
                        _ions.append(j)
                    else:
                        _ions.append('-' + j)

    if _ion_mode == '+':
        ion_pos2extra_multihot[v].add(0)  # positive mode
    else:
        ion_pos2extra_multihot[v].add(1)  # negative mode
    if '+Na' in _ions or '+K' in _ions:
        ion_pos2extra_multihot[v].add(2)  # alkali metal
    if '+H' in _ions:
        ion_pos2extra_multihot[v].add(3)  # add proton
    if '-H' in _ions:
        ion_pos2extra_multihot[v].add(4)  # lose proton
    if '+Cl' in _ions:
        ion_pos2extra_multihot[v].add(5)  # halogen
    if '-H2O' in _ions or '-H4O2' in _ions or '-2H2O' in _ions:
        ion_pos2extra_multihot[v].add(6)  # lose water
    if '-CO2' in _ions:
        ion_pos2extra_multihot[v].add(7)  # lose CO2
    if '+NH3' in _ions:
        ion_pos2extra_multihot[v].add(8)  # get NH3

# add equivalent keys: [M]1+ == [M]+, [NH3] == [H3N]
_ori_ions = list(ion2mass.keys())
for ion in _ori_ions:
    adduct, charge = ion.split(']')
    if not charge[0].isnumeric():
        eq_ion = adduct + ']1' + charge
        ion2mass[eq_ion] = ion2mass[ion]
        if ion in ion2onehot_pos:
            ion2onehot_pos[eq_ion] = ion2onehot_pos[ion]

_ori_ions = list(ion2mass.keys())
for ion in _ori_ions:
    adduct, charge = ion.split(']')
    if 'H3N' in adduct:
        eq_ion = ion.replace('H3N', 'NH3')
        ion2mass[eq_ion] = ion2mass[ion]
        if ion in ion2onehot_pos:
            ion2onehot_pos[eq_ion] = ion2onehot_pos[ion]
            

instrument2onehot_pos = {
    "Orbitrap": 0,
    "QTOF": 1, # Streamline upstream preprocessing to make QToF count here too 
}


def is_positive_adduct(adduct_str: str) -> bool:
    """Check the adduct string is positive or negative (return True if positive)"""
    return adduct_str[-1] == '+'


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def cross_sum(x, y):
    """cross_sum."""
    return (x[None, :, :] + y[:, None, :]).reshape(-1, y.shape[-1])


def get_all_subsets(chem_formula: str) -> (np.ndarray, np.ndarray):
    """get_all_subsets.

    Args:
        chem_formula (str): Chem formula
    Return:
        Tuple of vecs and their masses
    """

    dense_formula = formula_to_dense(chem_formula)
    non_zero = np.argwhere(dense_formula > 0).flatten()

    vectorized_formula = [
        ELEMENT_VECTORS[nonzero_ind]
        * np.arange(0, dense_formula[nonzero_ind] + 1)[:, None]
        for nonzero_ind in non_zero
    ]

    cross_prod = reduce(cross_sum, vectorized_formula)
    cross_prod_inds = rdbe_filter(cross_prod)
    cross_prod = cross_prod[cross_prod_inds]

    all_masses = np.einsum("ij,j->i", cross_prod, VALID_MONO_MASSES)
    return cross_prod, all_masses


def rdbe_filter(cross_prod):
    """rdbe_filter.

    Args:
        cross_prod:
    """
    # Filter
    pos_els = ["C", "C", "N", "P"]
    neg_els = ["H", "Cl", "Br", "I", "F"]

    # Apply rdbe filter
    # RDBE = 1 + 0.5 * (2#C − #H +#N+#P−#Cl−#Br−#I−#F)
    rdbe_total = np.zeros(cross_prod.shape[0])
    for pos_el in pos_els:
        rdbe_total += cross_prod[:, element_to_ind[pos_el]]

    for neg_el in neg_els:
        rdbe_total -= cross_prod[:, element_to_ind[neg_el]]

    # Manage
    rdbe_total = 1 + 0.5 * rdbe_total
    filter_inds = np.argwhere(rdbe_total >= 0).flatten()
    return filter_inds


def formula_to_dense_mass(chem_formula: str) -> np.ndarray:
    """formula_to_dense_mass.

    Return formula including full compound mass

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position_mass[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position_mass["H"]))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def formula_to_dense_mass_norm(chem_formula: str) -> np.ndarray:
    """formula_to_dense_mass_norm.

    Return formula including full compound mass and normalized

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    dense_vec = formula_to_dense_mass(chem_formula)
    dense_vec = dense_vec / NORM_VEC_MASS

    return dense_vec


def formula_mass(chem_formula: str) -> float:
    """get formula mass"""
    mass = 0
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass


def formula_difference(formula_1, formula_2):
    """formula_1 - formula_2"""
    form_1 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_1)
    }
    form_2 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_2)
    }

    for k, v in form_2.items():
        if k in form_1:
            form_1[k] = form_1[k] - form_2[k]
        else:
            form_1[k] = -form_2[k]

    out_formula = "".join([f"{k}{v}" for k, v in form_1.items() if v != 0])
    return out_formula


def standardize_form(i):
    return vec_to_formula(formula_to_dense(i))


def get_mol_from_structure_string(structure_string, structure_type):
    if structure_type == "InChI":
        mol = canonical_mol_from_inchi(structure_string)
    else:
        mol = Chem.MolFromSmiles(structure_string)
    return mol


def vec_to_formula(form_vec):
    """vec_to_formula."""
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def calc_structure_string_type(structure_string):
    """calc_structure_string_type.

    Args:
        structure_string:
    """
    structure_type = None
    if pd.isna(structure_string):
        structure_type = "empty"
    elif structure_string.startswith("InChI="):
        structure_type = "InChI"
    elif Chem.MolFromSmiles(structure_string) is not None:
        structure_type = "Smiles"
    return structure_type


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "inchi":
        mol = Chem.MolFromInchi(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def canonical_mol_from_inchi(inchi):
    """Canonicalize mol after Chem.MolFromInchi
    Note that this function may be 50 times slower than Chem.MolFromInchi"""
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    if _RD_TAUTOMER_CANONICALIZER == 'v1':
        _molvs_t = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
        mol = _molvs_t.canonicalize(mol)
    else:
        _te = TautomerEnumerator()
        mol = _te.Canonicalize(mol)
    return mol


def form_from_smi(smi: str) -> str:
    """form_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    return uncharged_formula(smi, mol_type="smiles")


def form_from_inchi(inchi: str) -> str:
    """form_from_inchi.

    Args:
        inchi (str): inchi

    Return:
        str
    """
    return uncharged_formula(inchi, mol_type="inchi")


def _mol_from_types(mol, mol_type):
    if mol_type == 'smi':
        mol = Chem.MolFromSmiles(mol)
    elif mol_type == 'inchi':
        mol = canonical_mol_from_inchi(mol)
    elif mol_type == 'mol':
        mol = mol
    else:
        raise ValueError(f"Unknown mol_type={mol_type}")
    return mol

def rm_stereo(mol: str, mol_type='smi') -> str:
    mol = _mol_from_types(mol, mol_type)

    if mol is None:
        return
    else:
        Chem.RemoveStereochemistry(mol)

    if mol_type == 'smi':
        return Chem.MolToSmiles(mol)
    elif mol_type == 'inchi':
        return Chem.MolToInchi(mol)
    else:
        return mol

def inchikey_from_smiles(smi: str) -> str:
    """inchikey_from_smiles.

    Args:
        smi (str): smi

    Returns:
        str:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchiKey(mol)


def inchi_from_smiles(smi: str) -> str:
    """inchi_from_smiles.

    Args:
        smi (str): smi

    Returns:
        str:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchi(mol)


def smi_inchi_round_mol(smi: str) -> Chem.Mol:
    """smi_inchi_round.

    Args:
        smi (str): smi

    Returns:
        mol:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    inchi = Chem.MolToInchi(mol)
    if inchi is None:
        return None

    mol = canonical_mol_from_inchi(inchi)
    return mol


def smiles_from_inchi(inchi: str) -> str:
    """smiles_from_inchi.

    Args:
        inchi (str): inchi

    Returns:
        str:
    """
    mol = canonical_mol_from_inchi(inchi)
    if mol is None:
        return ""
    else:
        return Chem.MolToSmiles(mol)


def mass_from_inchi(inchi: str) -> float:
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def mass_from_smi(smi: str) -> float:
    """mass_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def has_valid_els(chem_formula: str) -> bool:
    """has_valid_els"""
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        if chem_symbol not in VALID_ELEMENTS:
            return False
    return True


def npclassifier_query(smiles):
    """npclassifier_query _summary_

    Args:
        smiles (_type_): _description_

    Returns:
        _type_: _description_
    """
    import requests

    ikey = inchikey_from_smiles(smiles)
    endpoint = "https://npclassifier.ucsd.edu/classify"
    req_data = {"smiles": smiles}
    out = requests.get(f"{endpoint}", data=req_data)
    out.raise_for_status()
    out_json = out.json()
    return {ikey: out_json}


def get_collision_energy(filename):
    colli_eng = re.findall('collision +([0-9]+\.?[0-9]*|nan).*', filename)
    if len(colli_eng) > 1:
        raise ValueError(f'Multiple collision energies found in {filename}')
    if len(colli_eng) == 1:
        colli_eng = colli_eng[0].split()[-1]
    else:
        colli_eng = 'nan'
    return colli_eng


def is_charged(mol, mol_type='smi') -> bool:
    """check if the entire molecule has imbalanced charge"""
    mol = _mol_from_types(mol, mol_type)
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 0


def has_separate_components(mol, mol_type='smi') -> bool:
    """is salt"""
    mol = _mol_from_types(mol, mol_type)
    return len(Chem.GetMolFrags(mol)) > 1


def has_isotopes(mol, mol_type='smi') -> bool:
    mol = _mol_from_types(mol, mol_type)
    return any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())


def has_unsupported_elems(mol, mol_type='smi') -> bool:
    mol = _mol_from_types(mol, mol_type)
    return not all(atom.GetSymbol() in VALID_ELEMENTS for atom in mol.GetAtoms())


def collision_energy_to_float(colli_eng):
    if isinstance(colli_eng, str):
        return float(colli_eng.split()[0])
    else:
        return float(colli_eng)


def sanitize(mol_list: List[Chem.Mol], mol_type='mol', return_indices=False) -> List[Chem.Mol]:
    """sanitize a list of mols"""
    new_mol_list = []
    new_idx_list = []
    for idx, mol in enumerate(mol_list):
        if mol is None:
            continue

        if mol_type == 'mol':
            pass
        elif mol_type == 'smi':
            mol = Chem.MolFromSmiles(mol)
        elif mol_type == 'inchi':
            mol = Chem.MolFromInchi(mol)
        else:
            raise ValueError(f'Unknown mol_type: {mol}')

        if mol is None:
            continue

        try:
            inchi = Chem.MolToInchi(mol)
            mol = Chem.MolFromInchi(inchi)
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            inchi = Chem.MolToInchi(mol)
            mol = canonical_mol_from_inchi(inchi)
            if mol is not None:
                new_mol_list.append(mol)
                new_idx_list.append(idx)
        except ValueError:
            logging.warning(f"Bad smiles")

    if mol_type == 'smi':
        new_mol_list = [Chem.MolToSmiles(mol) for mol in new_mol_list]
    elif mol_type == 'inchi':
        new_mol_list = [Chem.MolToInchi(mol) for mol in new_mol_list]

    if return_indices:
        return new_mol_list, new_idx_list
    else:
        return new_mol_list
