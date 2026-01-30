from pathlib import Path
import json
import subprocess
from platformdirs import user_cache_dir
from difflib import get_close_matches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit import Chem
from rdkit.Chem import Draw
import seaborn as sns
from typing import List

import ms_pred.common as common
from ms_pred.retrieval.retrieval_benchmark import dist_bin

from ms_pred.dag_pred.iceberg_elucidation import load_pred_spec

def analyze_fragmentation_patterns(
    result_path: str,
    smiles: str,
    collision_energies: List[str],
    normalized_collision_energies: bool = True,
    file_name: str = "explained_peaks",
    save_path: str = "./",
):
    """
    Plot predicted spectra for different collision energies in a single figure with subplots,
    explaining the peaks using ICEBERG predictions.

    Args:
        result_path (str): Path to the ICEBERG prediction results.
        pmz (float): Precursor m/z value.
        smiles (str): SMILES string of the molecule.
        collision_energies (list[str]): List of collision energies to plot.
        normalized_collision_energies (bool): Whether the collision energies are normalized.
        file_name (str): Name of the output file (without extension).
        save_path (str): Path to save the output figure.
        num_peaks (int): Number of top peaks to explain.
    """
    from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
    import ms_pred.magma.fragmentation as fragmentation

    sns.set_theme(context="talk", style="white")

    smiles, pred_specs, pred_frags = load_pred_spec(result_path, normalized_collision_energies)
    engine = fragmentation.FragmentEngine(smiles[0], mol_str_type='smiles')

    num_energies = len(collision_energies)
    fig, axs = plt.subplots(num_energies, 1, figsize=(12, 6 * num_energies), sharex=True)
    if num_energies == 1:
        axs = [axs]

    fragmentation_data = []

    pmz = common.mass_from_smi(smiles[0])
    if normalized_collision_energies:
        collision_energies = [round(common.nce_to_ev(float(ce), pmz)) for ce in collision_energies]

    for i, ce in enumerate(collision_energies):
        pred_spec = pred_specs[0][str(ce)]
        pred_frag = pred_frags[0][str(ce)]
        
        # Extract fragmentation information
        ce_fragmentation_info = extract_fragmentation_info(pred_spec, pred_frag, engine)
        for info in ce_fragmentation_info:
            for fragment in info['fragments']:

                fragmentation_data.append({
                    'collision_energy': ce,
                    'smiles': smiles[0],
                    'smarts': fragment['full_mol_smarts'],
                    'mz': info['mz'],
                    'intensity': info['intensity'],
                    'bond_indices': tuple(fragment['broken_bonds']),
                    'fragment_smarts': fragment['fragment_smarts'],
                    'complement_smarts': fragment['complement_smarts'],
                })

        axs[i].set_title(f'{"Normalized " if normalized_collision_energies else ""}Collision Energy: {ce} eV')
        if i == num_energies - 1:
            axs[i].set_xlabel('m/z')
        axs[i].set_ylabel('Relative Intensity')

    # Create DataFrame from fragmentation data
    df_unique = pd.DataFrame(fragmentation_data)

    # Convert bond_indices and bonding_atom_indices back to lists for easier reading/processing
    df_unique['bond_indices'] = df_unique['bond_indices'].apply(list)
    
    # sort according to ce and then mz
    df_unique = df_unique.sort_values(by=['collision_energy', 'mz'])

    # Save DataFrame to CSV
    df_unique.to_csv(f'{save_path}/{file_name}_fragmentation_data.csv', index=False)

    return df_unique


def get_broken_bonds(engine, frag: int) -> List[tuple]:
    """
    Find the indices of broken bonds in the original molecule compared to the fragment.

    Args:
        engine (FragmentEngine): Fragmentation engine object.
        frag (int): Fragment integer.

    Returns:
        list: List of tuples containing the atom indices of the broken bonds.
    """
    mol = engine.mol
    atoms_in_fragment = set(engine.get_present_atoms(frag)[0])
    broken_bonds = []
    
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # A bond is broken if exactly one of its atoms is in the fragment
        if (begin in atoms_in_fragment) != (end in atoms_in_fragment):
            broken_bonds.append((begin, end))

    return broken_bonds

def get_bond_symbol(bond_type) -> str:
    """Get the bond symbol for a given bond type."""
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return '-'
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        return '='
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        return '#'
    else:
        return '~'  # For any other bond type

def set_atom_mapping(mol) -> Chem.Mol:
    """Set atom mapping for the entire molecule."""
    inchi = Chem.MolToInchi(mol)
    mol = Chem.MolFromInchi(inchi)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def generate_fragment_smarts(engine, fragment) -> str:
    """
    Generate SMARTS string for a fragment of the molecule.

    Args:
        engine (FragmentEngine): Fragmentation engine object.
        fragment (int): Fragment integer.
    
    Returns:
        str: SMARTS string for the fragment.
    """

    mol = set_atom_mapping(engine.mol)
    atoms_to_keep, _ = engine.get_present_atoms(fragment)
    
    # Create a new molecule with only the atoms in the fragment
    fragment_mol = Chem.RWMol()
    atom_map = {}
    for idx in atoms_to_keep:
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())  # Preserve the original atom mapping
        new_atom_idx = fragment_mol.AddAtom(new_atom)
        atom_map[idx] = new_atom_idx
    
    # Add bonds
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in atom_map and end in atom_map:
            fragment_mol.AddBond(atom_map[begin], atom_map[end], bond.GetBondType())
    
    smarts = Chem.MolToSmarts(fragment_mol)
    
    return smarts

def extract_fragmentation_info(pred_spec, pred_frag, engine) -> List[dict]:
    """
    Extract fragmentation information from the ICEBERG predictions.

    Args:
        pred_spec (list[tuple]): List of tuples containing the predicted m/z and intensity values.
        pred_frag (list[int]): List of predicted fragment indices.
        engine (FragmentEngine): Fragmentation engine object.
    
    Returns:
        list[dict]: List of dictionaries containing the fragmentation information.
    """

    fragmentation_info = []
    
    # Set atom mapping for the full molecule once
    mol = set_atom_mapping(engine.mol)
    full_mol_smarts = Chem.MolToSmarts(mol)
    
    for spec, fragment in list(zip(pred_spec, pred_frag)):
        mz, intensity = spec

        broken_bonds = get_broken_bonds(engine, fragment)
        
        # Generate SMARTS for both the fragment and its complement
        fragment_smarts = generate_fragment_smarts(engine, fragment)
        complement_fragment = engine.get_root_frag() ^ fragment  # XOR to get complement
        complement_smarts = generate_fragment_smarts(engine, complement_fragment)
        
        fragmentation_info.append({
            'mz': mz,
            'intensity': intensity,
            'fragments': [{
                'broken_bonds': broken_bonds,
                'fragment_smarts': fragment_smarts,
                'complement_smarts': complement_smarts,
                'full_mol_smarts': full_mol_smarts
            }]
        })

    return fragmentation_info
