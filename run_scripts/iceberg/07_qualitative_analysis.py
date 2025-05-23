"""Script to submit ICEBERG calls, plot spectra and save fragmentation patterns."""

import datetime
import os
from typing import List, Dict

import pandas as pd
from rdkit import Chem

from ms_pred.dag_pred.iceberg_elucidation import (
    iceberg_prediction,
    load_global_config
)
from ms_pred.dag_pred.iceberg_extract_fragments import (
    analyze_fragmentation_patterns,
)

# Setup paths and environment
SAVE_PATH = f'results/dag_inten_nist20/split_1_rnd1/qualitative'
os.makedirs(SAVE_PATH, exist_ok=True)

# Configuration
# Please modify the configs in `configs/iceberg/iceberg_elucidation.yaml` accordingly.
# If your hostname is ``server1``, create a new entry named ``server1``, and specify all the parameters as shown in
# the example.
CONFIG = load_global_config()


def count_atoms(smiles: str) -> int:
    """Count number of atoms in a molecule from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float('inf')  # Return infinity for invalid SMILES
        return mol.GetNumAtoms()
    except:
        return float('inf')


def elucidate(
        smiles: str,
        collision_energies: List[int],
        file_name: str,
        save_path: str = SAVE_PATH,
        normalized_collision_energies: bool = True,
        num_peaks: int = 5,
        ) -> pd.DataFrame:
    """Run ICEBERG prediction and analysis for a single molecule."""
    print('=' * 50)
    print(f'Elucidating {file_name} with SMILES: {smiles}')
    print('=' * 50)

    # Run ICEBERG to predict spectra
    result_path, pmz = iceberg_prediction(
        candidate_smiles=[smiles],
        collision_energies=collision_energies,
        nce=normalized_collision_energies,
        save_path=save_path,
        **CONFIG
    )

    # Analyze fragmentation patterns
    return analyze_fragmentation_patterns(
        result_path=result_path,
        smiles=smiles,
        collision_energies=collision_energies,
        normalized_collision_energies=normalized_collision_energies,
        file_name=file_name,
        save_path=save_path,
    )


def get_nist_test_data() -> Dict[str, str]:
    """
    Read NIST test data from TSV file and return a dictionary mapping NIST IDs to SMILES strings.
    
    Returns:
        Dict[str, str]: Dictionary with NIST IDs as keys and SMILES strings as values
    """
    nist20_dataset_path = '/home/runzhong/ms-pred/data/spec_datasets/nist20/labels.tsv'
    
    nist_dict = {}
    inchikeys = []
    with open(nist20_dataset_path, 'r') as f:
        # Skip header if present
        next(f)  # Skip the header line
        
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) >= 7:  # Ensure we have enough fields
                inchikey = fields[6]
                precursor_type = fields[4]
                if inchikey in inchikeys:
                    continue
                if precursor_type != '[M+H]+':
                    continue
                inchikeys.append(inchikey)
                spec_id = fields[1]    # spec ID is in second column
                smiles = fields[5]     # SMILES string is in sixth column
                nist_dict[spec_id] = smiles

    return nist_dict


def filter_for_test_data(molecules: Dict[str, str]) -> Dict[str, str]:
    """Filter molecules to only include those in the NIST test data."""

    splits_file = "/home/runzhong/ms-pred/data/spec_datasets/nist20/splits/split_1.tsv"

    with open(splits_file, 'r') as f:

        next(f)

        for line in f:
            fields = line.strip().split('\t')
            spec_id = fields[0]
            split_assignment = fields[1]

            if split_assignment != 'test':
                # filter this molecule out
                molecules.pop(spec_id, None)

    return molecules


def main():
    # Analysis parameters

    collision_energies = [5, 10, 20, 40, 80, 100] # eV
    
    # If using NIST test data
    molecules = get_nist_test_data()
    molecules = filter_for_test_data(molecules)
    
    # Process all molecules
    for molecule_name, smiles in molecules.items():
        df = elucidate(
            smiles=smiles,
            collision_energies=collision_energies,
            file_name=molecule_name,
            save_path=SAVE_PATH,
            normalized_collision_energies=False,
            num_peaks=5
        )


if __name__ == '__main__':
    main()