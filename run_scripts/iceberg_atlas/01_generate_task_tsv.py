# Make sure you already run
# data_scripts/pubchem/01_download_smiles.sh
# data_scripts/pubchem/02_make_formula_subsets.py
# and there is a file data/retrieval/pubchem/pubchem_formula_map.p

import pickle
import pandas as pd
from ms_pred import common
from ms_pred.dag_pred.iceberg_elucidation import sanitize_smiles
from pathlib import Path


pickle_file = "data/retrieval/pubchem/pubchem_formula_map.p"
adduct = '[M+H]+'
nces = [10, 20, 30, 40, 50]
save_dir = "data/retrieval/pubchem/atlas/formula"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

nist_specs = pd.read_csv('data/spec_datasets/nist23/labels.tsv', sep='\t')
nist_ikeys2d = [_.split('-')[0] for _ in nist_specs['inchikey']]


def parse_entries(tup):
    form, smiles_tups = tup
    if not common.has_valid_els(form):
        return []
    pmz = common.formula_mass(form) + common.ion2mass[adduct]
    if pmz > 1500:
        return []

    smiles = []
    for smi, ikey in smiles_tups:
        if not ikey.split('-')[0] in nist_ikeys2d:
            smiles.append(smi)
    smiles = sanitize_smiles(smiles, form, canonicalize=False)
    collision_energies = [common.nce_to_ev(_, pmz) for _ in nces]

    entries = []
    for smi in smiles:
        entries.append({
            'spec': 'iceberg_atlas', 'smiles': smi, 'ionization': adduct,
            'inchikey': common.inchikey_from_smiles(smi),
            'precursor': pmz, 'collision_energies': collision_energies,
        })
    if len(entries) == 0:
        return []
    return entries


def write_tsv(all_entries):
    out_counter = 0
    write_buffer = []
    for entries in all_entries:
        for entry in entries:
            if len(write_buffer) >= 100000:
                df = pd.DataFrame.from_dict(write_buffer)
                df.to_csv(save_dir / f'pubchem_{out_counter}.tsv', sep='\t', index=False)
                out_counter += 1
                write_buffer = []
            write_buffer.append(entry)

    if len(write_buffer) > 0: # write the remaining
        df = pd.DataFrame.from_dict(write_buffer)
        df.to_csv(save_dir / f'pubchem_{out_counter}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    full_map = pickle.load(open(pickle_file, "rb"))
    common.chunked_parallel(list(full_map.items()), parse_entries, output_func=write_tsv, chunks=1000, max_cpu=64)
