""" Make formula subsets

Process pubchem smiles subsets.

"""
import pandas as pd
from typing import List, Tuple
import pickle
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem

import ms_pred.common as common


def read_smi_txt(smi_file, debug=False):
    """Load in smiles txt file with one smiles per line"""
    smi_list = []
    with open(smi_file, "r") as fp:
        for index, line in enumerate(fp):
            line = line.strip()
            if line:
                smi = line.split("\t")[1].strip()
                smi_list.append(smi)
            if debug and index > 10000:
                return smi_list
    return smi_list


def calc_formula_to_moltuples(smi_list: List[str]) -> dict:
    """Map smiles to their formula + inchikey"""
    output_list = common.chunked_parallel(smi_list, single_form_from_smi)
    formulae, mol_tuples = zip(*output_list)

    outdict = defaultdict(lambda: set())
    for mol_tuple, formula in tqdm(zip(mol_tuples, formulae)):
        outdict[formula].add(mol_tuple)
    return dict(outdict)


def single_form_from_smi(smi: str) -> Tuple[str, Tuple[str, str]]:
    """Compute single formula + inchi key from a smiles string"""
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is not None:
            form = common.uncharged_formula(mol)

            # first remove stereochemistry
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            inchi_key = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))

            return form, (smi, inchi_key)
        else:
            return "", ("", "")
    except:
        return "", ("", "")


def build_form_map(smi_file, dump_file=None, debug=False):
    """build_form_map."""
    smi_list = read_smi_txt(smi_file, debug=debug)
    form_to_mols = calc_formula_to_moltuples(smi_list)

    if dump_file is not None:
        with open(dump_file, "wb") as f:
            pickle.dump(form_to_mols, f)

    return form_to_mols


if __name__ == "__main__":
    pubchem_file = "data/retrieval/pubchem/pubchem_full.txt"
    data_labels = "data/spec_datasets/nist20/labels.tsv"
    pubchem_out = "data/retrieval/pubchem/pubchem_formula_map.p"
    pubchem_sub_out = "data/retrieval/pubchem/pubchem_formula_map_subset.p"
    built_map = build_form_map(
        smi_file=pubchem_file, dump_file=pubchem_out, debug=False
    )

    full_map = pickle.load(open(pubchem_out, "rb"))
    uniq_forms = pd.unique(pd.read_csv(data_labels, sep="\t")["formula"])
    sub_map = {uniq_form: full_map.get(uniq_form, []) for uniq_form in uniq_forms}

    with open(pubchem_sub_out, "wb") as f:
        pickle.dump(sub_map, f)
