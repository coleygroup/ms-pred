""" run_magma.py

Entry point into running magma program

"""
import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import copy
import itertools
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from rdkit import RDLogger

# Custom import
import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation


FRAGMENT_ENGINE_PARAMS = {"max_broken_bonds": 6, "max_tree_depth": 3}


def greedy_prune(
    fe: fragmentation.FragmentEngine, included_nodes: list, tree_nodes: list
):

    """greedy_prune.

    Multiple paths can be used to access each of the nodes in the graph. This
    can double or triple the total number of nodes in the tree. We must prune
    these in order to get better nodes.

    We use a greedy set cover at each depth, attempting to find useful parents

    Args:
        fe (FragmentEngine): Fragment engine
        included_nodes: List of included nodes in the tree to be pruned
        tree_nodes: Nodes that must be included in the output arboresence

    Return:
        List of new nodes to include
    """
    # Get a queue for each node's priority based upon how many children it has
    # included in the tree nodes
    tree_set = set(tree_nodes)

    if len(tree_set) == 0:
        return []

    # Sort
    included_nodes = sorted(included_nodes)

    # Add whether or not the node is a member of existing nodes
    node_priorities = np.zeros(len(included_nodes))
    # Hash to index
    hash_to_pos = dict(zip(included_nodes, np.arange(len(included_nodes))))

    output_mask = np.array([j in tree_set for j in included_nodes])
    node_priorities += output_mask
    incoming_edges, outgoing_edges = fe.export_edges_dict(included_nodes)

    # Get adj dicts
    entries = [fe.frag_to_entry[i] for i in included_nodes]

    # Tie breaking
    node_scores = [i["score"] for i in entries]
    max_broken = np.array([i["max_broken"] for i in entries])
    tree_depths = np.array([i["tree_depth"] for i in entries])
    highest_depth = max(tree_depths)

    # Loop over tree depth from leaves up
    for depth in range(highest_depth, 0, -1):

        cur_layer_inds = tree_depths == depth

        # Define all parents of current layer (do this in loop below)
        cover_options = np.zeros(len(cur_layer_inds)).astype(bool)

        # Mask to cover --> all nodes in the current row requiring covering
        to_cover = np.logical_and(output_mask, cur_layer_inds)

        # Update parent priorities based upon what's in to_cover
        for node in np.where(to_cover)[0]:
            hash_key = included_nodes[node]

            # Get all parents
            incoming_parents = incoming_edges[hash_key]
            for p in incoming_parents:
                p_pos = hash_to_pos[p]
                node_priorities[p_pos] += 1
                cover_options[p_pos] = True

        num_to_cover = np.sum(to_cover)
        while num_to_cover > 0:

            # Get all cover options
            max_score = np.max(node_priorities[cover_options])
            best_cands_mask = node_priorities == max_score
            best_cands_mask[~cover_options] = False
            best_cands = np.where(best_cands_mask)[0]

            # Break ties by choosing those with min score
            best_cands = sorted(best_cands, key=lambda x: node_scores[x])
            new_output = best_cands[0]

            # Add this node to the return set and make sure it comes up for nex
            # time node priorities are computed
            output_mask[new_output] = True

            # Find all nodes that this covers and remove them from to cover
            # Decrement scores for the parents above based upon added value
            for child in outgoing_edges[included_nodes[new_output]]:
                new_covered = hash_to_pos[child]
                to_cover[new_covered] = False

                # Decrement scores for parents not in the output already
                for p in incoming_edges[child]:
                    p_pos = hash_to_pos[p]
                    if not output_mask[p_pos]:
                        node_priorities[hash_to_pos[p]] -= 1

            # Remove cover options
            cover_options[new_output] = False
            num_to_cover = np.sum(to_cover)
    output_hashes = np.array(included_nodes)[output_mask].tolist()
    return output_hashes


def magma_augmentation(
    spec_name: str,
    spec_dir: Path,
    spec_to_smiles: dict,
    spec_to_adduct: dict,
    max_peaks: int,
    ppm_diff: float = 10,
    merge_specs: bool = True,
    debug: bool = False,
):
    """magma_augmentation.

    Args:
        spec_name (str): spec_name
        spec_dir (Path): spec_dir
        spec_to_smiles (dict): spec_to_smiles
        spec_to_adduct (dict): Spec to adduct
        max_peaks (int): max_peaks
        ppm_diff (float): Max diff ppm
        debug (bool)
    """
    tsv_return = {}
    tree_return = {}

    spec_h5 = common.HDF5Dataset(spec_dir)
    spec_readlines = spec_h5.read_str(spec_name).split('\n')
    meta, spectras = common.parse_spectra(spec_readlines)

    spec_name_clean = Path(spec_name).stem  # remove '.json'
    spectra_smiles = spec_to_smiles.get(spec_name_clean, None)
    spectra_adduct = spec_to_adduct.get(spec_name_clean, None)

    # Step 1 - Generate fragmentations inside fragmentation engine
    fe = fragmentation.FragmentEngine(mol_str=spectra_smiles, **FRAGMENT_ENGINE_PARAMS)

    # Outside try except loop
    if debug:
        fe.generate_fragments()
    else:
        try:
            fe.generate_fragments()
        except:
            print(f"Error with generating fragments for spec {spec_name_clean}")
            return

    # Step 2: Process spec and get comparison points
    # Read in file and filter it down
    spectra = common.process_spec_file(meta, spectras, merge_specs=merge_specs)
    if spectra is None:
        print(f"Error with generating fragments for spec {spec_name_clean}")
        return

    if merge_specs:
        spectra = {'collision nan': spectra}
    for colli_eng, spectrum in spectra.items():
        tsv_filename = f"{spec_name_clean}_{colli_eng}.magma"
        tree_filename = f"{spec_name_clean}_{colli_eng}.json"

        spectrum = common.max_inten_spec(
            spectrum, max_num_inten=max_peaks, inten_thresh=0.001
        )
        s_m, s_i = spectrum[:, 0], spectrum[:, 1]

        # Correct for s_m to its [M]+ or [M]- mass
        adjusted_m = s_m - common.ion2mass[spectra_adduct]  # charge on adduct
        adjusted_e_m = s_m + common.ELECTRON_MASS if common.is_positive_adduct(spectra_adduct) \
            else s_m - common.ELECTRON_MASS  # charge transferred

        # Step 3: Make all assignments
        frag_hashes, frag_inds, shift_inds, masses, scores = fe.get_frag_masses()

        # Argsort by bond breaking scores
        # Lower bond scores are better
        new_order = np.argsort(scores)
        frag_hashes, frag_inds, shift_inds, masses, scores = (
            frag_hashes[new_order],
            frag_inds[new_order],
            shift_inds[new_order],
            masses[new_order],
            scores[new_order],
        )
        ppm_diffs = np.minimum(
            np.abs(masses[None, :] - adjusted_m[:, None]) / masses[None, :] * 1e6,
            np.abs(masses[None, :] - adjusted_e_m[:, None]) / masses[None, :] * 1e6
        )  # num of peaks x num of frags

        # Need to catch _all_ equivalent fragments
        # How do I remove the symmetry problem at each step and avoid branching
        # trees for the same examples??
        min_ppms = ppm_diffs.min(-1)
        is_min = min_ppms[:, None] == ppm_diffs
        peak_mask = min_ppms < ppm_diff  # num of peaks

        # Step 4: Make exports
        # Now collect all inds and results
        # Also record a map from hash, hshift to the peak_info
        tsv_export_list = []
        hash_to_peaks = defaultdict(lambda: [])
        max_labeled_inten = 0
        for ind, was_assigned in enumerate(peak_mask):
            new_entry = {
                "mz_observed": s_m[ind],
                # "mz_corrected": adjusted_m[ind],
                "inten": s_i[ind],
                "ppm_diff": "",
                "frag_inds": "",
                "frag_mass": "",
                "frag_h_shift": "",
                "frag_base_form": "",
                "frag_hashes": "",
            }
            if was_assigned:
                # Find all the fragments that have min ppm tolerance
                matched_peaks = is_min[ind]
                min_inds = np.argwhere(matched_peaks).flatten()

                # Get min score for this assignment
                min_score = np.min(scores[min_inds])

                # Filter even further down to inds that have min score and min ppm
                min_score_ppm = min_inds[
                    np.argwhere(scores[min_inds] == min_score).flatten()
                ]

                frag_inds_temp = [frag_inds[temp_ind] for temp_ind in min_score_ppm]
                frag_masses_temp = [masses[temp_ind] for temp_ind in min_score_ppm]
                frag_hashes_temp = [frag_hashes[temp_ind] for temp_ind in min_score_ppm]
                shift_inds_temp = [shift_inds[temp_ind] for temp_ind in min_score_ppm]
                frag_entries_temp = [
                    fe.frag_to_entry[frag_hash] for frag_hash in frag_hashes_temp
                ]
                frag_forms_temp = [frag_entry["form"] for frag_entry in frag_entries_temp]

                str_join = lambda x: ",".join([str(xx) for xx in x])
                new_entry["ppm_diff"] = min_ppms[ind]
                new_entry["frag_inds"] = str_join(frag_inds_temp)
                new_entry["frag_hashes"] = ",".join(frag_hashes_temp)
                new_entry["frag_mass"] = str_join(frag_masses_temp)
                new_entry["frag_h_shift"] = str_join(shift_inds_temp)
                new_entry["frag_base_form"] = ",".join(frag_forms_temp)
                peak_info_base = {
                    "mz_observed": s_m[ind],
                    # "mz_corrected": adjusted_m[ind],
                    "inten": s_i[ind],
                    "ppm_diff": min_ppms[0],
                    "frag_mass": frag_masses_temp[0],
                }
                max_labeled_inten = max(max_labeled_inten, s_i[ind])
                for h, s, f in zip(frag_hashes_temp, shift_inds_temp, frag_forms_temp):
                    peak_info_ex = copy.deepcopy(peak_info_base)
                    peak_info_ex["frag_hash"] = h
                    peak_info_ex["frag_h_shift"] = s
                    peak_info_ex["frag_base_form"] = f
                    hash_to_peaks[h].append(peak_info_ex)

            tsv_export_list.append(new_entry)

        df = pd.DataFrame(tsv_export_list)
        df.sort_values(by="mz_observed", inplace=True)
        tsv_return[tsv_filename] = df.to_csv(sep="\t", index=None)

        # Build trees
        tree_nodes = [
            j for i in tsv_export_list for j in i["frag_hashes"].split(",") if len(j) > 0
        ]
        tree_nodes = list(set(tree_nodes))

        # Now do a breadth first search back on the tree via its parents
        explore_queue = copy.deepcopy(tree_nodes)
        explored = set()
        while len(explore_queue) > 0:
            new_explore = explore_queue.pop()
            explored.add(new_explore)

            # Get parents for current node and add all of them
            entry = fe.frag_to_entry[new_explore]
            parent_hashes = entry["parent_hashes"]

            # Note: Parents are singular, but each parent has multiple potential
            explore_queue.extend(set([i for i in parent_hashes if i not in explored]))

        included_nodes = list(explored)
        pruned_nodes = greedy_prune(fe, included_nodes, tree_nodes)

        # Export and use to construct tree viz or others
        out_frags = {}
        pruned_node_set = set(pruned_nodes)
        node_to_pulled = defaultdict(lambda: set())
        node_pulled_sib = defaultdict(lambda: set())
        node_to_parents = defaultdict(lambda: set())
        for frag in pruned_nodes:
            entry = fe.frag_to_entry[frag]

            peak_intens = hash_to_peaks[frag]
            inten_vec = np.zeros(fe.shift_bucket_inds.shape[0])
            for i in peak_intens:
                # Do not renormalize!
                inten_vec[i["frag_h_shift"]] = i["inten"]  # / max_labeled_inten

            new_entry = {
                "frag_hash": frag,
                "frag": entry["frag"],
                "is_observed": frag in tree_nodes,
                "atoms_pulled": [],
                "parents": [],
                "base_mass": entry["base_mass"],
                "intens": inten_vec.tolist(),
                "id": entry["id"],
                "sib": False,
                "max_broken": entry["max_broken"],
                "tree_depth": entry["tree_depth"],
                "max_remove_hs": entry["max_remove_hs"],
                "max_add_hs": entry["max_add_hs"],
            }
            out_frags[frag] = new_entry
            for parent, pulled_atom, sibling_hash in zip(
                entry["parent_hashes"], entry["parent_ind_removed"], entry["sibling_hashes"]
            ):
                if parent in pruned_node_set:
                    node_to_parents[frag].add(parent)
                    node_to_pulled[parent].add(pulled_atom)
                    node_pulled_sib[(parent, pulled_atom)] = sibling_hash

        # Build up a list of (node, pulled) tuples and have a dict of pulled to
        # sibling
        out_frags_keys = list(out_frags.keys())
        for k in out_frags_keys:
            out_frags[k]["atoms_pulled"] = node_to_pulled[k]
            out_frags[k]["parents"] = node_to_parents[k]

            # Add in siblings for all pulled nodes!
            for a in out_frags[k]["atoms_pulled"]:
                sib_entries = node_pulled_sib[(k, a)]
                for sib_node in sib_entries:
                    fe_entry = fe.frag_to_entry[sib_node]

                    cur_entry = out_frags.get(sib_node)
                    # Make a new sib entry
                    if cur_entry is None:
                        inten_vec = np.zeros(fe.shift_bucket_inds.shape[0])
                        new_entry = {
                            "frag_hash": sib_node,
                            "frag": fe_entry["frag"],
                            "is_observed": False,
                            "atoms_pulled": [],
                            "parents": [k],
                            "base_mass": fe_entry["base_mass"],
                            "intens": inten_vec.tolist(),
                            "id": fe_entry["id"],
                            "sib": True,
                            "max_broken": fe_entry["max_broken"],
                            "tree_depth": fe_entry["tree_depth"],
                            "max_remove_hs": fe_entry["max_remove_hs"],
                            "max_add_hs": fe_entry["max_add_hs"],
                        }
                        out_frags[sib_node] = new_entry

                    # If we already have a non sib entry, continue
                    else:
                        if k not in cur_entry["parents"]:
                            cur_entry["parents"].append(k)

            out_frags[k]["atoms_pulled"] = list(out_frags[k]["atoms_pulled"])
            out_frags[k]["parents"] = list(out_frags[k]["parents"])

        export_tree = {
            "root_canonical_smiles": fe.smiles,
            "frags": out_frags,
            "collision_energy": float(colli_eng.split()[1]),
            "adduct": spectra_adduct,
        }
        # Export files when needed
        if len(export_tree["frags"]) > 0:
            tree_return[tree_filename] = json.dumps(export_tree, indent=2)

    return tsv_return, tree_return

def run_magma_augmentation(
    spectra_dir: str,
    output_dir: str,
    spec_labels: str,
    max_peaks: int,
    workers: int,
    debug: bool = False,
    ppm_diff: int = 20,
):
    """run_magma_augmentation.

    Runs magma augmentation

    Args:
        spectra_dir (str): spectra_dir
        output_dir (str): output_dir
        spec_labels (str): spec_labels
        max_peaks (int): max_peaks
        ppm_diff (int): PPM diff threshold
    """
    logging.info("Create magma spectra files")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    spectra_dir = Path(spectra_dir)
    spec_h5 = common.HDF5Dataset(spectra_dir)
    ms_files = spec_h5.get_all_names()

    # Read in spec to smiles
    df = pd.read_csv(spec_labels, sep="\t")
    spec_to_smiles = dict(df[["spec", "smiles"]].values)
    spec_to_adduct = dict(df[["spec", "ionization"]].values)

    # Run this over all files
    partial_aug_safe = lambda param: magma_augmentation(
        param['spec_file'],
        spectra_dir,
        spec_to_smiles,
        spec_to_adduct,
        max_peaks=max_peaks,
        debug=debug,
        ppm_diff=ppm_diff,
        merge_specs=param['merge_specs'],
    )
    # always include merged & unmerged specs in the dataset
    # params = [{'spec_file': f, 'merge_specs': m} for f, m in itertools.product(ms_files, [True, False])]
    # OR: only unmerged specs in the dataset
    params = [{'spec_file': f, 'merge_specs': m} for f, m in itertools.product(ms_files, [False])]

    if debug:
        write_objs = [partial_aug_safe(i) for i in tqdm(params[:10])]
    else:
        write_objs = common.chunked_parallel(params, partial_aug_safe, max_cpu=workers, chunks=1000)

    spec_h5.close()
    logging.info('Write to hdf5 file output')
    tsv_h5 = common.HDF5Dataset(output_dir / "magma_tsv.hdf5", mode='w')
    tree_h5 = common.HDF5Dataset(output_dir / "magma_tree.hdf5", mode='w')
    for obj in tqdm(write_objs):
        tsv_h5.write_dict(obj[0])
        tree_h5.write_dict(obj[1])
    tsv_h5.close()
    tree_h5.close()


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spectra-dir",
        default="data/spec_datasets/canopus_train_public/spec_files",
        help="Directory where spectra are stored",
    )
    parser.add_argument(
        "--spec-labels",
        default="data/spec_datasets/canopus_train_public/labels.tsv",
        help="TSV Location containing spectra labels",
    )
    parser.add_argument(
        "--output-dir",
        default="data/spec_datasets/canopus_train_public/magma_outputs",
        help="Output directory to save MAGMA files",
    )
    parser.add_argument(
        "--max-peaks",
        default=20,
        help="Maximum number of peaks",
        type=int,
    )
    parser.add_argument(
        "--ppm-diff",
        default=20,
        help="PPM threshold difference",
        type=int,
    )
    parser.add_argument(
        "--workers",
        default=16,
        help="Number of parallel workers",
        type=int,
    )
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # Define basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    RDLogger.DisableLog("rdApp.*")
    args = get_args()
    kwargs = args.__dict__
    run_magma_augmentation(**kwargs)
