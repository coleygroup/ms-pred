from pathlib import Path
import json
import subprocess

import matplotlib
from platformdirs import user_cache_dir
from difflib import get_close_matches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import pubchempy as pcp
import pickle
import copy
from typing import List
import pygmtools as pygm
import yaml
import socket
from collections.abc import Iterable

import ms_pred.common as common
from ms_pred.retrieval.retrieval_benchmark import dist_bin
from ms_pred import coster


def load_global_config(path_to_config='configs/iceberg/iceberg_elucidation.yaml', hostname=None):
    all_configs = yaml.safe_load(open(path_to_config, "r"))
    if hostname is None:
        hostname = socket.gethostname()
    if hostname in all_configs.keys():
        config = all_configs[hostname]
        if 'clone' in config.keys():
            config = all_configs[config['clone']]
    else:
        raise ValueError(f'Your current hostname {hostname} is not found in configuration file. Please add '
                         f'configurations to {path_to_config}')
    return config


def candidates_from_pubchem(
    formula:str,
    pubchem_form_map:str='data/retrieval/pubchem/pubchem_formulae_inchikey.hdf5',
):
    """
    Get structural candidates with the same chemical formula from pubchem

    Args:
        formula: chemical formula of interested
        pubchem_form_map: path to a pubchem mapping file

    Returns: list of SMILES

    """
    if pubchem_form_map[-5:] == '.hdf5':  # hdf5 file
        form_to_smi_mapping = common.HDF5Dataset(pubchem_form_map)
        def read_fn(obj, form):
            return obj.read_str(form)
    elif pubchem_form_map[-2:] == '.p':  # pickle file
        form_to_smi_mapping = pickle.load(open(pubchem_form_map, 'rb'))
        def read_fn(obj, form):
            return obj[form]
    else:
        raise ValueError('Unknown pubchem_form_map type. Supported: HDF5 (.hdf5) or Pickle (.p)')

    if formula in form_to_smi_mapping:
        smiles_inchikey = json.loads(read_fn(form_to_smi_mapping, formula))
        smiles = [_[0] for _ in smiles_inchikey]
    else:
        def smiles_from_pubchem(form):
            try:
                compounds = pcp.get_compounds(form, namespace='formula')
                return [cmpd.isomeric_smiles for cmpd in compounds]
            except pcp.BadRequestError:
                return []
        try:
            smiles = smiles_from_pubchem(formula)
        except pcp.ServerError: # retry
            smiles = smiles_from_pubchem(formula)

    target_mass = common.formula_mass(formula)

    # remove stereo chemistry, mass mismatch (due to isotopes) and duplicates
    smiles = [smi for smi in smiles if '.' not in smi]  # rm mixtures
    smiles = common.sanitize(smiles, 'smi')  # sanitize (remove weired structures)
    smiles = [common.rm_stereo(smi) for smi in smiles]  # rm stereo chemistry information
    smiles = [smi for smi in smiles if smi is not None]
    smiles = [smi for smi in smiles if np.abs(common.mass_from_smi(smi) - target_mass) < 0.01]  # rm mass mismatch (usually due to isotopes)
    smiles = np.array(smiles)
    charge = np.array([Chem.GetFormalCharge(Chem.MolFromSmiles(smi)) for smi in smiles])
    smiles = smiles[charge == 0]
    inchikey = [common.inchikey_from_smiles(smi) for smi in smiles]
    _, unique_ids = np.unique(inchikey, return_index=True)
    smiles = smiles[unique_ids]
    return smiles.tolist()


def iceberg_prediction(
    candidate_smiles:List[str],
    collision_energies:List[int],
    nce:bool=False,
    adduct:str='[M+H]+',
    exp_name:str='iceberg_elucidation',
    python_path:str='',
    gen_ckpt:str='',
    inten_ckpt:str='',
    cuda_devices:int=0,
    batch_size:int=8,
    num_workers:int=6,
    sparse_k:int=100,
    max_nodes:int=100,
    threshold:float=0.1,
    binned_out:bool=False,
    ppm:int=20,
    force_recompute:bool=False,
    **kwargs,
):
    """
    Run ICEBERG prediction over a candidate set of molecules

    Args:
        candidate_smiles: (List[str]) list of candidate SMILES
        collision_energies: (List[int]) list of collision energies. Could also be List[List[int]] for each candidate SMILES
        nce: (bool, default=False) if True, the collision energies are treated as normalized collision energy; otherwise, they are treated as absolute eV
        adduct: (str, default='[M+H]+') adduct type. Could also be List[str] for each candidate SMILES
        exp_name: (str, default='iceberg_elucidation') name of the experiment
        python_path: (str) path to python executable
        gen_ckpt: (str) path to ICEBERG generator model (model 1) checkpoint
        inten_ckpt: (str) path to ICEBERG intensity model (model 2) checkpoint
        cuda_devices: (int or list) CUDA visible devices. If None, ICEBERG will run on CPU
        batch_size: (int, default=8)
        num_workers: (int, default=6) number of parallel workers
        sparse_k: (int, default=100) number of unique peaks predictred by model 2
        max_nodes: (int, default=100) number of fragments generated by model 1
        threshold: (float, default=0.1) cutoff confidence for model 1
        binned_out: (bool, default=False) if True, model outputs binned spectrum; otherwise, model outputs high-precision m/z peaks
        ppm: (int, default=20) parts-per-million threshold for mass comparison
        force_recompute: (bool, default=False) if True, will always re-run ICEBERG no matter the result has been cached

    Returns: path to results (a temporary path that hashes all experiment parameters),
             precursor mass
    """
    #########################################
    #       Check & preprocess input        #
    #########################################

    python_path = Path(python_path)
    gen_ckpt = Path(gen_ckpt)
    inten_ckpt = Path(inten_ckpt)

    # cuda devices as string
    if isinstance(cuda_devices, Iterable):
        cuda_devices = ','.join([str(_) for _ in cuda_devices])
    elif isinstance(cuda_devices, int):
        cuda_devices = str(cuda_devices)
    elif cuda_devices is None:
        cuda_devices = ''
    assert isinstance(cuda_devices, str)

    if len(candidate_smiles) == 0:
        raise ValueError('Empty candidate list!')

    # check adduct type
    if isinstance(adduct, str):
        adducts = [adduct] * len(candidate_smiles)
    else:
        adducts = adduct
    for adduct in adducts:
        if not adduct in common.ion2mass:
            matches = get_close_matches(adduct, common.ion2mass.keys(), n=1, cutoff=0.7)
            if len(matches) > 0:
                raise ValueError(f'Unknown adduct {adduct}. Did you mean {matches[0]}? ')
            else:
                raise ValueError(f'Adduct type {adduct} is not supported. Supported adducts: '
                                 f'{list(common.ion2mass.keys())}')
    assert len(adducts) == len(candidate_smiles)

    # standardize collision energies
    if not common.is_iterable(collision_energies): # single value CE
        collision_energies = [[int(collision_energies)] for _ in adducts]
    elif not common.is_iterable(collision_energies[0]): # one set of CEs for all, list of ints
        collision_energies = [collision_energies for _ in adducts]
    else: # one set of CEs for each, list of lists
        collision_energies = collision_energies
    assert len(collision_energies) == len(candidate_smiles)
    # remove stereo
    candidate_smiles = [common.rm_stereo(smi) for smi in candidate_smiles]

    # remove duplicate inchikeys
    inchikeys = [common.inchikey_from_smiles(smi) for smi in candidate_smiles]
    _, uniq_idx = np.unique(inchikeys, return_index=True)
    candidate_smiles = np.array(candidate_smiles)[uniq_idx].tolist()
    # candidate_smiles = [Chem.MolToSmiles(common.canonical_mol_from_inchi(common.inchi_from_smiles(smi)))
    #                     for smi in candidate_smiles]
    adducts = np.array(adducts)[uniq_idx].tolist()
    collision_energies = [collision_energies[i] for i in uniq_idx]

    # get formula & mass & check candidate smiles
    # precursor_mass = common.mass_from_smi(candidate_smiles[0]) + common.ion2mass[adduct]
    # # formula check is skipped
    # # formula = common.form_from_smi(candidate_smiles[0])
    # for smi in candidate_smiles:
    #     cur_pm = common.mass_from_smi(smi) + common.ion2mass[adduct]
    #     if np.abs(precursor_mass - cur_pm) > precursor_mass * ppm * 1e-6:  # ppm diff
    #         raise ValueError(f'Precursor mass mismatch in input molecules. Got {smi}, mass={cur_pm}, expected mass={precursor_mass} inferred from {candidate_smiles[0]}')
    #     # if not formula == common.form_from_smi(smi):
    #     #     raise ValueError(f'Formula mismatch in input molecules. Got {smi}, form={common.form_from_smi(smi)}, expected form={formula} inferred from {candidate_smiles[0]}')

    # handle collision energies, convert all nce to ev
    collision_energies = [[float(e) for e in ces] for ces in collision_energies]
    new_collision_energies = []
    precursor_masses = []
    for cand_smi, adduct, ces in zip(candidate_smiles, adducts, collision_energies):
        precursor_mass = common.mass_from_smi(cand_smi) + common.ion2mass[adduct]
        precursor_masses.append(precursor_mass)
        if nce:
            ces = [common.nce_to_ev(e, precursor_mass) for e in ces]
        else:
            ces = ces
        new_collision_energies.append([f'{float(_):.0f}' for _ in sorted(ces)])
    collision_energies = new_collision_energies

    # generate temp directory
    param_str = exp_name + '|'
    for cand_smi, adduct, ce in sorted(zip(candidate_smiles, adducts, collision_energies)):
        param_str += '|' + cand_smi + ';' + str(common.ion2onehot_pos[adduct]) + ';' + ','.join(sorted(ce))
    param_str += '||' + str(gen_ckpt.absolute()) + '||' + str(inten_ckpt.absolute()) + '||' + cuda_devices + \
                 '||' + f'{batch_size:d}-{num_workers:d}-{sparse_k:d}-{max_nodes:d}||' + f'{threshold:.2f}' + \
                 '||' + ('binned_out' if binned_out else "")
    param_hash = common.str_to_hash(param_str)
    save_dir = Path(user_cache_dir(f"ms-pred/iceberg-elucidation/{param_hash}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    #########################################
    #            Call ICEBERG               #
    #########################################

    # skip model call if the results are cached
    if force_recompute or not (save_dir / 'iceberg_run_successful').exists():
        # write candidates to tsv
        entries = []
        for cand_smi, adduct, ce, pmz in zip(candidate_smiles, adducts, collision_energies, precursor_masses):
            entries.append({
                'spec': exp_name, 'smiles': cand_smi, 'ionization': adduct, 'inchikey': common.inchikey_from_smiles(cand_smi),
                'precursor': pmz, 'collision_energies': ce,
            })
        df = pd.DataFrame.from_dict(entries)
        df.to_csv(save_dir / f'cands_df_{exp_name}.tsv', sep='\t', index=False)

        # run iceberg to generate in-silico spectrum
        cmd = (f'''{python_path} src/ms_pred/dag_pred/predict_smis.py \\
               --batch-size {batch_size} \\
               --num-workers {num_workers} \\
               --dataset-labels {save_dir / f"cands_df_{exp_name}.tsv"} \\
               --sparse-out \\
               --sparse-k {sparse_k} \\
               --max-nodes {max_nodes} \\
               --threshold {threshold} \\
               --gen-checkpoint {gen_ckpt} \\
               --inten-checkpoint {inten_ckpt} \\
               --save-dir {save_dir} \\
               --adduct-shift''')
        if cuda_devices:
            cmd = f'CUDA_VISIBLE_DEVICES={cuda_devices} ' + cmd + ' \\\n           --gpu'
        assert not binned_out, 'Elucidation not supported for binned_out=True'
        if binned_out:
            cmd += ' \\           --binned_out'
        print(cmd)
        run_result = subprocess.run(cmd, shell=True)
        if run_result.returncode == 0:  # successful
            (save_dir / 'iceberg_run_successful').touch()

    precursor_mass = np.unique(np.array(precursor_masses).round(decimals=6)).tolist()
    if len(precursor_mass) == 1:
        precursor_mass = precursor_mass[0]

    return save_dir, precursor_mass


def load_real_spec(
    real_spec:str,
    real_spec_type:str,
    precursor_mass:float=None,
    nce:bool=False,
    ppm:int=20,
    nist_path:str='data/spec_datasets/nist20/spec_files.hdf5',
    denoise_spectrum:bool=True,
    intensity_threshold:float=0.05,
    **kwargs,
):
    if real_spec_type == 'raw':
        meta = {}
        real_spec = [(k, v) for k, v in real_spec.items()]
    elif real_spec_type == 'ms':
        real_spec_path = Path(real_spec)
        meta, real_spec = common.parse_spectra(real_spec_path)
    elif real_spec_type == 'nist':
        nist_h5 = common.HDF5Dataset(nist_path)
        real_spec = nist_h5.read_str(f"{real_spec}.ms").split("\n")
        meta, real_spec = common.parse_spectra(real_spec)
    else:
        raise ValueError(f'Unkown spectrum type {real_spec_type}')

    if 'parentmass' in meta:
        if precursor_mass is not None:
            precursor_mass = common.merge_mz(precursor_mass, ppm)
            # check if meta is matched
            if np.abs(precursor_mass - float(meta['parentmass'])) > precursor_mass * ppm * 1e-6:
                raise ValueError(f'Precursor mass is different from loaded spectrum metadata! Got m/z={precursor_mass}, loaded from spec={meta["parentmass"]}')
        else:
            precursor_mass = float(meta['parentmass'])
    assert precursor_mass is not None

    # denoise spectrum (thresholding)
    if denoise_spectrum:
        real_spec = [(k, common.max_inten_spec(v, max_num_inten=20, inten_thresh=intensity_threshold)) for k, v in real_spec]
        real_spec = [(k, common.electronic_denoising(v)) for k, v in real_spec]

    real_spec = common.process_spec_file(meta, real_spec, merge_specs=False)

    # round collision energy to integer
    real_spec = {float(common.get_collision_energy(k)): v for k, v in real_spec.items()}
    if nce:
        real_spec = {common.nce_to_ev(k, precursor_mass): v for k, v in real_spec.items()}
    real_spec = {f'{float(k):.0f}': v for k, v in real_spec.items()}

    return real_spec


def load_pred_spec(
    load_dir:str,
    merge_spec:bool,
    merge_method='sum',
):
    """
    Args:
        load_dir: str
        merge_spec: bool

    Returns:

    """
    load_dir = Path(load_dir)

    pred_specs = common.HDF5Dataset(load_dir / 'preds.hdf5')
    pred_spec_ars = []
    pred_smis = []
    pred_frags = []
    # iterate over h5 layers
    for pred_spec_obj in pred_specs.h5_obj.values():
        for smiles_obj in pred_spec_obj.values():
            smi = None
            spec_dict = {}
            frag_dict = {}
            for collision_eng_key, collision_eng_obj in smiles_obj.items():
                if smi is None:
                    smi = collision_eng_obj.attrs['smiles']
                collision_eng_key = common.get_collision_energy(collision_eng_key)
                spec_dict[collision_eng_key] = collision_eng_obj['spec'][:]
                frag_dict[collision_eng_key] = json.loads(collision_eng_obj['frag'][0])

            if merge_spec:
                mz_frag_to_tup = {}
                for collision_eng_key in spec_dict.keys():
                    for spec, frag in zip(spec_dict[collision_eng_key], frag_dict[collision_eng_key]):
                        mz, inten = spec
                        mz_frag = f'{mz:.4f}_{frag}'
                        cur_tup = mz_frag_to_tup.get(mz_frag)
                        if cur_tup is None:
                            mz_frag_to_tup[mz_frag] = [mz, inten, frag]
                        else:
                            if merge_method == 'sum':
                                cur_tup[1] += inten
                            elif merge_method == 'max':
                                cur_tup[1] = max(inten, cur_tup[1])
                            else:
                                raise ValueError(f'Unknown merge_method {merge_method}')

                merged_spec, merged_frag = [], []
                for tup in mz_frag_to_tup.values():
                    merged_spec.append((tup[0], tup[1]))
                    merged_frag.append(tup[2])
                merged_spec = np.array(merged_spec)
                merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()
                spec_dict, frag_dict = {}, {}
                spec_dict['nan'] = merged_spec  # 'nan' means merged
                frag_dict['nan'] = np.array(merged_frag)

            pred_spec_ars.append(spec_dict)
            pred_frags.append(frag_dict)
            pred_smis.append(smi)
    pred_specs.close()
    pred_specs = np.array(pred_spec_ars)
    smiles = np.array(pred_smis)

    return smiles, pred_specs, pred_frags


def elucidation_over_candidates(
    load_dir:str,
    real_spec:str,
    real_spec_type:str='ms',
    precursor_mass:float=None,
    nce:bool=False,
    step_collision_energy:bool=False,
    mol_name:str="",
    real_smiles:str=None,
    topk:int=10,
    ppm:int=20,
    num_bins:int=15000,
    ignore_precursor:bool=True,
    dist_func:str='entropy',
    **kwargs,
):
    """
    Run elucidation over a candidate set

    Args:
        load_dir: (str) path to result directory (return of function iceberg_prediction)
        real_spec: (str) the real spectrum. Depends on real_spec_type
        real_spec_type: (str, default='ms') 'ms': SIRIUS-style spectrum file (.ms), 'raw': processed dictionary
        precursor_mass: (float) mass of the precursor ion
        nce: (bool, default=False) if True, the collision energies are treated as normalized collision energy; otherwise, they are treated as absolute eV
        step_collision_energy: (bool, default=False) if True, it means step_collision_energy is turned on in the instrument and only one merged spectrum will be returned
        mol_name: (str, default="") name of the molecule and/or experiment
        real_smiles: (str, default=None) the real SMILES, if specified
        topk: (int, default=10) number of candidates returned
        ppm: (int, default=20) parts-per-million threshold for mass comparison
        num_bins: (int, default=15000) number of bins for binned spectrum
        ignore_precursor: (bool, default=True) ignore precursor peak
        dist_func: (str, default='entropy') distance function

    Returns: list of TopK molecules:
        [ (top1 SMILES, entropy distance, true molecule or not),
          (top2 SMILES, entropy distance, true molecule or not),
          ...
          (topK SMILES, entropy distance, true molecule or not),
        ]

    """
    # hack the precursor mz if there are multiple formulae within tolerance
    precursor_mass = common.merge_mz(precursor_mass, ppm)

    real_spec = load_real_spec(real_spec, real_spec_type, precursor_mass, nce, ppm)
    smiles, pred_specs, pred_frags = load_pred_spec(load_dir, step_collision_energy)

    # transform spec to binned spectrum
    real_binned = {k: common.bin_spectra([v], num_bins)[0] for k, v in real_spec.items()}
    pred_binned_specs = [
        {k: common.bin_spectra([v], num_bins, pool_fn='add')[0] for k, v in s.items()}
        for s in pred_specs]

    # get target inchikey (if any)
    if real_smiles is not None:
        target_inchikey = common.inchikey_from_smiles(common.rm_stereo(real_smiles))
    else:
        target_inchikey = None

    # compute distance
    dist = dist_bin(pred_binned_specs, real_binned, ignore_peak=(precursor_mass - 1) * 10 if ignore_precursor else None,
                    sparse=False, func=dist_func)

    sorted_indices = np.argsort(dist)

    found_true = False
    true_idx = -1
    for rnk, idx in enumerate(sorted_indices):
        d = dist[idx]
        smi = smiles[idx]
        inchikey = common.inchikey_from_smiles(smi)
        if target_inchikey is not None and inchikey in target_inchikey:
            print((f'[{mol_name}] ' if len(mol_name) > 0 else "") + f'Found target mol at {rnk+1}/{len(sorted_indices)}, ent_dist={d:.3f}')
            found_true = True
            true_idx = idx
        if idx >= topk and found_true:
            break
    return [(smiles[i], dist[i], i == true_idx) for i in sorted_indices[:topk]]


def plot_top_mols(
    topk_results,
    sa_score=False,
):
    """
    Turn the output of elucidation_over_candidates into plot

    Args:
        topk_results:

    Returns:
        an image object
    """
    mols = []
    legends = []
    for rnk, (smi, dist, is_true) in enumerate(topk_results):
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol)
        legend_str = f"top{rnk+1} ent_dis={dist:.3f}"
        if sa_score:
            sa = sascorer.calculateScore(mol)
            legend_str += f" SA={sa:.3f}"
        if is_true:
            legend_str += '\ntrue molecule'
        legends.append(legend_str)
    return Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250,250), legends=legends, maxMols=len(mols))


def explain_peaks(
    load_dir:str,
    real_spec:str,
    precursor_mass:float,
    mol_str:str,
    real_spec_type:str='ms',
    pred_label='Predicted',
    real_label='Experimental',
    mol_type:str='smi',
    num_peaks:int=5,
    nce:bool=False,
    merge_spec:bool=False,
    ppm:int=20,
    save_path:str=None,
    axes:list=None,
    display_ce:bool=True,
    display_expmass:bool=True,
    display_mass_inten_thresh:float=0.3,
    dpi:int=500,
    **kwargs,
):
    """
    Plot experiment spectrum and predicted spectrum head-to-head and explain the peaks using ICEBERG predictions

    Args:
        load_dir: (str) path to result directory (return of function iceberg_prediction)
        real_spec: (str) path to the experimental spectrum (.ms file)
        precursor_mass: (float) mass of the precursor ion
        mol_str: (str) the molecule of interest
        real_spec_type: (str, default='ms') 'ms': SIRIUS-style spectrum file (.ms), 'raw': processed dictionary
        pred_label: (str, default='Predicted') label for the predicted spectrum
        real_label: (str, default='Experimental') label for the experimental spectrum
        mol_type: (str, default='smi') type of mol_str. Supported values: 'smi' (SMILIES), 'inchi' (InChi), 'mol' (RDKit molecule objedt), 'inchikey' (InChiKey)
        num_peaks: (int, default=5) number of peaks to explain
        nce: (bool, default=False) if True, the collision energies are treated as normalized collision energy; otherwise, they are treated as absolute eV
        merge_spec: (bool, default=False) if True, spectra are merged
        ppm: (int, default=20) parts-per-million threshold for mass comparison
        save_path: (str) if specified, save result to the specified path
        axes: (list) matplotlib axis(es) for the plots
        display_ce: (bool, default=True) display collision energy
        display_expmass: (bool, default=True) display experiment mass values
        display_mass_inten_thresh: (float, default=0.05) when the intensity is over this threshold, display the m/z value in the plot
        dpi: (int, default=500) dpi of plots
    """
    from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
    import ms_pred.magma.fragmentation as fragmentation

    if 'step_collision_energy' in kwargs:
        merge_spec = kwargs['step_collision_energy']

    # hack the precursor mz if there are multiple formulae within tolerance
    precursor_mass = common.merge_mz(precursor_mass, ppm)

    # load predicted spec & get the interested prediction
    smiles, pred_specs, pred_frags = load_pred_spec(load_dir, merge_spec)
    cand_inchikeys = [common.inchikey_from_smiles(common.rm_stereo(smi)) for smi in smiles]

    # get the interested
    if mol_type == 'smi':
        inchikey_of_interest = common.inchikey_from_smiles(common.rm_stereo(mol_str, mol_type))
    elif mol_type == 'mol':
        inchikey_of_interest = Chem.MolToInchikey(common.rm_stereo(mol_str, mol_type))
    elif mol_type == 'inchi':
        inchikey_of_interest = Chem.InchiToInchikey(common.rm_stereo(mol_str, mol_type))
    elif mol_type == 'inchikey':
        inchikey_of_interest = mol_str
    else:
        raise ValueError(f'mol_type={mol_type} is not supported')

    if not inchikey_of_interest in cand_inchikeys:
        print('No inchikey matching is found for the compound of interested')
        return

    idx = cand_inchikeys.index(inchikey_of_interest)
    smi = smiles[idx]
    pred_spec = pred_specs[idx]
    pred_frag = pred_frags[idx]
    engine = fragmentation.FragmentEngine(smi, mol_str_type='smiles')

    if real_spec is None or real_spec_type == 'none':  # only plot the predicted spec
        all_ces = list(pred_spec.keys())
        real_spec = None
    else:
        real_spec = load_real_spec(real_spec, real_spec_type, precursor_mass, nce, ppm)
        if merge_spec:
            real_spec = common.merge_specs(real_spec)
        all_ces = set(pred_spec.keys()).intersection(real_spec.keys())

    if axes is None:
        axes = [None] * len(all_ces)
    elif not common.is_iterable(axes):
        axes = [axes]
    if not len(axes) == len(all_ces):
        raise ValueError(f"shape mismatch. Expected {len(all_ces)} axes because we have {len(all_ces)} collision "
                         f"energies but got {len(axes)}. \n"
                         "Predicted spec CEs: " + ",".join([f'{_}eV' for _ in pred_spec.keys()]) + "\n" +
                         ("Experimental spec CEs: " + ",".join([f'{_}eV' for _ in real_spec.keys()]) + "\n" if real_spec else ""))

    for idx, ((_, ce), ax) in enumerate(zip(sorted(zip([float(ce) for ce in all_ces], all_ces)), axes)):
        if real_spec is None:  # plot only predicted spec
            common.plot_ms(pred_spec[ce], pred_label,
                           '' if np.isnan(float(ce)) or not display_ce else f'{ce} eV',
                           dpi=dpi, ax=ax, largest_mz=precursor_mass)
        else:  # plot two specs head-to-head
            common.plot_compare_ms(pred_spec[ce], real_spec[ce], pred_label, real_label,
                                   '' if np.isnan(float(ce)) or not display_ce else f'{ce} eV',
                                   dpi=dpi, ax=ax, largest_mz=precursor_mass)

        # display mass of real spectrum
        if display_expmass and real_spec is not None:
            mz_to_plot = dict()
            for mz, inten in real_spec[ce]:
                if mz.round(2) in mz_to_plot:
                    if mz_to_plot[mz.round(2)][1] < inten:
                        mz_to_plot[mz.round(2)] = mz, inten
                else:
                    mz_to_plot[mz.round(2)] = mz, inten
            for _, (mz, inten) in mz_to_plot.items():
                if inten > display_mass_inten_thresh:
                    plt.text(mz, -inten - 0.06, f'{mz:.4f}', fontsize=4, alpha=0.7, horizontalalignment='center')

        if ax is None:
            ax = plt.gca()

        # explain predicted spectrum
        counter = 0
        pred_spec[ce][:, 1] = pred_spec[ce][:, 1] / np.max(pred_spec[ce][:, 1])
        for spec, frag in sorted(zip(pred_spec[ce], pred_frag[ce]), key=lambda x: x[0][1], reverse=True): # sort by inten
            if counter >= num_peaks:
                break
            mz, inten = spec
            mol_offset = (mz, inten + 0.05)
            text_offset = (mz, inten + 0.2)
            draw_dict = engine.get_draw_dict(frag)
            common.plot_mol_as_vector(
                draw_dict["mol"], hatoms=draw_dict["hatoms"], hbonds=draw_dict["hbonds"],
                offset=mol_offset, zoom=0.001, ax=ax
            )
            plt.text(text_offset[0], text_offset[1], f'{mz:.4f}', fontsize=4, horizontalalignment='center')
            counter += 1

        # hide x-axis if not the last spec
        if idx < len(all_ces) - 1:
            ax.set_xticklabels([])
            ax.set_xticklabels([], minor=True)
            ax.set_xlabel("")

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    return plt.gcf()


def modi_finder(
    formula_diff:str,
    load_dir:str,
    mol_str1:str,
    name1:str,
    precursor_mass1:float,
    real_spec1:str,
    real_spec_type1:str,
    name2:str,
    precursor_mass2:float,
    real_spec2:str,
    real_spec_type2:str,
    mol_type1:str='smi',
    nce1:bool=False,
    nce2:bool=False,
    mz_cutoff:float=1500,
    topk_peaks:int=10,
    top_score_thresh:float=0.1,
    step_collision_energy:bool=False,
    ppm:int=20,
    save_path: str = None,
    axes: list = None,
):
    """
    ModiFinder find modification sites between two chemical compounds, whereby their mass spec are obtained and the
    structure of mol1 and the formula difference are known. This implementation of ModiFinder only supports one
    modification site.

    Args:
        formula_diff: the formula difference. Examples: '+O', '-CH2'
        load_dir: path to ICEBERG prediction result
        mol_str1: SMILES/InChI of molecule 1 (type is specified in ``mol_type1``)
        name1: name of molecule 1 (for visualization)
        precursor_mass1: precursor mass of molecule 1
        real_spec1: experiment spectrum of molecule 1
        real_spec_type1: type of spectrum 1 (see documentation for function ``load_real_spec`` for details)
        name2: name of molecule 2 (for visualization)
        precursor_mass2: precursor mass of molecule 2
        real_spec2: experiment spectrum of molecule 2
        real_spec_type2: type of spectrum 2 (see documentation for function ``load_real_spec`` for details)
        mol_type1: (default: 'smi') type of mol_str1. Could be 'smi', 'inchi', 'mol', 'inchikey'
        mz_cutoff: (default: 1500) ignore peaks that has m/z larger than this value. Larger peaks sometimes introduces
          ambiguity to structures because too many atoms are included.
        topk_peaks: (default: 10) top k peaks considered to score the modification sites
        top_score_thresh: (default: 0.1) all sites higher than the score of max(scores) - top_score_thresh will be
            highlighted. A larger top_score_thresh will result in more highlighted atoms.
        nce1: (bool, default=False) if True, the collision energies are treated as normalized collision energy; otherwise, they are treated as absolute eV
        nce2: (bool, default=False) if True, the collision energies are treated as normalized collision energy; otherwise, they are treated as absolute eV
        step_collision_energy: (bool, default=False) if True, it means step_collision_energy is turned on in the instrument and only one merged spectrum will be returned
        ppm: (int, default=20) parts-per-million threshold for mass comparison
        save_path: (str) if specified, save result to the specified path
        axes: (list) matplotlib axis(es) for the plots

    Returns: N/A

    """
    import ms_pred.magma.fragmentation as fragmentation
    if axes is None:
        axes = [None] * len(real_spec2)
    elif type(axes) is not list:
        axes = [axes]
    if not len(axes) == len(real_spec2):
        raise ValueError("shape mismatch")

    # load experiment spectra
    real_spec1 = load_real_spec(real_spec1, real_spec_type1, precursor_mass1, nce1, ppm, denoise_spectrum=False)
    real_spec2 = load_real_spec(real_spec2, real_spec_type2, precursor_mass2, nce2, ppm, denoise_spectrum=False)

    # load predicted spectra and fragments
    smiles, pred_specs, pred_frags = load_pred_spec(load_dir, step_collision_energy)
    cand_inchikeys = [common.inchikey_from_smiles(common.rm_stereo(smi)) for smi in smiles]

    if mol_type1 == 'smi':
        inchikey_of_interest = common.inchikey_from_smiles(common.rm_stereo(mol_str1, mol_type1))
    elif mol_type1 == 'mol':
        inchikey_of_interest = Chem.MolToInchikey(common.rm_stereo(mol_str1, mol_type1))
    elif mol_type1 == 'inchi':
        inchikey_of_interest = Chem.InchiToInchikey(common.rm_stereo(mol_str1, mol_type1))
    elif mol_type1 == 'inchikey':
        inchikey_of_interest = mol_str1
    else:
        raise ValueError(f'mol_type={mol_type1} is not supported')

    if not inchikey_of_interest in cand_inchikeys:
        print('No inchikey matching is found for the compound of interested')
        return

    idx = cand_inchikeys.index(inchikey_of_interest)
    smi = smiles[idx]
    pred_spec = pred_specs[idx]
    engine = fragmentation.FragmentEngine(smi, mol_str_type='smiles')

    if any([ce2 not in pred_spec for ce2 in real_spec2.keys()]):  # collision energy mismatch
        colli_eng1 = list(pred_spec.keys())
        colli_eng2 = list(real_spec2.keys())
        colli_eng1_arr = np.array([float(ce) for ce in colli_eng1])
        colli_eng2_arr = np.array([float(ce) for ce in colli_eng2])
        colli_eng_diff = np.abs(colli_eng1_arr[:, None] - colli_eng2_arr[None, :])
        matching = np.nonzero(pygm.hungarian(-colli_eng_diff))
        matched_pairs = [(colli_eng1[idx1], colli_eng2[idx2]) for idx1, idx2 in zip(*matching)]
        real_spec2 = {ce1: real_spec2[ce2] for ce1, ce2 in matched_pairs}
        print(f'The following collision energies pairs are matched between two spectra:\n'
              f'{matched_pairs}')

    if not all([ce2 in real_spec1 and ce2 in pred_spec for ce2 in real_spec2.keys()]):
        raise ValueError('Collision energy mismatch:\n'
                         f'pred_spec1 {pred_spec.keys()}\n'
                         f'real_spec1 {real_spec1.keys()}\n'
                         f'real_spec2 {real_spec2.keys()}\n')

    # find peak matching with formula difference
    if formula_diff[0] == '-':
        mass_diff = -common.formula_mass(formula_diff[1:])
    elif formula_diff[0] == '+':
        mass_diff = common.formula_mass(formula_diff[1:])
    else:
        raise ValueError('formula_diff has to start with \'+\' or \'-\'!')
    assert np.abs(precursor_mass1 + mass_diff - precursor_mass2) < precursor_mass1 * 1e-6 * ppm, \
        f'precursor_mass1={precursor_mass1}, precursor_mass2={precursor_mass2}, mass_diff={mass_diff}'

    interested_peaks = {}
    for ce, spec2 in real_spec2.items():
        cur_matched_peaks = []
        for mz, inten in spec2:
            shifted_mz_in_spec_a = np.min(np.abs(mz - real_spec1[ce][:, 0] - mass_diff)) / mz < 1e-6 * ppm
            mz_in_spec_a = np.min(np.abs(mz - real_spec1[ce][:, 0])) / mz < 1e-6 * ppm
            if shifted_mz_in_spec_a:
                if mz_in_spec_a:
                    a_idx = np.where(np.abs(mz - real_spec1[ce][:, 0]) / mz < 1e-6 * ppm)[0][0]
                    a_mz, a_inten = real_spec1[ce][a_idx]
                    if inten > a_inten:
                        cur_matched_peaks.append((mz, inten - a_inten))
                else:
                    cur_matched_peaks.append((mz, inten))
        if len(cur_matched_peaks) == 0:
            raise ValueError('No peak matching found')
        interested_peaks[ce] = np.array(cur_matched_peaks)

    all_figs = []
    atom_scores = np.zeros(engine.natoms)
    for (ce, int_peaks), ax in zip(interested_peaks.items(), axes):
        # find peak matching and plot peaks
        peak_matching = np.abs(int_peaks[:, 0][:, None] - real_spec2[ce][:, 0][None, :]) < int_peaks[:, 0][:, None] * 1e-6 * ppm
        common.plot_compare_ms(int_peaks, real_spec2[ce], name1 + f'{formula_diff}', name2, f'# of matched peaks={np.sum(peak_matching.max(axis=-1))}', dpi=500, ax=ax)

        # find fragments to peaks
        counter = 0
        int_peaks[:, 1] = int_peaks[:, 1] / np.max(int_peaks[:, 1])
        sorted_idx = np.argsort(int_peaks[:, 1])[::-1]
        for i in sorted_idx:
            mz, inten = int_peaks[i]
            if mz > mz_cutoff:
                continue
            peak_matching = np.abs(mz - mass_diff - pred_spec[ce][:, 0]) < mz * 1e-6 * ppm
            plot_count = 0
            peak_atom_scores = np.zeros(engine.natoms)
            for j in np.where(peak_matching)[0]:
                covered_atoms = engine.get_present_atoms(pred_frags[idx][ce][j])[0] # bitmap to indices
                peak_atom_scores[covered_atoms] += pred_specs[idx][ce][j][1]
                draw_dict = engine.get_draw_dict(pred_frags[idx][ce][j])
                common.plot_mol_as_vector(
                    draw_dict["mol"], hatoms=draw_dict["hatoms"], hbonds=draw_dict["hbonds"],
                    offset=(mz, inten + plot_count * 0.1), zoom=0.0005, ax=plt.gca()
                )
                plot_count += 1
            if len(peak_atom_scores[peak_atom_scores > 0]) == 0:
                print(f'Uncovered peak: {mz:.5f}, {inten:.2f}')
            else:
                print(f'Covered peak: {mz:.5f}, {inten:.2f}')
                # atom_scores[peak_atom_scores > 0] += inten  # no weighting
                # *np.sum(peak_atom_scores > 0)
                atom_scores += peak_atom_scores / np.max(peak_atom_scores) * inten
                    # higher weights for structures that ICEBERG thinks more reasonable
            counter += 1
            if counter >= topk_peaks:
                break

        # normalize to 1
        atom_scores = atom_scores / max(atom_scores.max(), 1e-3)

        all_figs.append(plt.gcf())

    # draw modification site
    plt.figure()
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.axis('off')

    common.plot_mol_as_vector(
        engine.mol, hatoms=(np.where(atom_scores > atom_scores.max() - top_score_thresh)[0]).tolist(),
        hbonds=None, atomcmap=matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['#A7B7C3', '#FFD593']), atomscores=atom_scores, ax=plt.gca()
    )
    all_figs.append(plt.gcf())

    if save_path is not None:
        p = PdfPages(save_path)
        for fig in all_figs:
            fig.savefig(p, format='pdf', bbox_inches='tight')
        p.close()


def generate_buyable_report(
        feat_ids: list,
        exp_specs: list,
        adduct_to_smiles: List[dict],
        config_dict: dict,
        energy: list,
        aux_info: List[dict]=None,
        topk_per_row=3,
        rows_per_page=10,
        output_name='buyable_report',
        cs_coster=None,
        ppm=20,
):
    """
    Generate a buyable report

    Args:
        feat_ids: List of length N, contains the feature IDs
        exp_specs: List of length N, contains experiment spectrum which are {'30 eV': 2D array}
        adduct_to_smiles: List of length N, each contains a dictionary like {"[M+H]+": ['CCNc1nc(Cl)nc(NC(C)=O)n1'], ...}
        config_dict: ICEBERG run configuration dictionary
        energy: collision energies
        aux_info: auxiliary information to show in the report
        topk_per_row: how many items on each row in the report
        rows_per_page: how many rows on each page in the report
        output_name: the name of the output file
        cs_coster: Chem-Space coster
        ppm: (int, default=20) parts-per-million tolerance for mass comparison
    """
    if cs_coster is None:
        cs_coster = coster.ChemSpaceCoster('eFZBhgaLqwk4by3e567DhJavPvnerOX1cnmP6nkV9eMxybPvoZHQzVknSMq3z_8X')
        cs_coster.build_status_log()

    if aux_info is None:
        aux_info = [None] * len(feat_ids)

    config_dict = copy.deepcopy(config_dict)
    all_figs = []
    fig = None
    iterator = enumerate(zip(feat_ids, exp_specs, adduct_to_smiles, aux_info))
    if len(feat_ids) >= 10:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(feat_ids))
    for row_idx, (feat_id, exp_spec, adduct_to_smi, aux) in iterator:
        if row_idx % rows_per_page == 0:
            if fig is not None:
                all_figs.append(fig)
            fig = plt.figure(figsize=(topk_per_row * 10, rows_per_page * 7))

            # headers
            for col_idx in range(topk_per_row):
                plt.subplot(rows_per_page + 1, topk_per_row + 1, col_idx + 2)
                plt.text(0.4, 0.5, f'ICEBERG Top {col_idx + 1}', fontsize=20)
                plt.axis('off')

        # merge different adducts
        all_topk_results = []
        for adduct, smiles in adduct_to_smi.items():
            config_dict['adduct'] = adduct

            # Run ICEBERG to predict spectra
            result_path, pmz = iceberg_prediction(smiles, energy, **config_dict)

            # hack the precursor mz if there are multiple formulae within tolerance
            pmz = common.merge_mz(pmz, ppm)

            # Compare spectrum similarity for elucidation
            topk_results = elucidation_over_candidates(result_path, exp_spec, precursor_mass=pmz, mol_name=feat_id, topk=rows_per_page, **config_dict)
            topk_results = [list(r) + [result_path, pmz] for r in topk_results]
            all_topk_results += topk_results

        sorted(all_topk_results, key=lambda x: x[1])

        # row title
        plt.subplot(rows_per_page + 1, topk_per_row + 1, (row_idx % rows_per_page + 1) * (topk_per_row + 1) + 1)
        title_txt = f'FEATURE_ID={feat_id}'
        if aux:
            for k, v in aux.items():
                new_txt = f'\n{k}: {v}'
                split_indices = np.arange(0, len(new_txt), 30) # make a new line if a line is longer than 30 characters
                for i in range(len(split_indices)):
                    if i < len(split_indices) - 1:
                       title_txt += new_txt[split_indices[i]:split_indices[i+1]] + '\n'
                    else:
                        title_txt += new_txt[split_indices[i]:]
        plt.text(0.2, 0.5, title_txt, fontsize=20, horizontalalignment='left', verticalalignment='center')
        plt.axis('off')

        # plot results
        for col_idx, (smi, dist, _, result_path, pmz) in enumerate(all_topk_results[:topk_per_row]):
            plt.subplot(rows_per_page + 1, topk_per_row + 1, (row_idx % rows_per_page + 1) * (topk_per_row + 1) + col_idx + 2)
            explain_peaks(result_path, exp_spec, pmz, smi, num_peaks=10, axes=[plt.gca()], **config_dict)

            # Add molecule to plot
            common.plot_mol_as_vector(Chem.MolFromSmiles(smi), ax=plt.gca(), offset=(pmz/5, 0.7), zoom=0.003)
            plt.title('')  # remove title

            # Add entropy distance
            plt.text(pmz / 5 * 2, 0.9, f'entr_dist={dist:.3f}', fontsize=15)

            # Add price
            is_in_stock, cost = cs_coster.get_buyable_and_cost(smi)
            if is_in_stock:
                buyable_text = f'in-stock\n${cost[0]}/{cost[1]}mg, ships {cost[2]} days'
            else:
                is_on_demand, cost = cs_coster.get_buyable_and_cost(smi, in_stock_only=False)
                if is_on_demand and cost[0] is not None:
                    buyable_text = f'on-demand\n${cost[0]}/{cost[1]}mg, ships {cost[2]} days'
                else:
                    buyable_text = 'not buyable/custom synthesis'
            plt.text(pmz / 5 * 2, 0.6, buyable_text, fontsize=15)

            # Add smiles
            plt.text(pmz / 5 * 2, 0.4, smi, fontsize=10)

    all_figs.append(fig)

    with PdfPages(f'{output_name}.pdf') as pdf:
        for fig in all_figs:
            pdf.savefig(fig, bbox_inches='tight')


def form_from_mgf_buddy(
    inp_mgf: str,
    top_k_buddy_preds: int = 5,
    profile: str = 'orbitrap',
    ms1_ppm: int = 5,
    ms2_ppm: int = 10,
    fdr_cutoff: float = 0.1,
    halogen: bool = False,
):
    """
    Calculate chemical formula from MGF file by Buddy

    Args:
        inp_mgf: path to mgf
        top_k_buddy_preds: how many formula predictions are returned
        fdr_cutoff: if estimated false detection rate (FDR) is lower than fdr_cutoff, only top-1 will be returned
        profile: MS/MS profile (default: orbitrap)
        ms1_ppm: MS1 ppm tolerance
        ms2_ppm: MS2 ppm tolerance
        halogen: is halogen in formula
    """
    from msbuddy import Msbuddy, MsbuddyConfig

    msb_config = MsbuddyConfig(ms_instr=profile,  # supported: "qtof", "orbitrap", "fticr" or None
                               ppm=True,  # use ppm for mass tolerance
                               ms1_tol=ms1_ppm,  # MS1 tolerance in ppm or Da
                               ms2_tol=ms2_ppm,  # MS2 tolerance in ppm or Da
                               halogen=halogen)
    msb_engine = Msbuddy(msb_config)
    msb_engine.load_mgf(inp_mgf)
    msb_engine.annotate_formula()
    all_results = msb_engine.get_summary()
    feature_id_to_form = {}
    for result in all_results:
        adduct_and_form = []
        adduct = result['adduct']
        if adduct not in common.ion2mass:
            continue  # skip adduct types that are not supported by ICEBERG
        if result['estimated_fdr'] is not None and fdr_cutoff < result['estimated_fdr']:
            adduct_and_form.append(dict(rnk=1, adduct=adduct, form=result['formula_rank_1']))
        else:
            for i in range(1, top_k_buddy_preds+1):
                if f'formula_rank_{i}' in result and result[f'formula_rank_{i}'] is not None:
                    adduct_and_form.append(dict(rnk=i, adduct=adduct, form=result[f'formula_rank_{i}']))
        if len(adduct_and_form) > 0:
            feature_id_to_form[result['identifier']] = adduct_and_form
    return feature_id_to_form


def form_from_mgf_sirius(
    inp_mgf: str,
    top_k_sirius_preds: int = 5,
    profile: str = 'orbitrap',
    ppm: int = 10,
    sirius_path: str = 'sirius',
    elements_enforced: str = '',
    elements_considered: str = '',
    ions_considered: str = '',
    **kwargs
):
    """
    Calculate chemical formula from MGF file by SIRIUS

    Args:
        inp_mgf: path to mgf
        top_k_sirius_preds: how many formula predictions are returned
        profile: MS/MS profile (default: orbitrap)
        ppm: (default: 10)
        sirius_path: path to sirius runtime (default: local )
        elements_enforced: Enforce elements for molecular formula determination.
            Example: CHNOPSCl to allow the elements C, H, N, O, P, S and Cl. Add numbers in brackets to restrict the
            minimal and maximal allowed occurrence of these elements: CHNOP[5]S[8]Cl[1-2]. When one number is given
             then it is interpreted as upper bound.
        elements_considered: Set the allowed elements for rare element detection.
            Example: `SBrClBSe` to allow the elements S,Br,Cl,B and Se.
        ions_considered: the iontype/adduct of the MS/MS data.
            Example: [M+H]+, [M-H]-, [M+Cl]-, [M+Na]+, [M]+. You can also provide a comma separated list of adducts.
            Default: [M+H]+,[M+K]+,[M+Na]+,[M+H-H2O]+,[M+H-H4O2]+,[M+NH4]+,[M-H]-,[M+Cl]-,[M-H2O-H]-,[M+Br]-

    """
    sirius_config_cmd = f'--ignore-formula --noCite formula ' \
                        f'-p {profile} --ppm-max={ppm} '
    if len(elements_enforced) > 0:
        sirius_config_cmd += f'--elements-enforced={elements_enforced} '
    if len(elements_considered) > 0:
        sirius_config_cmd += f'--elements-considered={elements_considered} '
    if len(ions_considered) > 0:
        sirius_config_cmd += f'--ions-considered={ions_considered} '
    sirius_config_cmd += 'write-summaries'
    exp_hash = common.md5(inp_mgf) + '||' + sirius_config_cmd
    out_dir = Path(user_cache_dir(f"ms-pred/sirius-out/{common.str_to_hash(exp_hash)}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (out_dir / 'sirius_run_successful').exists():
        sirius_command = (f'{sirius_path} -o {out_dir} '
                          f'-i {inp_mgf} ' + sirius_config_cmd)
        print("Running SIRIUS, command:\n" + sirius_command + "\n")
        run_result = subprocess.run(sirius_command, shell=True)

        if run_result.returncode == 0:  # successful
            (out_dir / 'sirius_run_successful').touch()

    feature_id_to_form = {}
    for per_cmpd_out_dir in out_dir.glob('*'):
        feature_id = per_cmpd_out_dir.stem.split('_')[-1]
        sirius_cands_path = per_cmpd_out_dir / 'formula_candidates.tsv'
        if sirius_cands_path.exists():
            adduct_and_form = []
            df = pd.read_csv(sirius_cands_path, sep='\t')
            for idx, sirius_row in df.iterrows():
                if idx >= top_k_sirius_preds:
                    continue
                adduct = sirius_row['adduct'].replace(" ", "")
                adduct_and_form.append(dict(rnk=idx+1, adduct=adduct, form=sirius_row['molecularFormula']))
            feature_id_to_form[feature_id] = adduct_and_form
    return feature_id_to_form