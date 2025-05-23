import itertools

import pandas as pd
import subprocess
from pathlib import Path
from ms_pred import common
import pickle
from tqdm import tqdm


datasets = ["nist20"]
splits = ["split_1", "scaffold_1"]
num_threads = 96
jar_file = "../metfrag/MetFragCommandLine-2.6.5.jar"

metfrag_ion_mapping = {
    "[M+H]+": 1,
    "[M+H3N+H]+": 18,
    "[M+Na]+": 23,
    "[M+K]+": 39,
    "[M-H]-": -1,
    "[M+Cl]-": 35,
}

real_threads_per_task = max(num_threads // (len(datasets) * len(splits)), 1)


if __name__ == '__main__':
    metfrag_commands = []
    for dataset, split in itertools.product(datasets, splits):
        spec_files = f"data/spec_datasets/{dataset}/spec_files.hdf5"
        candidates = f"data/spec_datasets/{dataset}/retrieval/cands_pickled_{split}_50.p"
        spec_h5 = common.HDF5Dataset(spec_files)
        res_folder = Path(f"results/metfrag_{dataset}/{split}")
        res_folder.mkdir(exist_ok=True, parents=True)
        id_to_dict = pickle.load(open(candidates, 'rb'))

        # intermediate directories
        metfrag_params = res_folder / "params"
        metfrag_params.mkdir(exist_ok=True)
        metfrag_cands = res_folder / "candidates"
        metfrag_cands.mkdir(exist_ok=True)
        metfrag_specs = res_folder / "specs"
        metfrag_specs.mkdir(exist_ok=True)
        metfrag_output_rank = res_folder / "metfrag_out"
        metfrag_output_rank.mkdir(exist_ok=True)
        metfrag_runscripts = res_folder / "runscripts"
        metfrag_runscripts.mkdir(exist_ok=True)

        run_scripts = []
        print('Processing prediction entries')
        for spec_id, info_dict in tqdm(id_to_dict.items()):
            if info_dict["ionization"] not in metfrag_ion_mapping.keys(): # not supported adduct
                continue
            mono_mass = info_dict["precursor"] - common.ion2mass[info_dict["ionization"]]
            cand_database = [{"Identifier": f"{i:03}",
                              "InChI": common.inchi_from_smiles(smi),
                              "MonoisotopicMass": mono_mass,
                              "MolecularFormula": info_dict["formula"],
                              "InChIKey1": ikey.split('-')[0],
                              "InChIKey2": ikey.split('-')[1],
                              "SMILES": smi,
                              "Name": f"{i:03}",
                              "InChIKey3": ikey.split('-')[2]}
                             for i, (smi, ikey) in enumerate(info_dict['cands'])]
            df = pd.DataFrame(cand_database)
            cand_csv = metfrag_cands / f"{spec_id}_candidates.csv"
            df.to_csv(cand_csv, index=None)

            real_spec = spec_h5.read_str(f"{spec_id}.ms").split("\n")
            meta, specs = common.parse_spectra(real_spec)
            real_spec = common.process_spec_file(meta, specs, merge_specs=True)

            spec_file = metfrag_specs / f"{spec_id}.txt"
            with open(spec_file, 'w') as f:
                f.write("\n".join([f"{mz}\t{inten}" for mz, inten in real_spec]))

            param_str = \
f"""PeakListPath = {spec_file}

MetFragDatabaseType = LocalCSV
LocalDatabasePath = {cand_csv}
NeutralPrecursorMolecularFormula = {info_dict["formula"]}
NeutralPrecursorMass = {mono_mass}

FragmentPeakMatchAbsoluteMassDeviation = 0.001
FragmentPeakMatchRelativeMassDeviation = 5
PrecursorIonMode = {metfrag_ion_mapping[info_dict["ionization"]]}
IsPositiveIonMode = {info_dict["ionization"][-1] == '+'}

MetFragScoreTypes = FragmenterScore
MetFragScoreWeights = 1.0

MetFragCandidateWriter = CSV
SampleName = {spec_id}
ResultsPath = {metfrag_output_rank}

MaximumTreeDepth = 2
MetFragPreProcessingCandidateFilter = UnconnectedCompoundFilter
MetFragPostProcessingCandidateFilter = InChIKeyFilter"""
            param_file = metfrag_params / f'{spec_id}.txt'
            with open(param_file, 'w') as f:
                f.write(param_str)

            run_scripts.append(f'java -jar {jar_file} {param_file}')

        batches = common.batches_num_chunks(run_scripts, real_threads_per_task)
        for batch_ind, batch in enumerate(batches):
            cmd = metfrag_runscripts / f'metfrag_cmd_{batch_ind}.sh'
            with open(cmd, 'w') as f:
                f.write("\n".join(batch))
            metfrag_commands.append(cmd)

    full_cmd = "\n".join([f"sh {i} &" for i in metfrag_commands])
    cmd_file = res_folder.parent / "metfrag_full_cmd.sh"

    wait_forever_cmd = "\nwhile true; do\n\tsleep 100\ndone"
    with open(cmd_file, "w") as fp:
        fp.write(full_cmd)
        fp.write(wait_forever_cmd)

    print("Running predictions")
    subprocess.run(f"sh {cmd_file}", shell=True)
