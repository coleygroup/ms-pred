from ms_pred import common
from pathlib import Path
from datetime import datetime
import time

adduct = '[M+H]+'
pred_dir = "data/retrieval/pubchem/atlas/spectra"
pred_dir = Path(pred_dir)
out_dir = "data/retrieval/pubchem/atlas/h_plus_out_mgf"
out_dir = Path(out_dir)
out_dir.mkdir(exist_ok=True)


def read_pred(batch_path):
    specdb = common.PredSpecDB(batch_path / 'preds.hdf5')
    write_mgf_label = batch_path / 'written_mgf'
    form_meta_spec = {}
    if not write_mgf_label.exists():
        for name, ikeys, spec_dict in specdb.get_all_specs():
            for ce, spec in spec_dict.items():
                spec: common.MassSpec = spec.inten_thresh(1e-4)
                if spec.has_masses and spec.has_intens:
                    spec.sort_peaks_by_mz()  # sort peaks and frags so that don't need to sort again in mgf
                    form = spec.root_form

                    if not form in form_meta_spec:
                        form_meta_spec[form] = []

                    form_meta_spec[form].append(({
                        'CHARGE': '1' + adduct[-1],
                        'PEPMASS': spec.parent_mass,
                        'DESCRIPTION': 'ICEBERG prediction',
                        'FORMULA': form,
                        'INCHI': spec.inchi,
                        'SMILES': spec.root_canonical_smiles,
                        'INCHIKEY': spec.inchikey,
                        'ADDUCT': adduct,
                        'COLLISION_ENERGY': spec.collision_energy,
                        'FRAGS': ','.join([str(ifrag) for ifrag in spec.int_frags]) if spec.has_frags else '',
                    }, spec.spec))
    return form_meta_spec, write_mgf_label


def write_mgf(all_form_meta_spec_and_write_label):
    for form_meta_spec_and_write_label in all_form_meta_spec_and_write_label:
        form_meta_spec, write_label = form_meta_spec_and_write_label
        if not write_label.exists():
            for form, meta_spec_buffer in form_meta_spec.items():
                if len(meta_spec_buffer) > 0:
                    form_path = out_dir / common.get_formula_subdir(form) / f'{form}.mgf'
                    form_path.parent.mkdir(exist_ok=True, parents=True)
                    mgf_str = common.build_mgf_str(meta_spec_buffer, sort_peaks=False)
                    with open(form_path, 'a') as f:
                        f.write(mgf_str)
                        f.write('\n\n')
            write_label.touch()

if __name__ == '__main__':
    # This script has an infinite loop that handles unprocessed entries
    # You can start it as soon as 02_run_prediction is running
    while True:
        batch_paths = []
        for batch_path in pred_dir.iterdir():
            if (batch_path / 'success').exists() and not (batch_path / 'written_mgf').exists():
                batch_paths.append(batch_path)
        if len(batch_paths) > 0:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {len(batch_paths)} batches to write')
            common.chunked_parallel(batch_paths, read_pred, output_func=write_mgf, chunks=len(batch_paths), max_cpu=32)
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Written {len(batch_paths)} predicted batches to mgf')
        else:
            time.sleep(60)

