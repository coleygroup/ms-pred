from ms_pred import common
from pathlib import Path

adduct = '[M+H]+'
pred_dir = "data/retrieval/pubchem/atlas/spectra"
pred_dir = Path(pred_dir)
out_dir = "data/retrieval/pubchem/atlas/out_mgf"
out_dir = Path(out_dir)
out_dir.mkdir(exist_ok=True)


def read_pred(batch_path):
    specdb = common.PredSpecDB(batch_path / 'preds.hdf5')
    form_meta_spec = {}
    for name, ikeys, spec_dict in specdb.get_all_specs():
        for ce, spec in spec_dict.items():
            smi = spec.root_canonical_smiles
            form = common.form_from_smi(smi)
            inchi = common.inchi_from_smiles(smi)
            ikey = ikeys.strip('ikey ')

            if not form in form_meta_spec:
                form_meta_spec[form] = []

            form_meta_spec[form].append(({
                'CHARGE': '1+',
                'PEPMASS': common.formula_mass(form) + common.ion2mass[adduct],
                'DESCRIPTION': 'ICEBERG prediction',
                'FORMULA': form,
                'INCHI': inchi,
                'SMILES': smi,
                'INCHIKEY': ikey,
                'ADDUCT': adduct,
                'COLLISION_ENERGY': common.get_collision_energy(ce),
            }, spec.merged_spec))
    return form_meta_spec


def write_mgf(all_form_meta_spec):
    for form_meta_spec in all_form_meta_spec:
        for form, meta_spec_buffer in form_meta_spec.items():
            if len(meta_spec_buffer) > 0:
                form_path = out_dir / common.get_formula_subdir(form) / f'{form}.mgf'
                form_path.parent.mkdir(exist_ok=True, parents=True)
                mgf_str = common.build_mgf_str(meta_spec_buffer)
                with open(form_path, 'a') as f:
                    f.write(mgf_str)
                    f.write('\n\n')

if __name__ == '__main__':
    batch_paths = [batch_path for batch_path in pred_dir.iterdir()]
    common.chunked_parallel(batch_paths, read_pred, output_func=write_mgf, chunks=len(batch_paths), max_cpu=32)
