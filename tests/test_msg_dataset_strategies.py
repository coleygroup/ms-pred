"""Scientific subset, split, and spectrum checks for the public MSG builder."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'data_scripts/create_msg_simulation_dataset.py'
_spec = importlib.util.spec_from_file_location('msg_builder', SCRIPT)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


@pytest.fixture
def source_table(tmp_path):
    rows = []
    for spec, ce, simulation, fold in [
        ('known_sim', 30, True, 'train'),
        ('unknown_sim', None, True, 'val'),
        ('known_other', 12.5, False, 'test'),
        ('unknown_other', None, False, 'val'),
        ('known_val', 20, True, 'val'),
        ('known_test', 40, True, 'test'),
        ('imputed', '30 [imputed]', True, 'test'),
    ]:
        rows.append(dict(identifier=spec, collision_energy=ce, simulation_challenge=simulation,
                         fold=fold, smiles='CCO', inchikey='LFQSCWFLJHTTHZ', formula='C2H6O',
                         precursor_mz=47.04914, adduct='[M+H]+', instrument_type='Orbitrap',
                         mzs='15.023,29.039', intensities='0.25,1.0'))
    path = tmp_path / 'source.tsv'
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False)
    return path


def run_builder(source_table, output, *args, check=True):
    result = subprocess.run([sys.executable, str(SCRIPT), '--source-table', str(source_table),
                             '--output-dataset', str(output), '--chunksize', '2', *args],
                            text=True, capture_output=True)
    if check:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    return result


@pytest.mark.parametrize('strategy,expected,unknown', [
    ('msg_all', {'known_sim', 'unknown_sim', 'known_other', 'unknown_other', 'known_val', 'known_test', 'imputed'}, 3),
    ('msg_known_ce', {'known_sim', 'known_other', 'known_val', 'known_test'}, 0),
    ('msg_simulation', {'known_sim', 'known_val', 'known_test'}, 0),
])
def test_native_strategy_cohorts_and_spectra(source_table, tmp_path, strategy, expected, unknown):
    out = tmp_path / strategy
    audit = run_builder(source_table, out, '--strategy', strategy)
    labels = pd.read_csv(out / 'labels.tsv', sep='\t')
    splits = pd.read_csv(out / 'splits/split.tsv', sep='\t').set_index('name')['split'].to_dict()
    assert set(labels.spec) == expected == set(splits)
    assert audit['unknown_ce_rows'] == unknown
    assert audit['imputed_check'] == {'literal_imputed_count': 0, 'collision_imputed_true': 0}
    assert audit['spectra_origin'] == 'source_table'
    assert 'subformulae/no_subform.hdf5' in audit['pending_resources']
    source_folds = pd.read_csv(source_table, sep='\t').set_index('identifier')['fold'].to_dict()
    assert splits == {key: source_folds[key] for key in expected}
    with h5py.File(out / 'spec_files.hdf5') as h5:
        assert set(h5) == {key + '.ms' for key in expected}
        text = h5['known_sim.ms'][0].decode()
        metadata, peaks = text.split('\n\n')
        assert '#parentmass 47.04914' in metadata
        assert '#ionization [M+H]+' in metadata
        assert peaks.splitlines() == ['>collision 30 eV', '15.023 0.25', '29.039 1.0']
        if unknown:
            assert '>collision nan eV' in h5['unknown_sim.ms'][0].decode()
    with h5py.File(out / 'spec_files_debug.hdf5') as h5:
        assert set(h5) == {key + '.ms' for key in expected}


def test_default_strategy_and_dynamic_output(source_table, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT)])
    args = builder.get_args()
    assert args.strategy == 'msg_simulation'
    assert args.output_dataset == Path('data/spec_datasets/msg_simulation')
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--strategy', 'msg_all'])
    assert builder.get_args().output_dataset == Path('data/spec_datasets/msg_all')
    labels, _ = builder.read_simulation_source(str(source_table), 2)
    assert set(labels.spec) == {'known_sim', 'known_val', 'known_test'}


def test_reuse_filters_annotations_candidates_and_preserves_source(source_table, tmp_path):
    source = tmp_path / 'all'
    run_builder(source_table, source, '--strategy', 'msg_all')
    (source / 'subformulae').mkdir()
    with h5py.File(source / 'subformulae/no_subform.hdf5', 'w') as h5:
        for key in ['known_sim_collision 30.json', 'known_other_collision 12.json', 'unknown_sim_collision nan.json']:
            h5.create_dataset(key, data=['{}'], dtype=h5py.string_dtype())
    (source / 'retrieval').mkdir(exist_ok=True)
    candidates = pd.DataFrame({'spec': ['known_sim', 'known_other', 'unknown_sim'], 'smiles': ['CCO'] * 3})
    for name in ['formula', 'mass']:
        candidates.to_csv(source / f'retrieval/cands_df_test_{name}_256.tsv', sep='\t', index=False)
    source_bytes = (source / 'labels.tsv').read_bytes()
    out = tmp_path / 'sim'
    audit = run_builder(source_table, out, '--source-dataset', str(source))
    assert (out / 'spec_files.hdf5').is_symlink()
    assert audit['subformulae']['written_entries'] == 1
    assert audit['subformulae']['missing_label_pairs'] == 2
    assert audit['pending_resources'] == []
    with h5py.File(out / 'subformulae/no_subform.hdf5') as h5:
        assert set(h5) == {'known_sim_collision 30.json'}
    candidates = pd.read_csv(out / 'retrieval/cands_df_test_formula_256.tsv', sep='\t')
    assert set(candidates.spec) == {'known_sim'}
    assert (source / 'labels.tsv').read_bytes() == source_bytes


def test_reuse_without_optional_resources(source_table, tmp_path):
    source = tmp_path / 'all'
    run_builder(source_table, source, '--strategy', 'msg_all')
    audit = run_builder(source_table, tmp_path / 'known', '--strategy', 'msg_known_ce', '--source-dataset', str(source))
    assert audit['output_rows'] == 4
    assert len(audit['pending_resources']) == 3


def test_incomplete_source_hdf5_fails_before_writing(source_table, tmp_path):
    source = tmp_path / 'all'
    run_builder(source_table, source, '--strategy', 'msg_all')
    with h5py.File(source / 'spec_files.hdf5', 'a') as h5:
        del h5['known_sim.ms']
    out = tmp_path / 'sim'
    result = run_builder(source_table, out, '--source-dataset', str(source), check=False)
    assert result.returncode != 0 and 'missing 1 selected spectra' in result.stderr
    assert not out.exists()


def test_existing_outputs_require_overwrite(source_table, tmp_path):
    out = tmp_path / 'all'
    run_builder(source_table, out, '--strategy', 'msg_all')
    before = (out / 'spec_files.hdf5').read_bytes()
    result = run_builder(source_table, out, check=False)
    assert result.returncode != 0 and '--overwrite' in result.stderr
    assert (out / 'spec_files.hdf5').read_bytes() == before
    run_builder(source_table, out, '--strategy', 'msg_all', '--overwrite')


@pytest.mark.parametrize('column,value,error', [
    ('intensities', '1', 'Invalid peak array lengths'),
    ('mzs', 'nan,29', 'Nonfinite or negative peaks'),
    ('fold', 'unassigned', 'Unrecognized source fold'),
])
def test_invalid_source_does_not_replace_existing_outputs(source_table, tmp_path, column, value, error):
    out = tmp_path / 'all'
    run_builder(source_table, out, '--strategy', 'msg_all')
    before = {name: (out / name).read_bytes() for name in ['labels.tsv', 'spec_files.hdf5']}
    df = pd.read_csv(source_table, sep='\t')
    df.loc[0, column] = value
    df.to_csv(source_table, sep='\t', index=False)
    result = run_builder(source_table, out, '--strategy', 'msg_all', '--overwrite', check=False)
    assert result.returncode != 0 and error in result.stderr
    for name, content in before.items():
        assert (out / name).read_bytes() == content


def test_all_and_known_do_not_require_simulation_flag(source_table):
    df = pd.read_csv(source_table, sep='\t').drop(columns='simulation_challenge')
    df.to_csv(source_table, sep='\t', index=False)
    labels, _ = builder.read_source(str(source_table), 2, 'msg_all')
    assert len(labels) == 7
    labels, _ = builder.read_source(str(source_table), 2, 'msg_known_ce')
    assert len(labels) == 4


def test_source_output_alias_rejected(source_table, tmp_path):
    source = tmp_path / 'all'
    run_builder(source_table, source, '--strategy', 'msg_all')
    alias = tmp_path / 'alias'
    alias.symlink_to(source, target_is_directory=True)
    result = run_builder(source_table, alias, '--source-dataset', str(source), '--overwrite', check=False)
    assert result.returncode != 0 and 'different directories' in result.stderr


def test_explicit_missing_candidate_rejected(source_table, tmp_path):
    source = tmp_path / 'all'
    run_builder(source_table, source, '--strategy', 'msg_all')
    out = tmp_path / 'sim'
    result = run_builder(source_table, out, '--source-dataset', str(source), '--candidate-file', 'absent.tsv', check=False)
    assert result.returncode != 0 and 'absent.tsv' in result.stderr
    assert not out.exists()


def test_duplicate_ids_across_chunks_and_numeric_precision(source_table, tmp_path):
    df = pd.read_csv(source_table, sep='\t')
    df.loc[0, 'collision_energy'] = 30.123456789
    df.loc[0, 'mzs'] = '15.023456789012345,29.039'
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)
    df.to_csv(source_table, sep='\t', index=False)
    out = tmp_path / 'all'
    audit = run_builder(source_table, out, '--strategy', 'msg_all')
    assert audit['duplicate_selected_rows'] == 1
    labels = pd.read_csv(out / 'labels.tsv', sep='\t').set_index('spec')
    assert labels.loc['known_sim', 'collision_energies'] == "['30.123456789']"
    with h5py.File(out / 'spec_files.hdf5') as h5:
        lines = h5['known_sim.ms'][0].decode().splitlines()
        assert float(lines[-2].split()[0]) == float('15.023456789012345')
        assert len(h5) == 7


def test_overwrite_does_not_retain_stale_candidates(source_table, tmp_path):
    out = tmp_path / 'all'
    run_builder(source_table, out, '--strategy', 'msg_all')
    pd.DataFrame({'spec': ['unknown_sim']}).to_csv(
        out / 'retrieval/cands_df_test_formula_256.tsv', sep='\t', index=False,
    )
    before = (out / 'labels.tsv').read_bytes()
    result = run_builder(source_table, out, '--overwrite', check=False)
    assert result.returncode != 0 and 'stale resource' in result.stderr
    assert (out / 'labels.tsv').read_bytes() == before
