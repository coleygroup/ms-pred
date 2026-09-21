"""Smoke tests for the GLACIER training path.

Two regressions are covered:

1. ``data_scripts/dag/add_dag_intens.py --magma-output`` must be able to read
   the ``PredSpecDB`` ``magma_tree.hdf5`` written by ``run_magma.py`` and emit
   the ``<spec>_collision <ce>`` JSON trees consumed by the GLACIER dataset.
2. ``glacier.joint_model.JointModel`` must complete a training step, including
   the ``lr_scheduler_step`` hook, under the installed pytorch-lightning.

Everything runs on CPU in a temporary directory with a four-molecule dataset.
"""
import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import ms_pred.common as common
from ms_pred.glacier.dataset import IntenDataset, TreeProcessor
from ms_pred.glacier.joint_model import JointModel
from ms_pred.magma.fragmentation import FRAGMENT_ENGINE_PARAMS, FragmentEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLISION_ENERGY = 20

# (spec, smiles, formula, [M+H]+ m/z, fragment m/z values with a known subformula)
TINY_SPECS = [
    ("tiny_caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C8H10N4O2", 195.0877, [138.0662, 110.0713]),
    ("tiny_paracetamol", "CC(=O)Nc1ccc(O)cc1", "C8H9NO2", 152.0706, [110.0600, 93.0335]),
    ("tiny_nicotine", "CN1CCCC1c1cccnc1", "C10H14N2", 163.1230, [132.0808, 84.0808]),
    ("tiny_aspirin", "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4", 181.0495, [163.0390, 121.0284]),
]


def _peaks(precursor_mz, frag_mzs):
    mzs = [precursor_mz] + list(frag_mzs)
    intens = [1.0, 0.6, 0.3][: len(mzs)]
    return list(zip(mzs, intens))


def _labels_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": "tiny",
            "spec": [s[0] for s in TINY_SPECS],
            "ionization": "[M+H]+",
            "formula": [s[2] for s in TINY_SPECS],
            "smiles": [s[1] for s in TINY_SPECS],
            "inchikey": [f"TINY{i:010d}" for i in range(len(TINY_SPECS))],
            "instrument": "Orbitrap",
            "collision_energies": f"['{COLLISION_ENERGY}']",
            "precursor": [s[3] for s in TINY_SPECS],
        }
    )


def _write_tiny_dataset(data_dir: Path) -> None:
    """labels.tsv + spec_files.hdf5 in the layout the data scripts expect."""
    data_dir.mkdir(parents=True, exist_ok=True)
    _labels_df().to_csv(data_dir / "labels.tsv", sep="\t", index=False)
    spec_h5 = common.HDF5Dataset(data_dir / "spec_files.hdf5", mode="w")
    for spec, smiles, formula, precursor_mz, frag_mzs in TINY_SPECS:
        peak_str = "\n".join(f"{mz} {inten}" for mz, inten in _peaks(precursor_mz, frag_mzs))
        spec_h5.write_str(
            f"{spec}.ms",
            f">compound {spec}\n>formula {formula}\n>parentmass {precursor_mz}\n"
            f">ionization [M+H]+\n>smiles {smiles}\n\n"
            f">collision {COLLISION_ENERGY}\n{peak_str}\n",
        )
    spec_h5.close()


def _run_script(monkeypatch, script: str, *args) -> None:
    """Run a repository script in-process as ``__main__`` (avoids one interpreter start-up per stage)."""
    monkeypatch.setattr(sys, "argv", [script, *[str(a) for a in args]])
    runpy.run_path(str(REPO_ROOT / script), run_name="__main__")


def _tree_processor() -> TreeProcessor:
    return TreeProcessor(
        pe_embed_k=0,
        root_encode="graphormer",
        embed_elem_group=True,
        multi_hop_max_dist=3,
    )


def _inten_dataset(magma_h5_path: Path, tree_processor: TreeProcessor) -> IntenDataset:
    """Same construction as glacier/train_joint.py."""
    magma_h5 = common.HDF5Dataset(magma_h5_path)
    name_to_json = {Path(i).stem: i for i in magma_h5.get_all_names()}
    magma_h5.close()
    return IntenDataset(
        _labels_df(),
        magma_h5=magma_h5_path,
        magma_map=name_to_json,
        num_workers=0,
        root_encode="graphormer",
        embed_elem_group=True,
        tree_processor=tree_processor,
        datatype="HDF5",
    )


@pytest.mark.integration
def test_add_dag_intens_reads_magma_predspecdb(tmp_path, monkeypatch):
    """run_magma.py -> 01_assign_subformulae.py -> add_dag_intens.py --magma-output."""
    data_dir = tmp_path / "tiny"
    _write_tiny_dataset(data_dir)
    magma_dir = data_dir / "magma_outputs"
    out_h5_path = magma_dir / "magma_tree_with_inten.hdf5"

    # --debug selects the serial code path of each script (no worker pool)
    _run_script(
        monkeypatch,
        "src/ms_pred/magma/run_magma.py",
        "--spectra-dir", data_dir / "spec_files.hdf5",
        "--output-dir", magma_dir,
        "--spec-labels", data_dir / "labels.tsv",
        "--max-peaks", 50,
        "--ppm-diff", 20,
        "--workers", 1,
        "--debug",
    )
    magma_db = common.PredSpecDB(magma_dir / "magma_tree.hdf5")
    assert sorted(magma_db.get_all_names()) == sorted(s[0] for s in TINY_SPECS)
    magma_db.close()

    _run_script(
        monkeypatch,
        "data_scripts/forms/01_assign_subformulae.py",
        "--data-dir", data_dir,
        "--labels-file", data_dir / "labels.tsv",
        "--use-all",
        "--output-dir-name", "no_subform.hdf5",
        "--num-workers", 1,
        "--debug",
    )
    _run_script(
        monkeypatch,
        "data_scripts/dag/add_dag_intens.py",
        "--pred-dag-path", magma_dir / "magma_tree.hdf5",
        "--true-dag-path", data_dir / "subformulae" / "no_subform.hdf5",
        "--out-dag-path", out_h5_path,
        "--num-workers", 0,
        "--magma-output",
    )

    out_h5 = common.HDF5Dataset(out_h5_path)
    names = sorted(out_h5.get_all_names())
    assert names == sorted(f"{s[0]}_collision {COLLISION_ENERGY}" for s in TINY_SPECS)
    for name in names:
        tree = json.loads(out_h5.read_str(name))
        for key in ["root_canonical_smiles", "adduct", "collision_energy", "frags", "raw_spec"]:
            assert key in tree, f"{name} is missing {key}"
        assert tree["collision_energy"] == COLLISION_ENERGY
        assert len(tree["frags"]) > 0
        assert all("frag" in frag for frag in tree["frags"].values())
        assert len(tree["raw_spec"]) > 0
    out_h5.close()

    # The output must be consumable by the GLACIER dataset as train_joint.py builds it
    dataset = _inten_dataset(out_h5_path, _tree_processor())
    assert len(dataset) == len(TINY_SPECS)
    item = dataset[0]
    assert item["frag_targs"].shape[0] > 0
    assert len(item["inten_targs"]) > 0


def test_glacier_fast_dev_run_cpu(tmp_path):
    """One CPU training step, including the lr scheduler hook, under the installed Lightning."""
    # Hand-built trees in the format written by add_dag_intens.py --magma-output;
    # the only fragment is the intact molecule, which keeps this test independent of MAGMa.
    magma_h5_path = tmp_path / "magma_tree_with_inten.hdf5"
    magma_h5 = common.HDF5Dataset(magma_h5_path, mode="w")
    for spec, smiles, _, precursor_mz, frag_mzs in TINY_SPECS:
        engine = FragmentEngine(mol_str=smiles, **FRAGMENT_ENGINE_PARAMS)
        tree = {
            "root_canonical_smiles": engine.smiles,
            "adduct": "[M+H]+",
            "collision_energy": float(COLLISION_ENERGY),
            "frags": {"0": {"frag": (1 << engine.natoms) - 1}},
            "raw_spec": [list(p) for p in _peaks(precursor_mz, frag_mzs)],
        }
        magma_h5.write_str(f"{spec}_collision {COLLISION_ENERGY}", json.dumps(tree))
    magma_h5.close()

    tree_processor = _tree_processor()
    dataset = _inten_dataset(magma_h5_path, tree_processor)
    assert len(dataset) == len(TINY_SPECS)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.get_collate_fn())

    model = JointModel(
        hidden_size=32,
        graphormer_layers=1,
        frag_decoder_layers=1,
        frag_encoder_layers=0,
        inten_decoder_layers=1,
        inten_encoder_layers=1,
        node_feats=dataset.get_node_feats(),
        edge_feats=tree_processor.get_edge_feats(),
        multi_hop_max_dist=3,
        max_breakpoints=8,
        embed_adduct=True,
        embed_collision=True,
        embed_elem_group=True,
        embed_instrument=True,
        encode_forms=True,
        enable_aux_loss=True,
        warmup=2,
        magma_warmup_steps=2,
        magma_decay_steps=2,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader, loader)
    assert trainer.global_step == 1
