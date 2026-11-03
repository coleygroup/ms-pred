"""Training provenance must survive checkpoints and gate all inference paths."""

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path
import warnings

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from ms_pred.common.adduct_validation import (
    AdductTrainingMetadataWarning, TrainingAdductsMixin, UntrainedAdductError,
    UntrainedInstrumentError, declared_training_metadata, read_support_declaration,
    add_supported_condition_args, validate_training_declarations,
    validate_checkpoint_conditions,
)
from ms_pred.common.annotate_checkpoint_adducts import annotate_checkpoint
from ms_pred.common.chem_utils import ion2onehot_pos, instrument2onehot_pos


class RetainedTrees(Dataset):
    @property
    def spec_names(self):
        raise AssertionError("Support declarations must never inspect training metadata.")

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor(1.0)


class TinyModel(TrainingAdductsMixin, pl.LightningModule):
    def __init__(self, supported_adducts=None, supported_instruments=None):
        super().__init__()
        self.save_hyperparameters()
        if supported_adducts is not None or supported_instruments is not None:
            self.set_supported_conditions(supported_adducts, supported_instruments)
        self.weight = torch.nn.Parameter(torch.ones(1))

    def training_step(self, batch, batch_idx):
        return (self.weight * batch).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def checkpoint(adducts=("[M+H]+", "[M+Na]+"), complete=True):
    return {"training_metadata": {
        "schema_version": 1, "training_adducts": list(adducts),
        "training_adducts_complete": complete,
    }}


def test_training_checkpoint_uses_exact_declaration_without_dataset_detection(tmp_path):
    model = TinyModel(["[M+H]+", "[M+Na]+"], ["Orbitrap"])
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, DataLoader(RetainedTrees()))
    path = tmp_path / "best.ckpt"
    trainer.save_checkpoint(path)
    loaded = TinyModel.load_from_checkpoint(path)
    assert loaded.supported_adducts == ["[M+H]+", "[M+Na]+"]
    assert loaded.supported_instruments == ["Orbitrap"]
    loaded.validate_adducts("[M+H]+")
    with pytest.raises(UntrainedAdductError, match="not declared as supported"):
        loaded.validate_adducts("[M-H]-")


@pytest.mark.parametrize("adducts", ["[M-H]-", ["[M+H]+", "[M-H]-"],
    ("[M-H]-",), torch.tensor([ion2onehot_pos["[M-H]-"]])])
def test_unseen_adduct_rejected_for_scalar_batch_and_encoded_input(adducts):
    model = TinyModel()
    model.on_load_checkpoint(checkpoint())
    with pytest.raises(UntrainedAdductError, match="Trained adducts:.*M\\+H"):
        model.validate_adducts(adducts)


def test_equivalent_adduct_spellings_are_accepted():
    model = TinyModel()
    model.on_load_checkpoint(checkpoint(["[M+NH4]+", "[M+H-H2O]+", "[M+H]+"]))
    model.validate_adducts(["[M+H3N+H]+", "[M-H2O+H]+", "[M+H]1+"])


def test_legacy_checkpoint_warns_once_without_claiming_adduct_was_unseen():
    model = TinyModel()
    model.on_load_checkpoint({})
    with pytest.warns(AdductTrainingMetadataWarning, match="cannot be verified"):
        model.validate_adducts("[M-H]-")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.validate_adducts("[M-H]-")
    assert not caught
    with pytest.raises(ValueError, match="Unsupported adduct"):
        model.validate_adducts("nonsense")


def test_yaml_declaration_is_authoritative_after_resume_or_finetune():
    model = TinyModel(["[M+H]+"], ["Orbitrap"])
    # Resume loads the previous policy after constructing the current model.
    model.on_load_checkpoint({"training_metadata": declared_training_metadata(["[M-H]-"], ["QTOF"])})
    model.on_train_start()
    assert model.supported_adducts == ["[M+H]+"]
    assert model.supported_instruments == ["Orbitrap"]
    with pytest.raises(UntrainedAdductError):
        model.validate_adducts("[M-H]-")
    with pytest.raises(UntrainedInstrumentError):
        model.validate_instruments("QTOF")


def test_training_without_explicit_declarations_fails():
    with pytest.raises(ValueError, match="Dataset detection is disabled"):
        TinyModel().on_train_start()


def test_upgrade_preserves_checkpoint_and_model_tensors(tmp_path):
    original = tmp_path / "old.ckpt"
    output = tmp_path / "new.ckpt"
    torch.save({"state_dict": {"weight": torch.arange(4.0)},
                "optimizer_states": [{"step": 42}], "hyper_parameters": {"x": 3}}, original)
    original_bytes = original.read_bytes()
    annotate_checkpoint(original, output, ["[M+H]+", "[M+Na]+"], "author declaration", ["Orbitrap"])
    assert original.read_bytes() == original_bytes
    upgraded = torch.load(output, weights_only=False)
    assert torch.equal(upgraded["state_dict"]["weight"], torch.arange(4.0))
    assert upgraded["optimizer_states"] == [{"step": 42}]
    assert upgraded["hyper_parameters"] == {"x": 3}
    model = TinyModel()
    model.on_load_checkpoint(upgraded)
    with pytest.raises(UntrainedAdductError):
        model.validate_adducts("[M-H]-")
    with pytest.raises(ValueError, match="new path"):
        annotate_checkpoint(original, original, ["[M+H]+"], "verified")
    with pytest.raises(ValueError, match="new path"):
        annotate_checkpoint(original, output, ["[M+H]+"], "verified")


@pytest.mark.parametrize("bad", [{}, {"schema_version": 2},
    {"schema_version": 1, "training_adducts": ["nonsense"], "training_adducts_complete": True},
    {"schema_version": 1, "training_adducts": [], "training_adducts_complete": True}])
def test_malformed_metadata_is_rejected(bad):
    with pytest.raises(ValueError):
        TinyModel().on_load_checkpoint({"training_metadata": bad})


@pytest.mark.parametrize("kind", ["iceberg_gen", "iceberg_inten", "glacier"])
def test_production_checkpoint_load_and_inference_guards(tmp_path, kind):
    if kind == "iceberg_gen":
        from ms_pred.iceberg.gen_model import FragGNN
        model = FragGNN(hidden_size=8, layers=1, set_layers=1)
        calls = [lambda m: m.predict_mol("CCO", 20, 47, adduct="[M-H]-")]
    elif kind == "iceberg_inten":
        from ms_pred.iceberg.inten_model import IntenGNN
        model = IntenGNN(hidden_size=8, gnn_layers=1, set_layers=1)
        calls = [lambda m: m.predict(None, None, None, None, None,
                                    adducts=torch.tensor([ion2onehot_pos["[M-H]-"]]))]
    else:
        from ms_pred.glacier.joint_model import JointModel
        model = JointModel(hidden_size=8, frag_decoder_layers=1, inten_decoder_layers=1,
                           inten_encoder_layers=0, graphormer_layers=1, max_breakpoints=2,
                           encode_forms=True)
        batch = {"adducts": torch.tensor([ion2onehot_pos["[M-H]-"]])}
        calls = [lambda m: m.predict_mol("CCO", 20, "[M-H]-", "cpu"),
                 lambda m: m.predict_inten_frag_batch(batch),
                 lambda m: m.predict_inten_frag_batch_sparse(batch, sparse_k=5)]
    data = {"training_metadata": declared_training_metadata(["[M+H]+", "[M+Na]+"], ["Orbitrap"])}
    data.update(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams),
                **{"pytorch-lightning_version": pl.__version__})
    path = tmp_path / f"{kind}.ckpt"
    torch.save(data, path)
    loaded = type(model).load_from_checkpoint(path)
    for call in calls:
        with pytest.raises(UntrainedAdductError):
            call(loaded)
    if kind == "glacier" and hasattr(type(model), "load_from_checkpoint_flexible"):
        flexible = type(model).load_from_checkpoint_flexible(path)
        with pytest.raises(UntrainedAdductError):
            flexible.predict_mol("CCO", 20, "[M-H]-", "cpu")


def test_iceberg_checks_both_weights_before_generation(monkeypatch):
    from ms_pred.iceberg.gen_model import FragGNN
    from ms_pred.iceberg.inten_model import IntenGNN
    from ms_pred.iceberg.joint_model import JointModel
    gen = FragGNN(hidden_size=8, layers=1, set_layers=1)
    inten = IntenGNN(hidden_size=8, gnn_layers=1, set_layers=1)
    gen.on_load_checkpoint(checkpoint(["[M+H]+", "[M-H]-"]))
    inten.on_load_checkpoint(checkpoint(adducts=("[M+H]+",)))
    joint = JointModel(gen, inten)

    def should_not_run(*args, **kwargs):
        pytest.fail("Tree generation ran before validating the intensity checkpoint.")

    monkeypatch.setattr(gen, "predict_mol", should_not_run)
    with pytest.raises(UntrainedAdductError):
        joint.predict_mol("CCO", 20, 45, "[M-H]-", 0.1, "cpu", 5)


@pytest.mark.parametrize("instruments", ["QTOF", ["Orbitrap", "QTOF"],
    torch.tensor([instrument2onehot_pos["QTOF"]])])
def test_instrument_declaration_rejects_string_batch_and_encoded_inputs(instruments):
    model = TinyModel(["[M+H]+"], ["Orbitrap"])
    with pytest.raises(UntrainedInstrumentError, match="Declared supported instruments: Orbitrap"):
        model.validate_instruments(instruments)
    model.validate_instruments("orbitrap")


@pytest.mark.parametrize("kind", ["iceberg_gen", "iceberg_inten", "glacier"])
def test_production_instrument_guards(kind):
    conditions = dict(supported_adducts=["[M+H]+"], supported_instruments=["Orbitrap"])
    if kind == "iceberg_gen":
        from ms_pred.iceberg.gen_model import FragGNN
        model = FragGNN(hidden_size=8, layers=1, set_layers=1, **conditions)
        calls = [lambda: model.predict_mol("CCO", 20, 47, "[M+H]+", instrument="QTOF")]
    elif kind == "iceberg_inten":
        from ms_pred.iceberg.inten_model import IntenGNN
        model = IntenGNN(hidden_size=8, gnn_layers=1, set_layers=1, **conditions)
        calls = [lambda: model.predict(None, None, None, None, None,
                    adducts=torch.tensor([0]), instruments=torch.tensor([instrument2onehot_pos["QTOF"]]))]
    else:
        from ms_pred.glacier.joint_model import JointModel
        model = JointModel(hidden_size=8, frag_decoder_layers=1, inten_decoder_layers=1,
                           inten_encoder_layers=0, graphormer_layers=1, max_breakpoints=2,
                           encode_forms=True, **conditions)
        batch = {"adducts": torch.tensor([0]), "instruments": torch.tensor([instrument2onehot_pos["QTOF"]])}
        calls = [lambda: model.predict_mol("CCO", 20, "[M+H]+", "cpu", instrument="QTOF"),
                 lambda: model.predict_inten_frag_batch(batch),
                 lambda: model.predict_inten_frag_batch_sparse(batch, sparse_k=5)]
    for call in calls:
        with pytest.raises(UntrainedInstrumentError):
            call()


def test_legacy_adduct_metadata_warns_for_unknown_instrument_coverage():
    model = TinyModel()
    model.on_load_checkpoint(checkpoint())
    with pytest.warns(AdductTrainingMetadataWarning, match="instrument support"):
        model.validate_instruments("QTOF")


@pytest.mark.parametrize("adducts,instruments", [([], ["Orbitrap"]), (["[M+H]+"], []),
    (["nonsense"], ["Orbitrap"]), (["[M+H]+"], ["nonsense"]), ("[M+H]+", ["Orbitrap"])])
def test_invalid_or_missing_declarations_fail(adducts, instruments):
    with pytest.raises(ValueError):
        declared_training_metadata(adducts, instruments)


def test_launcher_yaml_roundtrips_declaration_lists_through_cli():
    path = Path("configs/glacier/msg_simulation/joint_train_msg_simulation.yaml")
    if not path.exists():
        path = Path("configs/glacier/joint_train_msg.yaml")
    declaration = read_support_declaration(path)
    assert declaration["supported_adducts"] == ["[M+H]+"]
    assert set(declaration["supported_instruments"]) == {"Orbitrap", "QTOF"}
    module_path = Path("launcher_scripts/run_from_config.py")
    spec = importlib.util.spec_from_file_location("support_test_launcher", module_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    command = " ".join(launcher.convert_flag(key.replace("_", "-"), declaration[key])
                       for key in ("supported_adducts", "supported_instruments"))
    assert "'[M+H]+'" in command
    parser = argparse.ArgumentParser()
    add_supported_condition_args(parser)
    args = vars(parser.parse_args(shlex.split(command)))
    validate_training_declarations(args)
    assert args["supported_adducts"] == ["[M+H]+"]
    with pytest.raises(ValueError, match="supported-adducts"):
        validate_training_declarations({})
    validate_training_declarations({"test_checkpoint": "legacy.ckpt"})


def test_flat_yaml_legacy_upgrade_uses_explicit_declarations(tmp_path):
    declaration_path = tmp_path / "support.yaml"
    declaration_path.write_text('supported-adducts: ["[M+H]+"]\nsupported-instruments: [Orbitrap]\n')
    declaration = read_support_declaration(declaration_path)
    original, output = tmp_path / "original.ckpt", tmp_path / "upgraded.ckpt"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, original)
    annotate_checkpoint(original, output, declaration["supported_adducts"],
                        str(declaration_path), declaration["supported_instruments"])
    upgraded = torch.load(output, weights_only=False)
    assert upgraded["training_metadata"]["schema_version"] == 2
    model = TinyModel()
    model.on_load_checkpoint(upgraded)
    with pytest.raises(UntrainedInstrumentError):
        model.validate_instruments("QTOF")


def test_checkpoint_policy_refreshes_when_weights_are_replaced(tmp_path):
    path = tmp_path / "weights.ckpt"
    torch.save({"training_metadata": declared_training_metadata(["[M+H]+"], ["Orbitrap"])}, path)
    validate_checkpoint_conditions(path, ["[M+H]+"], ["Orbitrap"])
    torch.save({"training_metadata": declared_training_metadata(["[M+Na]+"], ["QTOF"])}, path)
    with pytest.raises(UntrainedAdductError):
        validate_checkpoint_conditions(path, ["[M+H]+"], ["Orbitrap"])


@pytest.mark.parametrize("kind", ["iceberg", "glacier"])
def test_prediction_wrappers_reject_unsupported_types_even_with_cached_output(tmp_path, monkeypatch, kind):
    import importlib

    module_name = f"ms_pred.{kind}.{kind}_elucidation"
    if importlib.util.find_spec(module_name) is None:
        pytest.skip("This model version has no prediction wrapper.")
    module = importlib.import_module(module_name)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{kind}_run_successful").touch()
    monkeypatch.setattr(module, "user_cache_dir", lambda _: str(cache))
    path = tmp_path / "weights.ckpt"
    torch.save({"training_metadata": declared_training_metadata(["[M+H]+"], ["Orbitrap"])}, path)
    kwargs = dict(candidate_smiles=["CCO"], collision_energies=[20], adduct="[M-H]-",
                  instrument="Orbitrap", python_path=sys.executable, cuda_devices=None)
    kwargs.update({"gen_ckpt": str(path), "inten_ckpt": str(path)} if kind == "iceberg" else {"ckpt": str(path)})
    with pytest.raises(UntrainedAdductError):
        getattr(module, f"{kind}_prediction")(**kwargs)
    kwargs.update(adduct="[M+H]+", instrument="QTOF")
    with pytest.raises(UntrainedInstrumentError):
        getattr(module, f"{kind}_prediction")(**kwargs)
    kwargs["instrument"] = "Orbitrap"
    directory, _ = getattr(module, f"{kind}_prediction")(**kwargs)
    assert directory == cache
