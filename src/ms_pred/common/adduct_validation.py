"""Explicit checkpoint support declarations, independent of training data."""

import copy
from numbers import Integral
from pathlib import Path
import warnings

from . import chem_utils
from .chem_utils import instrument2onehot_pos, ion2onehot_pos, onehot_pos2ion


def canonical_instrument(value):
    normalizer = getattr(chem_utils, "canonical_instrument", None)
    if normalizer is not None:
        return normalizer(value)
    if not isinstance(value, str):
        return None
    names = {name.casefold(): name for name in instrument2onehot_pos}
    aliases = {"q-tof": "QTOF", "itft": "IT-FT"}
    key = value.strip().casefold()
    return names.get(key, aliases.get(key))


class UntrainedAdductError(ValueError):
    """The checkpoint does not support the requested inference adduct."""


class UntrainedInstrumentError(ValueError):
    """The checkpoint does not support the requested inference instrument."""


class AdductTrainingMetadataWarning(UserWarning):
    """A legacy checkpoint has incomplete support metadata."""


def declared_training_metadata(adducts, instruments):
    """Validate explicit declarations; never inspect or infer from a dataset."""
    for name, values in (("supported-adducts", adducts), ("supported-instruments", instruments)):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"Declare a nonempty {name} list in the training YAML "
                             f"(or pass --{name} on the command line).")
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"{name} must contain type names as strings.")
    invalid = [a for a in adducts if a not in ion2onehot_pos]
    if invalid:
        raise ValueError(f"Unsupported adduct declaration: {invalid}.")
    normalized = [canonical_instrument(i) for i in instruments]
    if any(i is None for i in normalized):
        raise ValueError(f"Unsupported instrument declaration: {instruments}. "
                         f"Valid instrument types: {list(instrument2onehot_pos)}.")
    return {"schema_version": 2,
            "supported_adducts": sorted(set(adducts)),
            "supported_instruments": sorted(set(normalized)),
            "provenance": {"method": "explicit_declaration"}}


def validate_training_metadata(metadata):
    """Read current declarations and the earlier adduct-only checkpoint schema."""
    if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int:
        raise ValueError("Invalid checkpoint training_metadata schema version.")
    if metadata["schema_version"] == 2:
        result = copy.deepcopy(metadata)
        declaration = declared_training_metadata(metadata.get("supported_adducts"),
                                                 metadata.get("supported_instruments"))
        for key in ("supported_adducts", "supported_instruments"):
            result[key] = declaration[key]
        return result
    if metadata["schema_version"] != 1:
        raise ValueError("Invalid checkpoint training_metadata schema; expected version 1 or 2.")
    adducts = metadata.get("training_adducts")
    if (not isinstance(adducts, list) or not adducts
            or any(not isinstance(a, str) or a not in ion2onehot_pos for a in adducts)):
        raise ValueError("training_metadata.training_adducts must be a nonempty list of supported adducts.")
    if type(metadata.get("training_adducts_complete")) is not bool:
        raise ValueError("training_metadata.training_adducts_complete must be a boolean.")
    return copy.deepcopy(metadata)


def add_supported_condition_args(parser):
    for name in ("adducts", "instruments"):
        parser.add_argument(f"--supported-{name}", nargs="+", default=None,
                            help=f"Explicit supported {name}; required for training and saved in checkpoints.")


def validate_training_declarations(kwargs):
    if not kwargs.get("test_checkpoint"):
        declared_training_metadata(kwargs.get("supported_adducts"), kwargs.get("supported_instruments"))


def read_support_declaration(path):
    """Read an explicit declaration from a flat YAML file or launcher YAML."""
    import yaml

    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML mapping containing explicit support declarations.")
    values = config.get("universal_args", config)
    if "universal_args" in config:
        keys = {"supported-adducts", "supported-instruments"}
        if any(keys.intersection(entry) for entry in config.get("iterative_args", [])):
            raise ValueError("Support declarations vary by iterative entry; provide a flat declaration YAML.")
        values = dict(values)
        for key in keys:
            options = values.get(key)
            if not isinstance(options, list) or len(options) != 1:
                raise ValueError(f"Declare one support list under universal_args.{key}.")
            values[key] = options[0]
    return declared_training_metadata(values.get("supported-adducts"), values.get("supported-instruments"))


def _requested_ids(values, mapping, kind):
    if hasattr(values, "detach"):
        values = values.detach().unique().cpu().tolist()
    if isinstance(values, (str, Integral)):
        values = [values]
    requested = set()
    for value in values:
        if isinstance(value, str):
            name = canonical_instrument(value) if kind == "instrument" else value
            if name not in mapping:
                raise ValueError(f"Unsupported {kind} {value!r} in the model vocabulary.")
            requested.add(mapping[name])
        elif value in mapping.values():
            requested.add(value)
        else:
            raise ValueError(f"Unsupported {kind} index {value!r} in the model vocabulary.")
    return requested


class TrainingConditionsMixin:
    """Save declared support in Lightning checkpoints and enforce it at inference."""

    @property
    def supported_adducts(self):
        metadata = getattr(self, "_training_metadata", None)
        if metadata is None:
            return None
        return list(metadata.get("supported_adducts", metadata.get("training_adducts", [])))

    @property
    def training_adducts(self):
        return self.supported_adducts

    @property
    def supported_instruments(self):
        metadata = getattr(self, "_training_metadata", None)
        return None if metadata is None or metadata["schema_version"] == 1 else list(metadata["supported_instruments"])

    def set_supported_conditions(self, adducts, instruments):
        declaration = declared_training_metadata(adducts, instruments)
        self._declared_training_metadata = declaration
        self._training_metadata = copy.deepcopy(declaration)
        self.hparams.update(supported_adducts=declaration["supported_adducts"],
                            supported_instruments=declaration["supported_instruments"])
        self._support_metadata_warned = set()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        metadata = checkpoint.get("training_metadata")
        self._training_metadata = None if metadata is None else validate_training_metadata(metadata)
        self._support_metadata_warned = set()

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        metadata = getattr(self, "_training_metadata", None)
        if metadata is not None:
            checkpoint["training_metadata"] = copy.deepcopy(metadata)

    def on_train_start(self):
        super().on_train_start()
        declaration = getattr(self, "_declared_training_metadata", None)
        if declaration is None:
            raise ValueError("Training requires explicit supported-adducts and supported-instruments "
                             "declarations in YAML. Dataset detection is disabled.")
        # The current YAML is authoritative, also after resume or fine-tuning.
        # Do not discover, union, or broaden support from data or history.
        self._training_metadata = copy.deepcopy(declaration)

    def _warn_missing_support(self, kind):
        warned = getattr(self, "_support_metadata_warned", set())
        if kind not in warned:
            warnings.warn(
                f"This checkpoint has missing or incomplete {kind} support metadata; "
                "support cannot be verified. Upgrade it using "
                "python -m ms_pred.common.annotate_checkpoint_adducts with explicit declarations.",
                AdductTrainingMetadataWarning, stacklevel=3)
            self._support_metadata_warned = warned | {kind}

    def validate_adducts(self, adducts):
        if adducts is None:
            return
        requested = _requested_ids(adducts, ion2onehot_pos, "adduct")
        metadata = getattr(self, "_training_metadata", None)
        if metadata is None or (metadata["schema_version"] == 1 and not metadata["training_adducts_complete"]):
            self._warn_missing_support("adduct")
            return
        allowed = self.supported_adducts
        unsupported = requested - {ion2onehot_pos[a] for a in allowed}
        if unsupported:
            names = ", ".join(sorted(onehot_pos2ion[a] for a in unsupported))
            if metadata["schema_version"] == 1:
                reason, label = "were not seen in training", "Trained adducts"
            else:
                reason, label = "are not declared as supported", "Declared supported adducts"
            raise UntrainedAdductError(f"Adduct(s) {names} {reason} for this checkpoint. "
                                      f"{label}: {', '.join(allowed)}. "
                                      "Use weights that support the requested adduct(s).")

    def validate_instruments(self, instruments):
        if instruments is None:
            return
        requested = _requested_ids(instruments, instrument2onehot_pos, "instrument")
        allowed = self.supported_instruments
        if allowed is None:
            self._warn_missing_support("instrument")
            return
        unsupported = requested - {instrument2onehot_pos[i] for i in allowed}
        if unsupported:
            reverse = {v: k for k, v in instrument2onehot_pos.items()}
            names = ", ".join(sorted(reverse[i] for i in unsupported))
            raise UntrainedInstrumentError(
                f"Instrument type(s) {names} are not declared as supported for this checkpoint. "
                f"Declared supported instruments: {', '.join(allowed)}. "
                "Use weights that support the requested instrument type(s).")


# Compatibility with imports from the initial adduct-only implementation.
TrainingAdductsMixin = TrainingConditionsMixin


def _checkpoint_support(path):
    """Read metadata without constructing a model or reading mapped tensors."""
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("training_metadata")
    validator = TrainingConditionsMixin()
    validator._training_metadata = None if metadata is None else validate_training_metadata(metadata)
    return validator


def validate_checkpoint_conditions(checkpoint, adducts, instruments):
    """Apply the same policy before a wrapper returns cached predictions."""
    path = Path(checkpoint).resolve()
    # Re-read the lightweight mapped metadata. Filesystems can report identical
    # size and timestamps after a rapid checkpoint replacement.
    validator = _checkpoint_support(str(path))
    validator.validate_adducts(adducts)
    validator.validate_instruments(instruments)
