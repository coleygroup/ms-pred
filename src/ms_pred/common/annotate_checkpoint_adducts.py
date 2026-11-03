"""Embed explicit adduct and instrument support in a new copy of a checkpoint."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from .adduct_validation import declared_training_metadata, read_support_declaration


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotate_checkpoint(checkpoint_path, output_path, adducts, source, instruments=None):
    """Write a separate checkpoint and verify every model tensor after saving.

    Support is explicitly declared by the model author, never detected from data.
    """
    checkpoint_path, output_path = Path(checkpoint_path), Path(output_path)
    if output_path.exists() or output_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Output must be a new path; the original checkpoint is preserved.")
    if not source or not source.strip():
        raise ValueError("A source description for the support declaration is required.")
    metadata = declared_training_metadata(adducts, instruments)
    metadata["provenance"] = {
        "method": "explicit_legacy_declaration",
        "source": source,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError("Expected a Lightning checkpoint containing state_dict.")
    checkpoint["training_metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        restored = torch.load(temporary, map_location="cpu", weights_only=False)
        if restored["state_dict"].keys() != checkpoint["state_dict"].keys():
            raise ValueError("Checkpoint tensor names changed during metadata upgrade.")
        for name, tensor in checkpoint["state_dict"].items():
            torch.testing.assert_close(restored["state_dict"][name], tensor,
                                       rtol=0, atol=0, equal_nan=True)
        # Exclusive creation also protects an output created while verifying.
        os.link(temporary, output_path)
    finally:
        os.unlink(temporary)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adducts", "--supported-adducts", nargs="+",
                        help="Explicit supported adduct list; alternative to --training-config.")
    parser.add_argument("--instruments", "--supported-instruments", nargs="+",
                        help="Explicit supported instrument list; alternative to --training-config.")
    parser.add_argument("--training-config", type=Path,
                        help="Read both declarations from training YAML (no dataset access).")
    parser.add_argument("--source", help="Describe the model author's support declaration.")
    args = parser.parse_args()
    if args.training_config:
        if args.adducts is not None or args.instruments is not None:
            parser.error("Use --training-config or explicit lists, not both.")
        declaration = read_support_declaration(args.training_config)
        args.adducts = declaration["supported_adducts"]
        args.instruments = declaration["supported_instruments"]
        source = f"{args.training_config} (SHA-256 {sha256_file(args.training_config)})"
    else:
        source = args.source
    metadata = annotate_checkpoint(args.checkpoint, args.output, args.adducts, source, args.instruments)
    print(f"Wrote {args.output}; all model tensors verified unchanged.")
    print(f"Supported adducts: {', '.join(metadata['supported_adducts'])}")
    print(f"Supported instruments: {', '.join(metadata['supported_instruments'])}")


if __name__ == "__main__":
    main()
