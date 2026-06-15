#!/usr/bin/env python3
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple, Optional
import threading
import time
import tempfile
import uuid
import os
import json
import secrets
import string
import smtplib
import sqlite3
import ipaddress
import functools
from datetime import datetime, timezone
from email.message import EmailMessage
from dataclasses import dataclass, field
from http.client import RemoteDisconnected
from urllib.error import URLError

from flask import Flask, render_template, request, send_file, flash, redirect, url_for, abort, jsonify, Response, stream_with_context, get_flashed_messages
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pubchempy as pcp
import yaml

from ms_pred import common
from ms_pred.common import MassSpec, CompositeMassSpec
import ms_pred.magma.fragmentation as fragmentation

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MGF_DIR = Path(os.environ.get("MSPRED_ATLAS_DIR", "/home/coley-group/atlas/"))
ADDUCT_TO_DIR = {
    '[M+H]+': BASE_MGF_DIR / 'h_plus_out_mgf',
    '[M-H]-': BASE_MGF_DIR / 'h_minus_out_mgf',
}

# NIST'23 atlas (gated — only authenticated internal users see it)
_NIST_DIR_ENV = os.environ.get("MSPRED_ATLAS_DIR_NIST", "")
NIST_MGF_DIR: Path | None = Path(_NIST_DIR_ENV) if _NIST_DIR_ENV else None
NIST_ADDUCT_TO_DIR = (
    {
        '[M+H]+': NIST_MGF_DIR / 'h_plus_out_mgf',
        '[M-H]-': NIST_MGF_DIR / 'h_minus_out_mgf',
    }
    if NIST_MGF_DIR is not None
    else {}
)

# Internal-user store (YAML file: email -> nested record with password hash + role)
ICEBERG_USERS_FILE: Path | None = (
    Path(os.environ["ICEBERG_USERS_FILE"])
    if "ICEBERG_USERS_FILE" in os.environ
    else None
)

# Comma-separated list of emails that always have admin privilege (safety net)
_ADMIN_EMAILS_ENV: set = {
    e.strip().lower()
    for e in os.environ.get("ICEBERG_ADMIN_EMAILS", "").split(",")
    if e.strip()
}

# SMTP config for sending registration emails (all optional — if unset, email is skipped)
_SMTP_HOST: str = os.environ.get("SMTP_HOST", "")
_SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
_SMTP_USER: str = os.environ.get("SMTP_USER", "")
_SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
_SMTP_FROM: str = os.environ.get("SMTP_FROM", _SMTP_USER)
_SMTP_USE_TLS: bool = os.environ.get("SMTP_USE_TLS", "true").lower() not in ("0", "false", "no")
# Contact address shown in outgoing emails and the atlas header — defaults to SMTP_FROM if unset
_CONTACT_EMAIL: str = os.environ.get("ICEBERG_CONTACT_EMAIL", _SMTP_FROM)
# Full email templates for credential emails.  Placeholders: {email}, {password}, {contact}.
# Use \n for newlines when setting via an environment variable.
# Defaults are intentionally generic — open-source adopters should override these.
_EMAIL_SUBJECT_NEW: str = os.environ.get(
    "ICEBERG_EMAIL_SUBJECT_NEW",
    "Access to the ICEBERG Mass Spectrometry Atlas",
)
_EMAIL_BODY_NEW: str = os.environ.get("ICEBERG_EMAIL_BODY_NEW", "").replace("\\n", "\n") or (
    "Hello,\n\n"
    "You have been granted access to the ICEBERG Mass Spectrometry Atlas,\n"
    "a tool for exploring predicted MS2 spectra.\n\n"
    "Username:    {email}\n"
    "Password:    {password}\n\n"
    "Please log in and use the \"Change password\" link to set your own password.\n\n"
    "For any questions, reach out to {contact}.\n"
)
_EMAIL_SUBJECT_RESET: str = os.environ.get(
    "ICEBERG_EMAIL_SUBJECT_RESET",
    "ICEBERG Mass Spectrometry Atlas — password reset",
)
_EMAIL_BODY_RESET: str = os.environ.get("ICEBERG_EMAIL_BODY_RESET", "").replace("\\n", "\n") or (
    "Hello,\n\n"
    "Your password for the ICEBERG Mass Spectrometry Atlas has been reset by an\n"
    "administrator. Your new temporary login credentials are below.\n\n"
    "Username:    {email}\n"
    "Password:    {password}\n\n"
    "Please log in and use the \"Change password\" link to set your own password.\n\n"
    "For any questions, reach out to {contact}.\n"
)

# SQLite analytics database
_ANALYTICS_DB_PATH: Path | None = (
    Path(os.environ["ICEBERG_ANALYTICS_DB"])
    if "ICEBERG_ANALYTICS_DB" in os.environ
    else (ICEBERG_USERS_FILE.parent / "analytics.db" if ICEBERG_USERS_FILE else None)
)
_analytics_db_lock = threading.Lock()

# MaxMind GeoLite2 database for geo-IP resolution (optional)
_GEOIP_DB_PATH: str = os.environ.get("GEOIP_DB", "")

# Example spectrum
EXAMPLE_MS_PATH = (
    Path(__file__).resolve().parent.parent / "data/exp_specs/clinical/lpc19-0_standard.ms"
)

# Collision energy matching tolerance (in eV)
EV_TOLERANCE = 1.0

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def round_ev(ev: Any) -> int:
    """Round a collision-energy value to the nearest integer eV."""
    return int(round(float(ev)))


def get_formula_subdir(formula: str, adduct='[M+H]+') -> Path:
    """
    Wrapper around common.get_formula_subdir to return a Path (public atlas).
    """
    subdir = common.get_formula_subdir(formula)
    return ADDUCT_TO_DIR[adduct] / subdir


def get_mgf_path_for_formula(formula: str, adduct='[M+H]+') -> Path:
    """
    Get the .mgf path for a given chemical formula (public atlas only).
    """
    subdir = get_formula_subdir(formula, adduct)
    mgf_path = subdir / f"{formula}.mgf"
    return mgf_path


def mgf_paths_for_formula(
    formula: str,
    adduct: str = '[M+H]+',
    include_nist: bool = False,
) -> List[Path]:
    """
    Return ordered list of existing .mgf paths for a formula.
    Public atlas is always first; NIST atlas is appended when include_nist=True
    and MSPRED_ATLAS_DIR_NIST is configured.
    """
    paths: List[Path] = []

    pub_path = get_mgf_path_for_formula(formula, adduct)
    if pub_path.exists():
        paths.append(pub_path)

    if include_nist and NIST_MGF_DIR is not None and adduct in NIST_ADDUCT_TO_DIR:
        nist_subdir = common.get_formula_subdir(formula)
        nist_path = NIST_ADDUCT_TO_DIR[adduct] / nist_subdir / f"{formula}.mgf"
        if nist_path.exists():
            paths.append(nist_path)

    return paths


def parse_user_spectrum(text: str) -> np.ndarray:
    """
    Parse pasted spectrum text into an (N, 2) numpy array: [m/z, intensity].
    Lines must be 'mz intensity'.
    """
    mz_int_list = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            mz = float(parts[0])
            inten = float(parts[1])
        except ValueError:
            # ignore malformed/non-numeric lines
            continue
        mz_int_list.append((mz, inten))

    if not mz_int_list:
        raise ValueError("No valid peaks found in the pasted spectrum.")

    arr = np.array(mz_int_list, dtype=float)
    return arr


def prepare_user_spectra(
    formula: str,
    adduct: str,
    texts: List[str],
    nces: List[str],
) -> CompositeMassSpec:
    """
    Parse user spectra texts, compute precursor m/z from formula + adduct,
    convert NCE to eV using common.nce_to_ev, and return a mapping:
    {ev_value: MassSpec}.
    """
    precursor_mz = common.formula_mass(formula) + common.ion2mass[adduct]

    all_specs: List[MassSpec] = []

    for spec_text, nce in zip(texts, nces):
        spec_text = spec_text.strip()
        nce = nce.strip()
        if not spec_text or not nce:
            continue

        peaks = parse_user_spectrum(spec_text)
        masses = peaks[:, 0]
        intens = peaks[:, 1]

        ev_value = round_ev(common.nce_to_ev(nce, precursor_mz))

        spec_obj = MassSpec(
            collision_energy=ev_value,
            root_canonical_smiles=None,
            adduct=adduct,
            remark="User query",
            masses=masses,
            intens=intens,
        )
        # Cache the original NCE on the MassSpec so we never have to
        # recompute it from eV later.
        try:
            spec_obj.nce = float(nce)
        except Exception:
            pass
        all_specs.append(spec_obj)

    if not all_specs:
        raise ValueError("No valid user spectra were provided.")

    ev_to_spec = CompositeMassSpec(all_specs)
    return ev_to_spec


def smiles_to_lookup_keys(smiles: str) -> Dict[str, str]:
    """
    Parse a SMILES string and return canonical SMILES, molecular formula,
    and InChIKey for atlas lookup.

    Canonicalization matches what joint_model.predict_mol does when building
    the atlas: SMILES -> InChI -> tautomer-canonical SMILES (via
    common.smiles_from_inchi). Without this round-trip, tautomer-equivalent
    user input misses atlas entries whose InChIKey was computed post-canon.
    """
    raw_mol = Chem.MolFromSmiles(smiles)
    if raw_mol is None:
        raise ValueError("Invalid SMILES")

    inchi = common.inchi_from_smiles(smiles)
    canonical_smiles = common.smiles_from_inchi(inchi) if inchi else ""
    if not canonical_smiles:
        canonical_smiles = Chem.MolToSmiles(raw_mol)
    canon_mol = Chem.MolFromSmiles(canonical_smiles) or raw_mol

    return {
        "canonical_smiles": canonical_smiles,
        "formula": rdMolDescriptors.CalcMolFormula(canon_mol),
        "inchikey": Chem.MolToInchiKey(canon_mol),
    }


def _canonicalize_smiles_safe(smi: str) -> str:
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def _load_mgf_index(mgf_path: Path) -> Dict[str, List[List[int]]] | None:
    """
    Load the .idx sidecar for an MGF file, or return None if absent/unreadable.
    The sidecar is produced by run_scripts/iceberg_atlas/04_build_index.py and
    has the shape: { "<INCHIKEY>": [[byte_offset, block_length], ...], ... }.
    """
    idx_path = mgf_path.with_suffix(".mgf.idx")
    if not idx_path.exists():
        return None
    try:
        with open(idx_path, "r") as fh:
            return json.load(fh)
    except Exception:
        return None


def _parse_mgf_block_text(text: str) -> Tuple[Dict[str, str], np.ndarray] | Tuple[None, None]:
    """
    Parse a single BEGIN IONS … END IONS text chunk into (meta, peaks).
    Shared by the full-scan and the indexed reader so the parsing logic
    lives in exactly one place.
    """
    meta: Dict[str, str] = {}
    peaks: List[Tuple[float, float]] = []
    in_block = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "BEGIN IONS":
            in_block = True
            continue
        if line == "END IONS":
            break
        if not in_block:
            continue
        if "=" in line and not line[0].isdigit():
            k, _, v = line.partition("=")
            meta[k] = v
        else:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass

    if not peaks:
        return None, None
    return meta, np.asarray(peaks, dtype=float)


def inchikey_in_mgf(mgf_path: Path, inchikey: str) -> bool:
    """
    Cheap existence check.  When an .idx sidecar is present, this is O(1)
    (key lookup in the loaded JSON).  Falls back to a full line-scan otherwise.
    """
    if not inchikey:
        return False

    idx = _load_mgf_index(mgf_path)
    if idx is not None:
        return inchikey in idx

    needle = f"INCHIKEY={inchikey}"
    try:
        with open(mgf_path, "r") as f:
            for line in f:
                if needle in line:
                    return True
    except OSError:
        return False
    return False


def extract_mgf_blocks_by_inchikey(
    mgf_path: Path, inchikey: str
) -> List[Tuple[Dict[str, str], np.ndarray]]:
    """
    Retrieve all MGF blocks for `inchikey` from `mgf_path`.

    Fast path: if a .idx sidecar exists, seeks directly to each block's byte
    range (O(1) per block) instead of scanning the whole file.
    Fallback: full single-pass line scan when no sidecar is available.
    """
    # ---- indexed fast path --------------------------------------------------
    idx = _load_mgf_index(mgf_path)
    if idx is not None:
        ranges = idx.get(inchikey)
        if not ranges:
            return []
        blocks: List[Tuple[Dict[str, str], np.ndarray]] = []
        try:
            with open(mgf_path, "rb") as f:
                for offset, length in ranges:
                    f.seek(offset)
                    chunk = f.read(length).decode("utf-8", errors="replace")
                    meta, peaks = _parse_mgf_block_text(chunk)
                    if meta is not None and peaks is not None:
                        blocks.append((meta, peaks))
        except OSError:
            pass
        return blocks

    # ---- full-scan fallback -------------------------------------------------
    needle = f"INCHIKEY={inchikey}"
    blocks_scan: List[Tuple[Dict[str, str], np.ndarray]] = []

    in_block = False
    meta: Dict[str, str] = {}
    peaks: List[Tuple[float, float]] = []
    matches = False

    with open(mgf_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                in_block = True
                meta = {}
                peaks = []
                matches = False
                continue
            if line == "END IONS":
                if matches and peaks:
                    blocks_scan.append((meta, np.asarray(peaks, dtype=float)))
                in_block = False
                continue
            if not in_block:
                continue
            if "=" in line and not line[0].isdigit():
                k, _, v = line.partition("=")
                meta[k] = v
                if line == needle:
                    matches = True
                continue
            # peak row
            if matches:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass

    return blocks_scan


def build_composite_from_blocks(
    blocks: List[Tuple[Dict[str, str], np.ndarray]]
) -> Tuple[Dict[str, Any], CompositeMassSpec] | Tuple[None, None]:
    """
    Convert raw (meta, peaks) blocks for a single structure into
    (representative_meta, CompositeMassSpec). Mirrors the per-block
    logic of load_library_spectra_grouped but skips the grouping step
    since all blocks already share the same structure.
    """
    if not blocks:
        return None, None

    specs: List[MassSpec] = []
    rep_meta: Dict[str, Any] = {}

    for meta, peaks in blocks:
        if len(peaks) == 0:
            continue
        ce_str = meta.get("COLLISION_ENERGY")
        if ce_str is None:
            continue
        try:
            ce_val = round_ev(ce_str)
        except ValueError:
            continue

        masses = peaks[:, 0]
        intens = peaks[:, 1]

        frags_raw = meta.get("FRAGS")
        int_frags = None
        if frags_raw is not None:
            try:
                frag_tokens = [
                    tok for tok in frags_raw.replace(",", " ").split() if tok
                ]
                frags = [int(tok) for tok in frag_tokens]
                if len(frags) == len(masses):
                    int_frags = np.asarray(frags, dtype=int)
            except Exception:
                int_frags = None

        ms = MassSpec(
            collision_energy=ce_val,
            root_canonical_smiles=meta.get("SMILES"),
            adduct=meta.get("ADDUCT"),
            remark=meta.get("DESCRIPTION"),
            masses=masses,
            intens=intens,
            int_frags=int_frags,
            **{
                k: v
                for k, v in meta.items()
                if k not in ["SMILES", "ADDUCT", "DESCRIPTION", "COLLISION_ENERGY"]
            },
        )
        specs.append(ms)

        if not rep_meta:
            rep_meta = dict(meta)
            rep_meta.pop("COLLISION_ENERGY", None)

    if not specs:
        return None, None
    return rep_meta, CompositeMassSpec(specs)


def find_structure_in_library(
    grouped: Dict[str, Dict[str, Any]],
    inchikey: str,
    canonical_smiles: str,
) -> Dict[str, Any] | None:
    """
    Locate a structure in a grouped library dict by InChIKey (preferred)
    or canonical SMILES.
    """
    if inchikey:
        for info in grouped.values():
            if info["meta"].get("INCHIKEY") == inchikey:
                return info
    if canonical_smiles:
        for info in grouped.values():
            lib_smi = info["meta"].get("SMILES", "")
            if lib_smi == canonical_smiles:
                return info
            if _canonicalize_smiles_safe(lib_smi) == canonical_smiles:
                return info
    return None


def _structure_key(meta: Dict[str, Any]) -> str:
    """
    Build a key that identifies a unique structure across collision energies.

    Prefer INCHIKEY; fall back to combination of formula, SMILES, and adduct.
    """
    inchikey = meta.get("INCHIKEY")
    if inchikey:
        return inchikey

    formula = meta.get("FORMULA", "")
    smiles = meta.get("SMILES", "")
    adduct = meta.get("ADDUCT", "")
    return f"{formula}|{smiles}|{adduct}"


def load_library_spectra_grouped(
    mgf_path: Path,
    progress_cb=None,
    interested_ces=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load predicted spectra from an .mgf file and group them by structure.

    Returns:
        dict keyed by structure_key with:
        {
          "meta": representative_meta_for_structure (no COLLISION_ENERGY),
          "spectra": List[MassSpec],
        }
    """
    if not mgf_path.exists():
        raise FileNotFoundError(f"MGF file not found: {mgf_path.stem}")

    parsed = common.parse_spectra_mgf(str(mgf_path))
    if progress_cb:
        progress_cb("load_mgf", 20, "Loading predicted spectra (.mgf)")

    grouped: Dict[str, Dict[str, Any]] = {}
    total = max(1, len(parsed))

    for idx, (meta, peaks) in enumerate(parsed, start=1):
        if len(peaks) == 0:
            continue

        ce_str = meta.get("COLLISION_ENERGY", None)
        if ce_str is None:
            continue
        try:
            ce_val = round_ev(ce_str)
            if interested_ces is not None:
                if str(ce_val) not in interested_ces:
                    continue
        except ValueError:
            continue

        masses = peaks[:, 0]
        intens = peaks[:, 1]

        frags_raw = meta.get("FRAGS")
        if frags_raw is not None:
            # FRAGS like: "65535,65520,31,65520,..."
            try:
                # Split on comma or whitespace
                frag_tokens = [
                    tok for tok in frags_raw.replace(",", " ").split() if tok
                ]
                frags = [int(tok) for tok in frag_tokens]
                if len(frags) == len(masses):
                    # attach as numpy array or list – explain_peaks expects something iterable
                    int_frags = np.asarray(frags, dtype=int)
                else:
                    # length mismatch – safest is to skip
                    int_frags = None
            except Exception:
                int_frags = None
        else:
            int_frags = None

        ms = MassSpec(
            collision_energy=ce_val,
            root_canonical_smiles=meta.get("SMILES"),
            adduct=meta.get("ADDUCT"),
            remark=meta.get("DESCRIPTION"),
            masses=masses,
            intens=intens,
            int_frags=int_frags,
            **{
                k: v
                for k, v in meta.items()
                if k not in ["SMILES", "ADDUCT", "DESCRIPTION", "COLLISION_ENERGY"]
            },
        )

        key = _structure_key(meta)
        if key not in grouped:
            # store a representative meta for the structure (without COLLISION_ENERGY)
            rep_meta = dict(meta)
            rep_meta.pop("COLLISION_ENERGY", None)
            grouped[key] = {
                "meta": rep_meta,
                "spectra": [],
            }

        grouped[key]["spectra"].append(ms)

        if progress_cb and (idx % 25 == 0 or idx == total):
            pct = int(idx / total * 80 + 20)
            progress_cb("load_mgf", pct, f"Translating ICEBERG spectra ({idx}/{total})")

    # build CompositeMassSpec objects
    for v in grouped.values():
        if len(v['spectra']) > 0:
            v['spectra'] = CompositeMassSpec(v['spectra'])

    return grouped


def _massspec_to_peak_objects(ms: MassSpec) -> List[Tuple]:
    """
    Convert a MassSpec into a list of simple peak objects suitable for JSON.

    Each peak is:
        {"mz": float, "inten": float, "frag_id": int or None}
    """
    # IMPORTANT: use the original masses/intens arrays so that indices stay
    # aligned with ms.int_frags.
    mzs = ms.masses
    intens = ms.intens

    # Fragment indices (for ICEBERG explanations)
    int_frags = ms.int_frags
    if int_frags is not None:
        try:
            int_frags = np.asarray(int_frags, dtype=int)
            if int_frags.shape != mzs.shape:
                # shape mismatch – ignore fragment info rather than misalign
                int_frags = None
        except Exception:
            int_frags = None

    peaks: List[Tuple] = []
    for i in range(mzs.size):
        frag_id = None
        if int_frags is not None:
            try:
                frag_id = int(int_frags[i])
            except Exception:
                frag_id = None
        peaks.append(
            (
                float(mzs[i]),
                float(intens[i]),
                frag_id,
            )
        )
    return peaks


def serialize_pred_spectra_for_frontend(lib_specs: CompositeMassSpec) -> List[Dict[str, Any]]:
    """
    Serialize all library (predicted) spectra for a single structure into a JSON-friendly list.

    Each element:
        {
          "collision_energy_ev": float,
          "peaks": [ {mz, inten, frag_id}, ... ]
        }
    """
    payload: List[Dict[str, Any]] = []
    for _, ms_lib in lib_specs.items():
        try:
            ce_val = float(getattr(ms_lib, "collision_energy", 0.0))
        except Exception:
            ce_val = 0.0

        peaks = _massspec_to_peak_objects(ms_lib)
        if not peaks:
            continue

        payload.append(
            {
                "collision_energy_ev": round_ev(ce_val),
                "peaks": peaks,
            }
        )
    return payload


def serialize_user_specs_for_frontend(
    user_specs: Any,
    stepped_mode: bool,
) -> List[Dict[str, Any]]:
    """
    Serialize the *processed* experimental spectra that were actually used for retrieval.

    Returns a list of:
        {
          "mode": "single" or "stepped",
          "collision_energy_ev": float or None,
          "nce": float or None,
          "peaks": [ {mz, inten}, ... ]
        }
    """
    def _build(ms: MassSpec, mode: str) -> Dict[str, Any]:
        # For experimental spectra we do not need frag_id
        if getattr(ms, "spec", None) is not None:
            spec = np.asarray(ms.spec, dtype=float)
            if spec.ndim != 2 or spec.shape[1] < 2:
                return {}
            mzs = spec[:, 0]
            intens = spec[:, 1]
        else:
            mzs = np.asarray(getattr(ms, "masses", []), dtype=float)
            intens = np.asarray(getattr(ms, "intens", []), dtype=float)
            if mzs.shape != intens.shape or mzs.size == 0:
                return {}

        try:
            ce_ev = round_ev(getattr(ms, "collision_energy", 0.0))
        except Exception:
            ce_ev = 0.0

        # Prefer using a cached NCE on the MassSpec, rather than
        # recomputing it from eV.
        nce_val = None
        if hasattr(ms, "nce"):
            try:
                nce_val = float(getattr(ms, "nce"))
            except Exception:
                nce_val = None

        peaks = [
            (float(mzs[i]), float(intens[i]))
            for i in range(mzs.size)
        ]

        return {
            "mode": mode,
            "collision_energy_ev": ce_ev,
            "nce": nce_val,
            "peaks": peaks,
        }

    out: List[Dict[str, Any]] = []

    if isinstance(user_specs, CompositeMassSpec):
        # Normal case: one MS per CE bucket
        for _, ms in user_specs.items():
            obj = _build(ms, mode="single")
            if obj:
                out.append(obj)
    elif isinstance(user_specs, MassSpec):
        # Stepped (or single) acquisition collapsed into a single MassSpec
        mode = "stepped" if stepped_mode else "single"
        obj = _build(user_specs, mode=mode)
        if obj:
            out.append(obj)

    return out


def retrieve_candidates(
    formula: str,
    adduct: str,
    user_specs,
    ignore_precursor: bool = True,
    apply_denoise: bool = False,
    top_k: int = 50,
    stepped_mode: bool = False,
    stepped_ces: List[str] | None = None,
    progress_cb=None,
    include_nist: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform spectrum retrieval for a given formula and user spectra.

    When `include_nist=True` and MSPRED_ATLAS_DIR_NIST is configured, also
    loads structures from the NIST'23 atlas tree and merges them with the
    public candidates before ranking.
    """
    # Precursor m/z for ignoring precursor peak if requested
    precursor_mz = common.formula_mass(formula) + common.ion2mass[adduct]
    ignore_mass_val = precursor_mz - 1 if ignore_precursor else None

    if progress_cb:
        progress_cb("preprocess", 10, "Pre-processing experimental spectrum")

    # Process the experimental spectra
    if isinstance(user_specs, CompositeMassSpec):
        all_ces = [ce for ce in user_specs.keys()]
        user_specs.process_spec_file(parentmass=precursor_mz, denoise=apply_denoise)
    elif isinstance(user_specs, MassSpec):
        all_ces = [f'{user_specs.collision_energy:.0f}']
        user_specs.process_spec_file(parentmass=precursor_mz, denoise=apply_denoise)
    else:
        raise TypeError(
            f"user_specs must be CompositeMassSpec or MassSpec, got {type(user_specs)}"
        )

    if progress_cb:
        progress_cb("preprocess", 100, "Experimental spectrum processed")

    if progress_cb:
        progress_cb("load_mgf", 2, "Loading predicted spectra (.mgf)")

    # Load from public atlas
    mgf_path = get_mgf_path_for_formula(formula, adduct)
    lib_grouped = load_library_spectra_grouped(mgf_path, progress_cb, all_ces)

    # Merge NIST atlas (authenticated users only)
    if include_nist and NIST_MGF_DIR is not None and adduct in NIST_ADDUCT_TO_DIR:
        nist_subdir = common.get_formula_subdir(formula)
        nist_mgf_path = NIST_ADDUCT_TO_DIR[adduct] / nist_subdir / f"{formula}.mgf"
        if nist_mgf_path.exists():
            try:
                nist_grouped = load_library_spectra_grouped(nist_mgf_path, None, all_ces)
                # Merge: public keys take priority on collision; NIST adds new structures
                for k, v in nist_grouped.items():
                    if k not in lib_grouped:
                        lib_grouped[k] = v
            except Exception:
                pass  # NIST merge failure is non-fatal

    if progress_cb:
        progress_cb("load_mgf", 100, f"Loaded {len(lib_grouped)} candidate structures")

    results: List[Dict[str, Any]] = []
    total = max(1, len(lib_grouped))
    if progress_cb:
        progress_cb("rank", 0, "Ranking candidates")

    for idx, (struct_key, info) in enumerate(lib_grouped.items(), start=1):
        meta_struct = info["meta"]
        lib_specs: CompositeMassSpec = info["spectra"]

        if stepped_mode:
            # Merge the *library* CEs that correspond to the stepped acquisition
            lib_ce_keys = list(lib_specs.keys())  # e.g. ["20", "30", "50"]
            if not stepped_ces:
                continue

            # Only keep energies that exist in the library for this structure
            used_ces = [ce for ce in stepped_ces if ce in lib_ce_keys]
            if not used_ces:
                continue

            try:
                final_score, ce_values = lib_specs.entr_sim(
                    user_specs,
                    merge_method="stepped",
                    stepped_ce=used_ces,
                    ignore_mass=ignore_mass_val,
                    return_ce=True,
                )
            except ValueError:
                continue

        else:
            # Original behaviour: user has separate spectra at each CE
            final_score, ce_values = lib_specs.entr_sim(
                user_specs,
                merge_method="unmerged",
                ignore_mass=ignore_mass_val,
                return_ce=True,
            )

        # Collect collision energies for display
        ce_values = sorted(ce_values)
        ce_str = ", ".join(f"{ce:.0f}" for ce in ce_values)

        # Serialize all library spectra (unmerged) for caching on the frontend
        pred_spectra_payload = serialize_pred_spectra_for_frontend(lib_specs)

        results.append(
            {
                "similarity": final_score,
                "meta": meta_struct,
                "formula": meta_struct.get("FORMULA", ""),
                "smiles": meta_struct.get("SMILES", ""),
                "inchikey": meta_struct.get("INCHIKEY", ""),
                "adduct": meta_struct.get("ADDUCT", ""),
                "collision_energy": ce_str,
                "pred_spectra": pred_spectra_payload,
            }
        )

        if progress_cb and (idx % 25 == 0 or idx == total):
            pct = int(idx / total * 100)
            progress_cb("rank", pct, f"Ranking candidates ({idx}/{total})")

    if progress_cb:
        progress_cb("postprocess", 30, "Post-processing results")

    # Sort by similarity descending and take top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    out = results[:top_k]

    if progress_cb:
        progress_cb("postprocess", 100, f"Finishing...")

    return out


def build_preloaded_spectra_from_form_lists(
    spectrum_texts: Sequence[str],
    spectrum_nces: Sequence[str],
    spectrum_evs: Sequence[str],
    spectrum_nce_conts: Sequence[str],
    spectrum_modes: Sequence[str],
    spectrum_stepped_nces: Sequence[str],
) -> list[Dict[str, Any]]:
    """
    Normalize the spectra fields coming from the HTML form into the structure
    expected by index.html (sp.text, sp.ev, sp.nce_cont, sp.mode, sp.stepped_nces).

    This is deliberately dumb: it just trusts the hidden fields that were
    already computed on the client or previous request, instead of trying to
    recompute NCE/eV relationships.
    """
    preloaded: list[Dict[str, Any]] = []
    n = len(spectrum_texts)

    for i in range(n):
        raw_text = spectrum_texts[i] or ""
        text = raw_text.strip()
        if not text:
            continue

        # NCE guess
        nce_guess = None
        if i < len(spectrum_nces) and spectrum_nces[i]:
            try:
                nce_guess = int(float(spectrum_nces[i]))
            except ValueError:
                nce_guess = None

        # Continuous NCE (if available)
        nce_cont = None
        if i < len(spectrum_nce_conts) and spectrum_nce_conts[i]:
            try:
                nce_cont = float(spectrum_nce_conts[i])
            except ValueError:
                nce_cont = None

        # EV (if available)
        ev = None
        if i < len(spectrum_evs) and spectrum_evs[i]:
            try:
                ev = float(spectrum_evs[i])
            except ValueError:
                ev = None

        mode = spectrum_modes[i] if i < len(spectrum_modes) and spectrum_modes[i] else "single"
        stepped_str = (
            spectrum_stepped_nces[i]
            if i < len(spectrum_stepped_nces) and spectrum_stepped_nces[i]
            else ""
        )

        preloaded.append(
            {
                "text": text,
                "nce_guess": nce_guess,
                "nce_cont": nce_cont,
                "ev": ev,
                "mode": mode,
                "stepped_nces": stepped_str,
            }
        )

    return preloaded


# -----------------------------
# Retrieval job state (file-backed, multi-process safe)
# -----------------------------

JOB_STORE_DIR = Path(os.environ.get("MSPRED_JOB_DIR", "/tmp/ms_pred_jobs"))
_JOB_FILE_LOCK = threading.Lock()  # intra-process guard for writes


def _ensure_job_store_dir() -> None:
    try:
        JOB_STORE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot create the directory, later operations will fail
        # and surface a clear error to the caller.
        pass


@dataclass
class JobState:
    created_at: float = field(default_factory=time.time)
    status: str = "queued"   # queued | running | done | error
    message: str = ""
    # progress in [0, 100]
    progress: dict = field(
        default_factory=lambda: {
            "load_mgf": 0,
            "preprocess": 0,
            "rank": 0,
            "postprocess": 0,
        }
    )
    result_context: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "result_context": self.result_context,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobState":
        js = cls()
        js.created_at = float(data.get("created_at", time.time()))
        js.status = data.get("status", "queued")
        js.message = data.get("message", "")
        js.progress = data.get(
            "progress",
            {
                "load_mgf": 0,
                "preprocess": 0,
                "rank": 0,
                "postprocess": 0,
            },
        )
        js.result_context = data.get("result_context")
        js.error = data.get("error")
        return js


def _job_path(job_id: str) -> Path:
    _ensure_job_store_dir()
    return JOB_STORE_DIR / f"{job_id}.json"


def _job_save(job_id: str, state: JobState) -> None:
    """
    Atomically write job state to disk so that multiple processes can
    read it safely. We write to a temporary file and then rename.
    """
    path = _job_path(job_id)
    tmp_path = path.with_suffix(".json.tmp")
    payload = state.to_dict()

    with _JOB_FILE_LOCK:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)


def _job_load(job_id: str) -> JobState | None:
    """
    Read job state from disk. Returns None if the job does not exist
    or if the file is unreadable/corrupt.
    """
    path = _job_path(job_id)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return JobState.from_dict(data)
    except FileNotFoundError:
        return None
    except Exception:
        # Corrupted/partial file – treat as missing
        return None


def _job_create(job_id: str, **kwargs) -> JobState:
    js = JobState(**kwargs)
    _job_save(job_id, js)
    return js


def _job_set(job_id: str, **kwargs) -> None:
    js = _job_load(job_id)
    if not js:
        return
    for k, v in kwargs.items():
        setattr(js, k, v)
    _job_save(job_id, js)


def _job_update_progress(
    job_id: str, key: str, value: int, message: str | None = None
) -> None:
    value = int(max(0, min(100, value)))
    js = _job_load(job_id)
    if not js:
        return
    js.progress[key] = value
    if message is not None:
        js.message = message
    _job_save(job_id, js)


def _job_get(job_id: str) -> JobState | None:
    return _job_load(job_id)


def _cleanup_old_jobs(ttl_seconds: int = 3600) -> None:
    """
    Best-effort cleanup of old job files based on modification time.
    This is safe across processes because we're only deleting files.
    """
    _ensure_job_store_dir()
    now = time.time()
    try:
        for path in JOB_STORE_DIR.glob("*.json"):
            try:
                st = path.stat()
            except FileNotFoundError:
                continue
            if now - st.st_mtime > ttl_seconds:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
    except Exception:
        # Cleanup is non-critical; ignore errors here.
        pass


# -----------------------------------------------------------------------------
# Flask application + Flask-Login
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)  # real IPs from NGINX

login_manager = LoginManager(app)
login_manager.login_view = "login"

app.jinja_env.globals["CONTACT_EMAIL"] = _CONTACT_EMAIL

_VALID_ROLES = {"admin", "authorized_user", "user"}


class _User(UserMixin):
    """Flask-Login user backed by ICEBERG_USERS_FILE YAML with role support."""

    def __init__(self, email: str, role: str = "user"):
        self.id = email
        self.role = role

    @property
    def is_admin(self) -> bool:
        return self.role == "admin" or self.id in _ADMIN_EMAILS_ENV

    @property
    def can_access_nist(self) -> bool:
        return self.role in ("admin", "authorized_user") or self.id in _ADMIN_EMAILS_ENV


def _load_users() -> Dict[str, Any]:
    """Load the users YAML, returning a dict. Returns {} if missing/unconfigured."""
    if ICEBERG_USERS_FILE is None or not ICEBERG_USERS_FILE.exists():
        return {}
    try:
        with open(ICEBERG_USERS_FILE, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _save_users(users: Dict[str, Any]) -> None:
    """Atomically write the users dict to ICEBERG_USERS_FILE."""
    if ICEBERG_USERS_FILE is None:
        raise RuntimeError("ICEBERG_USERS_FILE is not configured.")
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=ICEBERG_USERS_FILE.parent, suffix=".yaml.tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            yaml.dump(users, fh, default_flow_style=False)
        os.replace(tmp_path, ICEBERG_USERS_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_record(users: Dict[str, Any], email: str) -> Optional[Dict[str, Any]]:
    """
    Return the user record for *email*, normalising legacy flat-string entries.
    Legacy format: {email: "<hash>"} → treated as role "user".
    New format:    {email: {password: "<hash>", role: "...", created: "..."}}
    Returns None if the email is not present.
    """
    raw = users.get(email)
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"password": raw, "role": "user", "created": ""}
    return raw


@login_manager.user_loader
def _load_user(user_id: str):
    users = _load_users()
    record = _get_record(users, user_id)
    if record is not None:
        role = record.get("role", "user")
        if user_id in _ADMIN_EMAILS_ENV:
            role = "admin"
        return _User(user_id, role=role)
    return None


def _check_credentials(email: str, password: str) -> bool:
    """Return True if email/password match an entry in the users file."""
    users = _load_users()
    record = _get_record(users, email)
    if record is None:
        return False
    try:
        return check_password_hash(record["password"], password)
    except Exception:
        return False


def _update_password(email: str, new_password: str) -> None:
    """Atomically write a new password hash for *email* into the users file."""
    if ICEBERG_USERS_FILE is None:
        raise RuntimeError("ICEBERG_USERS_FILE is not configured.")
    users = _load_users()
    record = _get_record(users, email) or {}
    record["password"] = generate_password_hash(new_password)
    if "role" not in record:
        record["role"] = "user"
    users[email] = record
    _save_users(users)


def admin_required(f):
    """Decorator: login required AND the user must have admin role."""
    @functools.wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Password-strength rules (used by both /change_password and admin add-user)
# ---------------------------------------------------------------------------

_PW_RULES = [
    (lambda p: len(p) >= 8,                 "at least 8 characters"),
    (lambda p: any(c.isupper() for c in p), "an uppercase letter"),
    (lambda p: any(c.islower() for c in p), "a lowercase letter"),
    (lambda p: any(c.isdigit() for c in p), "a digit"),
    (lambda p: any(not c.isalnum() for c in p), "a special character"),
]


def _password_errors(password: str) -> List[str]:
    """Return a list of unmet rule descriptions, empty if the password is strong."""
    return [desc for check, desc in _PW_RULES if not check(password)]


# ---------------------------------------------------------------------------
# Email + random password helpers
# ---------------------------------------------------------------------------

def _generate_initial_password(length: int = 14) -> str:
    """
    Generate a random password that is guaranteed to satisfy _password_errors().
    Ensures at least one upper, lower, digit, and special character; then fills
    the remainder from the full printable ASCII alphabet and shuffles.
    """
    specials = "!@#$%^&*()-_=+[]{}|;:,.<>?"
    alphabet = string.ascii_letters + string.digits + specials
    while True:
        # Build mandatory characters first
        chars = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
            secrets.choice(specials),
        ]
        # Fill the rest
        chars += [secrets.choice(alphabet) for _ in range(length - 4)]
        secrets.SystemRandom().shuffle(chars)
        pw = "".join(chars)
        if not _password_errors(pw):  # double-check; defined below
            return pw


def _send_email(to: str, subject: str, body: str) -> Optional[str]:
    """
    Send a plain-text email via SMTP relay.
    Returns None if SMTP is not configured, empty string on success,
    or an error message string if the send failed.
    """
    if not _SMTP_HOST:
        app.logger.warning("SMTP not configured; skipping email to %s", to)
        return None
    try:
        msg = EmailMessage()
        msg["From"] = _SMTP_FROM or _SMTP_USER
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as smtp:
            if _SMTP_USE_TLS:
                smtp.starttls()
            if _SMTP_USER and _SMTP_PASSWORD:
                smtp.login(_SMTP_USER, _SMTP_PASSWORD)
            smtp.send_message(msg)
        return ""
    except Exception as exc:
        app.logger.error("Failed to send email to %s: %s", to, exc)
        return str(exc)


def _build_credential_email(email: str, password: str, kind: str = "new") -> tuple[str, str]:
    """Return (subject, body) for a credential email using the configured templates."""
    if kind == "new":
        subject, body_tpl = _EMAIL_SUBJECT_NEW, _EMAIL_BODY_NEW
    else:
        subject, body_tpl = _EMAIL_SUBJECT_RESET, _EMAIL_BODY_RESET
    body = body_tpl.format(email=email, password=password, contact=_CONTACT_EMAIL)
    return subject, body


# ---------------------------------------------------------------------------
# Analytics helpers (SQLite)
# ---------------------------------------------------------------------------

_ANALYTICS_INIT_DONE = False
_SKIP_ANALYTICS_PREFIXES = ("/static/", "/admin/api/", "/favicon")


def _analytics_conn() -> Optional[sqlite3.Connection]:
    """Open a WAL-mode SQLite connection to the analytics DB, or None if unconfigured."""
    if _ANALYTICS_DB_PATH is None:
        return None
    try:
        conn = sqlite3.connect(str(_ANALYTICS_DB_PATH), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except Exception:
        return None


def _ensure_analytics_schema() -> None:
    """Lazily create tables on first use."""
    global _ANALYTICS_INIT_DONE
    if _ANALYTICS_INIT_DONE:
        return
    conn = _analytics_conn()
    if conn is None:
        return
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT NOT NULL,
                    ip      TEXT,
                    path    TEXT,
                    method  TEXT,
                    user_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ip_geo (
                    ip           TEXT PRIMARY KEY,
                    country      TEXT,
                    lat          REAL,
                    lon          REAL,
                    resolved_ts  TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts  ON requests(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ip  ON requests(ip)")
        _ANALYTICS_INIT_DONE = True
    except Exception as exc:
        app.logger.debug("Analytics schema init error: %s", exc)
    finally:
        conn.close()


def _log_request(response):
    """after_request hook: record each request in the analytics DB."""
    try:
        path = request.path
        if any(path.startswith(p) for p in _SKIP_ANALYTICS_PREFIXES):
            return response
        _ensure_analytics_schema()
        conn = _analytics_conn()
        if conn is None:
            return response
        ts = datetime.now(timezone.utc).isoformat()
        ip = request.remote_addr or ""
        user_id = current_user.id if current_user.is_authenticated else ""
        with conn:
            conn.execute(
                "INSERT INTO requests (ts, ip, path, method, user_id) VALUES (?,?,?,?,?)",
                (ts, ip, path, request.method, user_id),
            )
        conn.close()
    except Exception:
        pass  # never break a response
    return response


def _is_private_ip(ip: str) -> bool:
    """Return True for loopback/private/link-local IPs."""
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return True


def _resolve_geo_ips(ips: List[str]) -> None:
    """Resolve a list of IPs into ip_geo using the MaxMind GeoLite2 DB."""
    if not _GEOIP_DB_PATH:
        return
    try:
        import geoip2.database  # type: ignore
        with geoip2.database.Reader(_GEOIP_DB_PATH) as reader:
            conn = _analytics_conn()
            if conn is None:
                return
            ts = datetime.now(timezone.utc).isoformat()
            for ip in ips:
                if _is_private_ip(ip):
                    continue
                try:
                    r = reader.city(ip)
                    country = r.country.name or ""
                    lat = r.location.latitude
                    lon = r.location.longitude
                    with conn:
                        conn.execute(
                            """INSERT OR REPLACE INTO ip_geo (ip, country, lat, lon, resolved_ts)
                               VALUES (?,?,?,?,?)""",
                            (ip, country, lat, lon, ts),
                        )
                except Exception:
                    pass
            conn.close()
    except ImportError:
        pass
    except Exception as exc:
        app.logger.debug("geo-IP resolution error: %s", exc)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        if _check_credentials(email, password):
            login_user(_User(email), remember=True)
            try:
                users = _load_users()
                record = _get_record(users, email)
                if record is not None:
                    record["last_login"] = datetime.now(timezone.utc).isoformat()
                    users[email] = record
                    _save_users(users)
            except Exception:
                pass
            return redirect(request.args.get("next") or url_for("index"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    errors: List[str] = []
    success = False
    if request.method == "POST":
        current_pw = request.form.get("current_password") or ""
        new_pw = request.form.get("new_password") or ""
        confirm_pw = request.form.get("confirm_password") or ""

        if not _check_credentials(current_user.id, current_pw):
            errors.append("Current password is incorrect.")
        else:
            rule_errors = _password_errors(new_pw)
            if rule_errors:
                errors.append("New password must contain " + ", ".join(rule_errors) + ".")
            elif new_pw != confirm_pw:
                errors.append("New passwords do not match.")
            elif new_pw == current_pw:
                errors.append("New password must differ from the current password.")
            else:
                try:
                    _update_password(current_user.id, new_pw)
                    success = True
                except Exception as e:
                    errors.append(f"Failed to save password: {e}")

    return render_template("change_password.html", errors=errors, success=success)


# Register the analytics logging hook now that the app object exists
app.after_request(_log_request)


# ---------------------------------------------------------------------------
# Admin dashboard routes
# ---------------------------------------------------------------------------

@app.route("/admin")
@admin_required
def admin_dashboard():
    """Admin landing page: user table + analytics."""
    users = _load_users()
    user_list = []
    for email, raw in sorted(users.items()):
        record = _get_record(users, email)
        user_list.append({
            "email": email,
            "role": record.get("role", "user") if record else "user",
            "created": record.get("created", "") if record else "",
            "last_login": record.get("last_login", "") if record else "",
        })
    # Resolve any un-geocoded IPs on dashboard load
    if _ANALYTICS_DB_PATH and _ANALYTICS_DB_PATH.exists():
        try:
            conn = _analytics_conn()
            if conn:
                cur = conn.execute(
                    "SELECT DISTINCT ip FROM requests WHERE ip != '' "
                    "AND ip NOT IN (SELECT ip FROM ip_geo)"
                )
                unresolved = [r[0] for r in cur.fetchall()]
                conn.close()
                if unresolved:
                    _resolve_geo_ips(unresolved)
        except Exception:
            pass
    return render_template("admin.html", users=user_list, valid_roles=sorted(_VALID_ROLES),
                           geoip_available=bool(_GEOIP_DB_PATH))


@app.route("/admin/users/add", methods=["POST"])
@admin_required
def admin_add_user():
    email = (request.form.get("email") or "").strip().lower()
    role = (request.form.get("role") or "user").strip()
    if role not in _VALID_ROLES:
        role = "user"

    if not email or "@" not in email:
        flash("Invalid email address.", "error")
        return redirect(url_for("admin_dashboard"))

    users = _load_users()
    if email in users:
        flash(f"User {email} already exists.", "error")
        return redirect(url_for("admin_dashboard"))

    password = _generate_initial_password()
    users[email] = {
        "password": generate_password_hash(password),
        "role": role,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)

    subject, body = _build_credential_email(email, password, kind="new")
    sent = _send_email(email, subject, body)

    if sent is None:
        flash(f"User {email} created. SMTP not configured — temp password: {password}", "warning")
    elif sent == "":
        flash(f"User {email} created and credentials emailed. Temp password: {password}", "success")
    else:
        flash(f"User {email} created but email failed ({sent}) — temp password: {password}", "error")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/reset", methods=["POST"])
@admin_required
def admin_reset_password():
    email = (request.form.get("email") or "").strip().lower()
    users = _load_users()
    if email not in users:
        flash(f"User {email} not found.", "error")
        return redirect(url_for("admin_dashboard"))

    password = _generate_initial_password()
    record = _get_record(users, email) or {}
    record["password"] = generate_password_hash(password)
    users[email] = record
    _save_users(users)

    subject, body = _build_credential_email(email, password, kind="reset")
    sent = _send_email(email, subject, body)

    if sent is None:
        flash(f"Password reset for {email}. SMTP not configured — temp password: {password}", "warning")
    elif sent == "":
        flash(f"Password reset for {email} and emailed. Temp password: {password}", "success")
    else:
        flash(f"Password reset for {email} but email failed ({sent}) — temp password: {password}", "error")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/delete", methods=["POST"])
@admin_required
def admin_delete_user():
    email = (request.form.get("email") or "").strip().lower()
    if email == current_user.id:
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("admin_dashboard"))

    users = _load_users()
    if email not in users:
        flash(f"User {email} not found.", "error")
        return redirect(url_for("admin_dashboard"))

    # Guard: don't delete the last admin
    admins = [e for e, raw in users.items()
              if (_get_record(users, e) or {}).get("role") == "admin" or e in _ADMIN_EMAILS_ENV]
    if email in admins and len(admins) <= 1:
        flash("Cannot delete the last admin account.", "error")
        return redirect(url_for("admin_dashboard"))

    del users[email]
    _save_users(users)
    flash(f"User {email} deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/set_role", methods=["POST"])
@admin_required
def admin_set_role():
    email = (request.form.get("email") or "").strip().lower()
    new_role = (request.form.get("role") or "user").strip()
    if new_role not in _VALID_ROLES:
        flash(f"Invalid role '{new_role}'.", "error")
        return redirect(url_for("admin_dashboard"))

    users = _load_users()
    if email not in users:
        flash(f"User {email} not found.", "error")
        return redirect(url_for("admin_dashboard"))

    # Guard: don't demote the last admin
    if new_role != "admin":
        admins = [e for e, raw in users.items()
                  if (_get_record(users, e) or {}).get("role") == "admin" or e in _ADMIN_EMAILS_ENV]
        if email in admins and len(admins) <= 1:
            flash("Cannot demote the last admin account.", "error")
            return redirect(url_for("admin_dashboard"))

    record = _get_record(users, email) or {}
    record["role"] = new_role
    users[email] = record
    _save_users(users)
    flash(f"Role for {email} set to '{new_role}'.", "success")
    return redirect(url_for("admin_dashboard"))


# ---------------------------------------------------------------------------
# Admin analytics API
# ---------------------------------------------------------------------------

@app.route("/admin/api/stats")
@admin_required
def admin_api_stats():
    """Return daily request counts + unique IPs, top paths, top countries (last 90 days)."""
    _ensure_analytics_schema()
    conn = _analytics_conn()
    if conn is None:
        return jsonify({"error": "Analytics DB not configured"}), 503
    try:
        days = int(request.args.get("days", 90))
        # Daily counts
        cur = conn.execute(
            """SELECT date(ts) AS day, COUNT(*) AS reqs, COUNT(DISTINCT ip) AS ips
               FROM requests
               WHERE ts >= datetime('now', ? || ' days')
               GROUP BY day ORDER BY day""",
            (f"-{days}",),
        )
        daily = [{"day": r[0], "requests": r[1], "unique_ips": r[2]} for r in cur.fetchall()]
        # Top paths
        cur = conn.execute(
            """SELECT path, COUNT(*) AS n FROM requests
               WHERE ts >= datetime('now', ? || ' days')
               GROUP BY path ORDER BY n DESC LIMIT 20""",
            (f"-{days}",),
        )
        top_paths = [{"path": r[0], "count": r[1]} for r in cur.fetchall()]
        # Top countries
        cur = conn.execute(
            """SELECT g.country, COUNT(*) AS n
               FROM requests r JOIN ip_geo g ON r.ip = g.ip
               WHERE r.ts >= datetime('now', ? || ' days') AND g.country != ''
               GROUP BY g.country ORDER BY n DESC LIMIT 20""",
            (f"-{days}",),
        )
        top_countries = [{"country": r[0], "count": r[1]} for r in cur.fetchall()]
        return jsonify({"daily": daily, "top_paths": top_paths, "top_countries": top_countries})
    finally:
        conn.close()


@app.route("/admin/api/geo")
@admin_required
def admin_api_geo():
    """Return aggregated geo points [{lat, lon, country, count}] for the map."""
    _ensure_analytics_schema()
    conn = _analytics_conn()
    if conn is None:
        return jsonify({"error": "Analytics DB not configured"}), 503
    try:
        days = int(request.args.get("days", 90))
        cur = conn.execute(
            """SELECT g.lat, g.lon, g.country, COUNT(*) AS n
               FROM requests r JOIN ip_geo g ON r.ip = g.ip
               WHERE r.ts >= datetime('now', ? || ' days')
                 AND g.lat IS NOT NULL AND g.lon IS NOT NULL
               GROUP BY g.ip
               ORDER BY n DESC""",
            (f"-{days}",),
        )
        points = [{"lat": r[0], "lon": r[1], "country": r[2], "count": r[3]}
                  for r in cur.fetchall()]
        return jsonify({"points": points})
    finally:
        conn.close()


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    formula = ""
    adduct = "[M+H]+"
    preloaded_spectra = []  # for rendering spectrum blocks from form data
    retrieval_data = None

    return render_template(
        "index.html",
        results=results,
        formula=formula,
        adduct=adduct,
        preloaded_spectra=preloaded_spectra,
        retrieval_data=retrieval_data,
    )


def _spectra_payload_from_ms_lines(raw_lines: List[str]) -> List[Dict[str, Any]]:
    """
    Shared helper: given the raw lines from a SIRIUS .ms file, return the
    spectra payload expected by the frontend (for populateSpectraFromUpload).
    """
    meta, comp_ms = common.parse_spectra(raw_lines)  # metadata, CompositeMassSpec

    # Try to get parentmass (precursor m/z) from metadata
    parentmass = None
    if meta is not None and "parentmass" in meta:
        try:
            parentmass = float(meta["parentmass"])
        except Exception:
            parentmass = None

    spectra_payload: List[Dict[str, Any]] = []

    # comp_ms.ce_to_ms is a dict; iterating .items() exactly once
    for ce_key, ms_obj in comp_ms.items():
        # Assume the labels are NCE
        if parentmass is not None:
            ms_obj.nce_to_ev(parentmass)

        # ms_obj.collision_energy has already been normalized to a float-like
        try:
            ce_val = float(ms_obj.collision_energy)
        except Exception:
            ce_val = 0.0

        spec = ms_obj.spec  # (N, 2) array of [m/z, intensity]
        if spec is None or len(spec) == 0:
            continue

        spec_text = "\n".join(f"{mz:.4f} {inten:.4f}" for mz, inten in spec)

        # Derive NCE directly from the collision-energy label (ce_key),
        # which we assume is given in NCE (%). This avoids converting eV
        # back to NCE and keeps NCE->eV as the single source of truth.
        nce_guess = None
        nce_cont = None
        try:
            nce_cont = float(ce_key)
            # Snap to {5, 10, 15, 20, ..., 100}
            nce_guess = int(round(nce_cont / 5.0) * 5)
            nce_guess = max(5, min(100, nce_guess))
        except Exception:
            nce_guess = None
            nce_cont = None

        spectra_payload.append(
            {
                "collision_energy": round_ev(ce_val),   # eV
                "spectrum_text": spec_text,   # "mz intensity" lines
                "nce_guess": nce_guess,       # bucketed NCE (may be None)
                "nce_cont": nce_cont,         # estimated NCE (may be None)
            }
        )

    return spectra_payload


@app.route("/upload_ms", methods=["POST"])
def upload_ms():
    """
    Parse an uploaded .ms (SIRIUS format) file and return spectra as JSON
    so the frontend can populate spectrum blocks.
    """
    file = request.files.get("ms_file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    raw_lines = file.read().decode("utf-8", errors="ignore").splitlines()
    spectra_payload = _spectra_payload_from_ms_lines(raw_lines)

    return jsonify({"spectra": spectra_payload})


@app.route("/load_example_ms", methods=["GET"])
def load_example_ms():
    """
    Load a bundled example SIRIUS .ms file and return spectra as JSON,
    so the frontend can auto-populate the query spectra.
    """
    try:
        if not EXAMPLE_MS_PATH.is_file():
            return jsonify({"error": f"Example .ms file not found at {EXAMPLE_MS_PATH}"}), 500

        with EXAMPLE_MS_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            raw_lines = f.read().splitlines()

        spectra_payload = _spectra_payload_from_ms_lines(raw_lines)
        if not spectra_payload:
            return jsonify({"error": "No spectra found in example .ms file."}), 500

        return jsonify({"spectra": spectra_payload})
    except Exception as e:
        return jsonify({"error": f"Failed to load example spectrum: {e}"}), 500


@app.route("/api/retrieve_start", methods=["POST"])
def api_retrieve_start():
    _cleanup_old_jobs()

    # Create job
    job_id = uuid.uuid4().hex
    _job_create(job_id, status="queued", message="Queued")

    # Capture whether the user is authenticated (for NIST access).
    # current_user is request-scoped; capture here before the thread starts.
    include_nist: bool = current_user.is_authenticated and current_user.can_access_nist

    # Capture the request payload (we will re-use your existing parsing logic)
    # Use request.form and request.files similarly to your current index() POST handler
    form_data = request.form.to_dict(flat=False)

    def worker():
        try:
            _job_set(job_id, status="running", message="Starting")

            # Reconstruct key values from form_data
            formula = (form_data.get("formula", [""])[0] or "").strip()
            adduct = (form_data.get("adduct", ["[M+H]+"])[0] or "[M+H]+").strip()
            precursor_mz = None

            spectrum_texts = [s for s in form_data.get("spectrum_text", [])]
            spectrum_nces = [s for s in form_data.get("spectrum_nce", [])]
            spectrum_modes = [s for s in form_data.get("spectrum_mode", [])]
            spectrum_stepped_nces = [s for s in form_data.get("spectrum_stepped_nces", [])]

            ignore_precursor = "ignore_precursor" in form_data
            apply_denoise = "apply_denoise" in form_data

            if not formula:
                raise ValueError("Chemical formula is required.")

            if adduct not in common.ion2mass:
                raise ValueError(
                    f"Unknown adduct: {adduct}. This adduct type is not supported by ICEBERG."
                )

            # Spectra from text areas (these include anything populated from .ms)
            has_nonempty_text = any(t.strip() for t in spectrum_texts)
            if not has_nonempty_text:
                raise ValueError(
                    "No valid user spectra were provided (neither pasted nor from .ms file)."
                )

            # Decide whether we are in stepped-CE mode or normal multi-CE mode
            # Minimal behaviour: if at least one block is "stepped", we use that
            # single stepped spectrum and ignore others for now.
            stepped_indices = [
                i
                for i, (txt, mode) in enumerate(zip(spectrum_texts, spectrum_modes))
                if txt.strip() and mode.strip() == "stepped"
            ]

            if stepped_indices:
                # For simplicity, require a single stepped spectrum block
                if len(stepped_indices) > 1:
                    raise ValueError(
                        "Stepped collision energy mode currently supports one "
                        "spectrum block per query. Please merge your stepped spectra "
                        "into a single block or use single-NCE mode for others."
                    )

                idx = stepped_indices[0]

                spec_text = spectrum_texts[idx].strip()
                if not spec_text:
                    raise ValueError("Stepped spectrum block is empty.")

                # Parse the stepped experimental spectrum as a single MassSpec
                peaks = parse_user_spectrum(spec_text)
                masses = peaks[:, 0]
                intens = peaks[:, 1]

                # Use a reasonable placeholder CE value; it is not used for matching
                # in 'stepped' mode except as a key for internal processing.
                base_nce_str = spectrum_nces[idx] if idx < len(spectrum_nces) else "30"
                try:
                    base_nce = float(base_nce_str)
                except Exception:
                    base_nce = 30.0

                if formula and adduct in common.ion2mass:
                    precursor_mz = common.formula_mass(formula) + common.ion2mass[adduct]
                    try:
                        ce_val = float(common.nce_to_ev(base_nce, precursor_mz))
                    except Exception:
                        ce_val = base_nce
                else:
                    ce_val = base_nce

                user_specs = MassSpec(
                    collision_energy=ce_val,
                    root_canonical_smiles=None,
                    adduct=adduct,
                    remark="User stepped query",
                    masses=masses,
                    intens=intens,
                )
                # Cache the base stepped NCE on this MassSpec for later display.
                try:
                    user_specs.nce = float(base_nce)
                except Exception:
                    pass

                # Parse stepped NCE list, e.g. "20,30,50"
                stepped_str = (
                    spectrum_stepped_nces[idx]
                    if idx < len(spectrum_stepped_nces)
                    else ""
                )
                stepped_str = (stepped_str or "").strip()
                if not stepped_str:
                    raise ValueError(
                        "No stepped collision energies were specified for the stepped spectrum."
                    )

                stepped_ces = [
                    s.strip()
                    for s in stepped_str.split(",")
                    if s.strip()
                ]
                if precursor_mz is not None:
                    stepped_ces = [common.nce_to_ev(_, precursor_mz) for _ in stepped_ces]
                if not stepped_ces:
                    raise ValueError(
                        "Could not parse any valid stepped collision energies."
                    )

                stepped_mode = True

            else:
                # Normal behaviour: all spectra are treated as single-NCE and
                # bundled into a CompositeMassSpec keyed by (approximate) eV.
                user_specs = prepare_user_spectra(
                    formula=formula,
                    adduct=adduct,
                    texts=spectrum_texts,
                    nces=spectrum_nces,
                )
                stepped_mode = False
                stepped_ces = None

            if user_specs is None:
                raise ValueError(
                    "No valid user spectra were provided (neither pasted nor from .ms file)."
                )

            def pcb(key, pct, msg):
                _job_update_progress(job_id, key, pct, msg)

            hits = retrieve_candidates(
                formula=formula,
                adduct=adduct,
                user_specs=user_specs,
                ignore_precursor=ignore_precursor,
                apply_denoise=apply_denoise,
                top_k=50,
                stepped_mode=stepped_mode,
                stepped_ces=stepped_ces,
                progress_cb=pcb,
                include_nist=include_nist,
            )

            # Build retrieval_data for mirror plots
            exp_payload = serialize_user_specs_for_frontend(user_specs, stepped_mode=stepped_mode)
            retrieval_data = {
                "exp_spectra": exp_payload,
                "hits": hits,
                "stepped_mode": stepped_mode,
                "warning_msg": None,
            }

            # Also store enough to re-render input blocks (preloaded_spectra)
            spectrum_evs = form_data.get("spectrum_ev", [])
            spectrum_nce_conts = form_data.get("spectrum_nce_cont", [])

            if stepped_mode:
                kept_indices = [i for i, m in enumerate(spectrum_modes) if m == 'stepped']
                if len(kept_indices) > 1:
                    kept_indices = [kept_indices[0]]
                if len(kept_indices) != len(spectrum_modes):
                    retrieval_data["warning_msg"] = "Redundant single-collision-energy spectra have been ignored."
            else:
                kept_indices = [i for i, m in enumerate(spectrum_modes) if m != 'stepped']

            preloaded_spectra = build_preloaded_spectra_from_form_lists(
                [spectrum_texts[i] for i in kept_indices],
                [spectrum_nces[i] for i in kept_indices],
                [spectrum_evs[i] for i in kept_indices] if len(spectrum_evs) > 0 else [],
                [spectrum_nce_conts[i] for i in kept_indices] if len(spectrum_nce_conts) > 0 else [],
                [spectrum_modes[i] for i in kept_indices],
                [spectrum_stepped_nces[i] for i in kept_indices],
            )

            ctx = {
                "results": hits,
                "formula": formula,
                "adduct": adduct,
                "preloaded_spectra": preloaded_spectra,
                "retrieval_data": retrieval_data,
            }

            _job_set(job_id, status="done", message="Done", result_context=ctx)

        except Exception as e:
            _job_set(job_id, status="error", message="Error", error=str(e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/retrieve_progress/<job_id>", methods=["GET"])
def api_retrieve_progress(job_id: str):
    js = _job_get(job_id)
    if not js:
        return jsonify({"error": "job_not_found"}), 404
    return jsonify({
        "status": js.status,
        "message": js.message,
        "progress": js.progress,
        "error": js.error,
    })


@app.route("/result/<job_id>", methods=["GET"])
def retrieval_result(job_id: str):
    js = _job_get(job_id)
    if not js:
        flash("Result not found (job expired). Please run retrieval again.", "warning")
        return redirect(url_for("index"))

    if js.status == "error":
        flash(f"Retrieval failed: {js.error}", "danger")
        return redirect(url_for("index"))

    if js.status != "done" or not js.result_context:
        flash("Retrieval is still running. Please wait.", "warning")
        return redirect(url_for("index"))

    ctx = js.result_context

    if 'warning_msg' in ctx["retrieval_data"] and ctx["retrieval_data"]['warning_msg']:
        flash(ctx["retrieval_data"]['warning_msg'], "warning")

    return render_template(
        "index.html",
        results=ctx["results"],
        formula=ctx["formula"],
        adduct=ctx["adduct"],
        preloaded_spectra=ctx["preloaded_spectra"],
        retrieval_data=ctx["retrieval_data"],
    )


@app.route("/download_mgf")
def download_mgf():
    """
    Download the .mgf file for a given formula.
    Authenticated internal users also get NIST'23 structures concatenated.
    Usage: /download_mgf?formula=CH9N3O9S4
    """
    formula = request.args.get("formula", "").strip()
    adduct = request.args.get("adduct", "[M+H]+").strip()
    if adduct not in ADDUCT_TO_DIR:
        adduct = "[M+H]+"

    if not formula:
        flash("Chemical formula is required to download MGF.", "danger")
        return redirect(url_for("index"))

    include_nist: bool = current_user.is_authenticated and current_user.can_access_nist
    paths = mgf_paths_for_formula(formula, adduct, include_nist=include_nist)

    if not paths:
        flash(f"MGF file not found for formula {formula}.", "danger")
        return redirect(url_for("index"))

    # Single public file — stream directly (most common case, zero-copy)
    if len(paths) == 1:
        return send_file(
            paths[0],
            mimetype="text/plain",
            as_attachment=True,
            download_name=f"{formula}.mgf",
        )

    # Multiple files (public + NIST) — stream sequentially so the browser can
    # start downloading before both files have been fully read.
    def generate_mgf():
        for p in paths:
            with open(p, "rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
            yield b"\n"

    content_length = sum(p.stat().st_size for p in paths) + len(paths)
    return Response(
        stream_with_context(generate_mgf()),
        mimetype="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{formula}.mgf"',
            "Content-Length": str(content_length),
        },
    )


@app.route("/mol_png")
def mol_png():
    """
    Render a SMILES string to a PNG using RDKit.
    Called by the frontend when hovering over a prediction row.
    Usage: /mol_png?smiles=CCO
    """
    smiles = request.args.get("smiles", "").strip()
    if not smiles:
        abort(400, description="No SMILES provided.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            abort(400, description="Invalid SMILES.")
        img = Draw.MolToImage(mol, size=(300, 300))
        bio = BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")
    except Exception:
        abort(500, description="Failed to render molecule.")


@app.route("/fragment_svg", methods=["POST"])
def fragment_svg():
    """
    Return an RDKit SVG highlighting the substructure corresponding to a fragment.

    Expected JSON body:
    {
        "smiles": "<parent SMILES>",
        "frag_id": <integer fragment id>
    }
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    smiles = (data.get("smiles") or "").strip()
    frag_id = data.get("frag_id", None)

    if not smiles:
        return jsonify({"error": "Missing 'smiles'"}), 400
    if frag_id is None:
        return jsonify({"error": "Missing 'frag_id'"}), 400

    try:
        frag_id_int = int(frag_id)
    except Exception:
        return jsonify({"error": "Invalid 'frag_id'"}), 400

    # Optional sizing + chemistry context (used by the SMILES lookup panel).
    try:
        size_px = int(data.get("size", 300))
    except Exception:
        size_px = 300
    size_px = max(120, min(800, size_px))

    adduct = (data.get("adduct") or "").strip()
    try:
        mz_obs = float(data.get("mz")) if data.get("mz") is not None else None
    except Exception:
        mz_obs = None

    try:
        # Build fragment engine and get highlight information
        engine = fragmentation.FragmentEngine(smiles, mol_str_type="smiles", mol_str_canonicalized=True)
        draw_dict = engine.get_draw_dict(frag_id_int)

        mol = draw_dict["mol"]
        hatoms = list(draw_dict.get("hatoms") or [])
        hbonds = list(draw_dict.get("hbonds") or [])

        # Draw highlighted substructure as SVG
        d2d = rdMolDraw2D.MolDraw2DSVG(size_px, size_px)
        d2d.DrawMolecule(mol, highlightAtoms=hatoms, highlightBonds=hbonds)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()

        # Compute fragment formula, with h-shift inferred from observed m/z if given.
        h_shift = 0
        if mz_obs is not None:
            try:
                base_mass = float(engine.single_mass(frag_id_int))
                h_mass = 1.00784
                h_shift = int(round((mz_obs - base_mass) / h_mass))
            except Exception:
                h_shift = 0
        try:
            base_formula = engine.formula_from_frag(frag_id_int, h_shift=h_shift)
        except Exception:
            base_formula = ""

        charge_sign = ""
        if adduct.endswith("+"):
            charge_sign = "+"
        elif adduct.endswith("-"):
            charge_sign = "-"

        return jsonify({
            "svg": svg,
            "formula": base_formula,
            "h_shift": h_shift,
            "charge_sign": charge_sign,
            "formula_display": (base_formula + charge_sign) if base_formula else "",
        })
    except Exception as e:
        # Best-effort error message for debugging
        return jsonify({"error": f"Failed to draw fragment: {e}"}), 500


@app.route("/api/query_smiles", methods=["GET"])
def api_query_smiles():
    """
    Look up the predicted spectrum for a SMILES in the precomputed atlas.

    Query params:
        smiles (required)
        adduct (default '[M+H]+')
        nce    (default 50, NCE %)
        instrument (default 'Orbitrap', currently passthrough)

    Returns JSON with canonical_smiles, formula, inchikey, adduct, instrument,
    requested_nce, requested_ce_ev, matched_ce_ev, available_ces, pred_spectrum.
    """
    smiles = (request.args.get("smiles") or "").strip()
    adduct = (request.args.get("adduct") or "[M+H]+").strip()
    instrument = (request.args.get("instrument") or "Orbitrap").strip()
    nce_raw = request.args.get("nce", "50")

    if not smiles:
        return jsonify({"error": "Missing 'smiles'"}), 400

    if adduct not in ADDUCT_TO_DIR:
        return jsonify({
            "error": f"Unsupported adduct '{adduct}'. Supported: {list(ADDUCT_TO_DIR.keys())}"
        }), 400

    try:
        nce = float(nce_raw)
    except ValueError:
        return jsonify({"error": f"Invalid nce '{nce_raw}'"}), 400

    try:
        keys = smiles_to_lookup_keys(smiles)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    formula = keys["formula"]
    canonical_smiles = keys["canonical_smiles"]
    inchikey = keys["inchikey"]
    include_nist: bool = current_user.is_authenticated and current_user.can_access_nist

    # Collect all atlas paths for this formula/adduct (public, then NIST if authed)
    mgf_paths = mgf_paths_for_formula(formula, adduct, include_nist=include_nist)
    if not mgf_paths:
        return jsonify({
            "error": f"No predicted spectra found for formula {formula}",
            "formula": formula,
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "adduct": adduct,
        }), 404

    # NCE -> eV via precursor m/z (formula + adduct mass offset)
    try:
        precursor_mz = common.formula_mass(formula) + common.ion2mass[adduct]
        ce_ev = round_ev(common.nce_to_ev(nce, precursor_mz))
    except Exception as e:
        return jsonify({"error": f"Failed to convert NCE to eV: {e}"}), 500

    # Search each atlas path; stop at first hit (public before NIST).
    # Uses the indexed fast path when a .idx sidecar is present.
    blocks = []
    for mgf_path in mgf_paths:
        try:
            blocks = extract_mgf_blocks_by_inchikey(mgf_path, inchikey)
        except Exception as e:
            return jsonify({"error": f"Failed to scan MGF: {e}"}), 500
        if blocks:
            break

    if not blocks:
        return jsonify({
            "error": f"Structure not present in atlas for {formula} / {adduct}",
            "formula": formula,
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "adduct": adduct,
        }), 404

    rep_meta, lib_specs = build_composite_from_blocks(blocks)
    if lib_specs is None:
        return jsonify({"error": "Matched structure has no usable spectra"}), 404

    all_ms = list(lib_specs.values())
    available_ces = sorted({round_ev(getattr(ms, "collision_energy", 0.0)) for ms in all_ms})
    best_ms = min(all_ms, key=lambda ms: abs(round_ev(getattr(ms, "collision_energy", 0.0)) - ce_ev))
    matched_ce = round_ev(getattr(best_ms, "collision_energy", 0.0))

    ce_warning = None
    if abs(matched_ce - ce_ev) > 5.0:
        ce_warning = (
            f"No spectrum within ±5 eV of {ce_ev:d} (NCE {nce:g}%); "
            f"showing closest available ({matched_ce:d} eV)."
        )

    single = CompositeMassSpec([best_ms])
    payload_list = serialize_pred_spectra_for_frontend(single)
    pred_spectrum = payload_list[0] if payload_list else None

    return jsonify({
        "formula": formula,
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        "adduct": adduct,
        "instrument": instrument,
        "requested_nce": nce,
        "requested_ce_ev": ce_ev,
        "precursor_mz": precursor_mz,
        "matched_ce_ev": matched_ce,
        "available_ces": available_ces,
        "ce_warning": ce_warning,
        "pred_spectrum": pred_spectrum,
        "library_meta": {
            "SMILES": rep_meta.get("SMILES", "") if rep_meta else "",
            "INCHIKEY": rep_meta.get("INCHIKEY", "") if rep_meta else "",
            "FORMULA": rep_meta.get("FORMULA", "") if rep_meta else "",
            "ADDUCT": rep_meta.get("ADDUCT", "") if rep_meta else "",
        },
    })


@app.route("/api/resolve_query", methods=["GET"])
def api_resolve_query():
    """
    Resolve a compound name / InChI / InChIKey to a list of canonical SMILES
    via the PubChem REST API.

    Query params:
        query       (required) — the query string
        query_type  (required) — one of: name | inchi | inchikey

    Returns JSON:
        {
          "candidates": [
            {
              "canonical_smiles": str,
              "formula": str,
              "inchikey": str,
              "name": str,   # IUPAC name if available
              "cid": int,
            },
            ...
          ],
          "error": str | null
        }

    On PubChem failure returns HTTP 200 with an "error" key so the frontend
    can surface a user-friendly message without treating it as a network error.
    """
    query = (request.args.get("query") or "").strip()
    query_type = (request.args.get("query_type") or "name").strip().lower()

    if not query:
        return jsonify({"candidates": [], "error": "Missing 'query'"}), 400

    SUPPORTED_NAMESPACES = {
        "name": "name",
        "inchi": "inchi",
        "inchikey": "inchikey",
        "cid": "cid",
    }
    if query_type not in SUPPORTED_NAMESPACES:
        return jsonify({
            "candidates": [],
            "error": f"Unsupported query_type '{query_type}'. Use name, inchi, inchikey, or cid.",
        }), 400

    namespace = SUPPORTED_NAMESPACES[query_type]

    # Retry wrapper mirroring candidates_from_pubchem in iceberg_elucidation.py
    def _pcp_get(q: str, ns: str) -> List[Any]:
        for attempt in range(3):
            try:
                return pcp.get_compounds(q, namespace=ns)
            except pcp.BadRequestError:
                return []
            except (pcp.ServerError, RemoteDisconnected, URLError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        return []

    try:
        compounds = _pcp_get(query, namespace)
    except Exception as e:
        return jsonify({
            "candidates": [],
            "error": f"PubChem lookup failed: {e}",
        })

    if not compounds:
        return jsonify({
            "candidates": [],
            "error": f"No PubChem match for '{query}'",
        })

    candidates: List[Dict[str, Any]] = []
    seen_inchikeys: set = set()

    for cmpd in compounds:
        smi = getattr(cmpd, "isomeric_smiles", None) or getattr(cmpd, "canonical_smiles", None)
        if not smi:
            continue

        try:
            keys = smiles_to_lookup_keys(smi)
        except Exception:
            continue

        ikey = keys.get("inchikey", "")
        if not ikey or ikey in seen_inchikeys:
            continue
        seen_inchikeys.add(ikey)

        cid = getattr(cmpd, "cid", None)
        name = getattr(cmpd, "iupac_name", None) or getattr(cmpd, "synonyms", [None])[0] or ""

        candidates.append({
            "canonical_smiles": keys["canonical_smiles"],
            "formula": keys["formula"],
            "inchikey": ikey,
            "name": str(name) if name else "",
            "cid": int(cid) if cid is not None else None,
        })

        if len(candidates) >= 25:
            break

    if not candidates:
        return jsonify({
            "candidates": [],
            "error": f"No valid structures resolved for '{query}'",
        })

    return jsonify({"candidates": candidates, "error": None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4285, debug=True)
