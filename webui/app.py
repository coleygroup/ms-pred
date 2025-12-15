#!/usr/bin/env python3
import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Dict, Any
from werkzeug.datastructures import FileStorage

from flask import Flask, render_template, request, send_file, flash, redirect, url_for, abort, jsonify
import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt

from ms_pred import common
from ms_pred.common import MassSpec, CompositeMassSpec
import ms_pred.magma.fragmentation as fragmentation

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MGF_DIR = Path("/home/runzhong/ms-pred/data/retrieval/pubchem/atlas_20250816_model_ce10to50/out_mgf")

# Collision energy matching tolerance (in eV)
EV_TOLERANCE = 1.0

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def get_formula_subdir(formula: str) -> Path:
    """
    Wrapper around common.get_formula_subdir to return a Path.
    """
    subdir = common.get_formula_subdir(formula)
    return BASE_MGF_DIR / subdir


def get_mgf_path_for_formula(formula: str) -> Path:
    """
    Get the .mgf path for a given chemical formula.
    """
    subdir = get_formula_subdir(formula)
    mgf_path = subdir / f"{formula}.mgf"
    return mgf_path


def parse_ms_upload(ms_file: "FileStorage"):
    """
    Parse an uploaded SIRIUS .ms file using common.parse_spectra.

    Returns:
        meta_ms: dict metadata (includes 'parentmass' if present)
        cms: CompositeMassSpec
    """
    if ms_file is None or ms_file.filename == "":
        return None, None

    # Read file content as text lines
    content = ms_file.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    meta_ms, cms = common.parse_spectra(lines)
    return meta_ms, cms


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

        ev_value = common.nce_to_ev(nce, precursor_mz)
        ev_value = float(ev_value)

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

    grouped: Dict[str, Dict[str, Any]] = {}

    for meta, peaks in parsed:
        if len(peaks) == 0:
            continue

        ce_str = meta.get("COLLISION_ENERGY", None)
        if ce_str is None:
            continue
        try:
            ce_val = float(ce_str)
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

    # build CompositeMassSpec objects
    for v in grouped.values():
        if len(v['spectra']) > 0:
            v['spectra'] = CompositeMassSpec(v['spectra'])

    return grouped


def match_ev(ev_user: float, ev_lib: float, tol: float = EV_TOLERANCE) -> bool:
    return abs(ev_user - ev_lib) <= tol


def _massspec_to_peak_objects(ms: MassSpec) -> List[Dict[str, Any]]:
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

    peaks: List[Dict[str, Any]] = []
    for i in range(mzs.size):
        frag_id = None
        if int_frags is not None:
            try:
                frag_id = int(int_frags[i])
            except Exception:
                frag_id = None
        peaks.append(
            {
                "mz": float(mzs[i]),
                "inten": float(intens[i]),
                "frag_id": frag_id,
            }
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
                "collision_energy_ev": ce_val,
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
            ce_ev = float(getattr(ms, "collision_energy", 0.0))
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
            {"mz": float(mzs[i]), "inten": float(intens[i])}
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


def build_mirror_entries_for_hit(
    user_specs: Any,
    lib_specs: CompositeMassSpec,
    ce_keys: List[str],
    stepped_mode: bool,
) -> List[Dict[str, Any]]:
    """
    Build mirror-plot CE entries for a (experimental, library) pair.

    Each entry has the structure:

        {
          "ceEvList": [float, ...],   # list of CE values in eV
          "ceLabel": str,             # human-readable label for the tab
          "exp": {
              "collision_energy_ev": float or None,
              "nce": float or None,
              "peaks": [ {mz, inten}, ... ],
          },
          "pred": {
              "collision_energy_ev": float or None,
              "peaks": [ {mz, inten, frag_id}, ... ],
          },
        }

    All merging is done on the backend using:
      - MassSpec.merged_spec for per-CE spectra
      - CompositeMassSpec.merge_spectra for multi-CE merges
    """
    entries: List[Dict[str, Any]] = []

    # Normalize CE keys to a unique, sorted list of strings
    ce_keys_unique = sorted({str(k) for k in ce_keys}, key=lambda x: float(x))
    if not ce_keys_unique:
        return entries

    # Helper: safe float conversion
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    if stepped_mode:
        # ---------------------------------------------------------
        # Stepped mode:
        #   - Experimental: single stepped spectrum
        #   - Library: merged across the requested stepped energies
        # ---------------------------------------------------------
        if not isinstance(user_specs, MassSpec):
            # Defensive: stepped mode should be MassSpec on the experimental side
            return entries

        # Experimental: merge within the stepped MassSpec
        try:
            exp_ms_merged = user_specs.merged_spec()
        except Exception:
            exp_ms_merged = user_specs

        exp_peaks = [
            {"mz": float(m), "inten": float(i)}
            for m, i in zip(exp_ms_merged.masses, exp_ms_merged.intens)
        ]

        # Library: merge all requested CEs for this structure
        try:
            pred_ms_merged = lib_specs.merge_spectra(energies=ce_keys_unique)
        except Exception:
            # If merging fails, fall back to a best-effort single CE (first in list)
            try:
                pred_ms_merged = lib_specs[ce_keys_unique[0]]
            except Exception:
                pred_ms_merged = None

        pred_peaks = _massspec_to_peak_objects(pred_ms_merged) if pred_ms_merged else []

        if not exp_peaks or not pred_peaks:
            return entries

        ce_ev_list = [ev for ev in (_to_float(k) for k in ce_keys_unique) if ev is not None]
        ce_ev_list_sorted = sorted(ce_ev_list)

        # Experimental NCE (one stepped value)
        nce_val = None
        if hasattr(user_specs, "nce"):
            try:
                nce_val = float(user_specs.nce)
            except Exception:
                nce_val = None

        ce_ev_str = ", ".join(f"{ev:.0f} eV" for ev in ce_ev_list_sorted)
        if nce_val is not None:
            ce_label = f"Stepped CE (eV): {ce_ev_str} | Stepped NCE ≈ {nce_val:.0f}%"
        else:
            ce_label = f"Stepped CE (eV): {ce_ev_str}"

        entries.append(
            {
                "ceEvList": ce_ev_list_sorted,
                "ceLabel": ce_label,
                "exp": {
                    "collision_energy_ev": _to_float(getattr(user_specs, "collision_energy", None)),
                    "nce": nce_val,
                    "peaks": exp_peaks,
                },
                "pred": {
                    "collision_energy_ev": None,
                    "peaks": pred_peaks,
                },
            }
        )
        return entries

    # -------------------------------------------------------------
    # Non-stepped mode:
    #   user_specs is CompositeMassSpec with one MS per CE bucket.
    #   We build:
    #     - one entry per CE
    #     - one "merged" entry across all shared CEs (if >1)
    # -------------------------------------------------------------
    if not isinstance(user_specs, CompositeMassSpec):
        return entries

    # Per-CE entries
    per_entries: List[Dict[str, Any]] = []
    ce_ev_by_key: Dict[str, float] = {}
    for k in ce_keys_unique:
        ev = _to_float(k)
        if ev is None:
            continue
        ce_ev_by_key[k] = ev

    for ce_key, ev in sorted(ce_ev_by_key.items(), key=lambda kv: kv[1]):
        try:
            exp_ms_raw = user_specs[ce_key]
            pred_ms_raw = lib_specs[ce_key]
        except Exception:
            continue

        # Merge within each MassSpec using MassSpec.merged_spec
        try:
            exp_ms = exp_ms_raw.merged_spec()
        except Exception:
            exp_ms = exp_ms_raw

        try:
            pred_ms = pred_ms_raw.merged_spec()
        except Exception:
            pred_ms = pred_ms_raw

        exp_peaks = [
            {"mz": float(m), "inten": float(i)}
            for m, i in zip(exp_ms.masses, exp_ms.intens)
        ]
        pred_peaks = _massspec_to_peak_objects(pred_ms)

        if not exp_peaks or not pred_peaks:
            continue

        # Experimental NCE for this CE, if available
        nce_val = None
        if hasattr(exp_ms_raw, "nce"):
            try:
                nce_val = float(exp_ms_raw.nce)
            except Exception:
                nce_val = None

        label = f"CE {ev:.0f} eV"
        if nce_val is not None:
            label += f" | NCE ≈ {nce_val:.0f}%"

        per_entries.append(
            {
                "ceEvList": [ev],
                "ceLabel": label,
                "exp": {
                    "collision_energy_ev": ev,
                    "nce": nce_val,
                    "peaks": exp_peaks,
                },
                "pred": {
                    "collision_energy_ev": ev,
                    "peaks": pred_peaks,
                },
            }
        )

    entries.extend(per_entries)

    # Multi-CE merged entry (if >1 CE)
    if len(ce_keys_unique) > 1 and per_entries:
        try:
            exp_merged_ms = user_specs.merge_spectra(energies=ce_keys_unique)
        except Exception:
            exp_merged_ms = None
        try:
            pred_merged_ms = lib_specs.merge_spectra(energies=ce_keys_unique)
        except Exception:
            pred_merged_ms = None

        exp_merged_peaks = (
            [
                {"mz": float(m), "inten": float(i)}
                for m, i in zip(exp_merged_ms.masses, exp_merged_ms.intens)
            ]
            if exp_merged_ms is not None
            else []
        )
        pred_merged_peaks = (
            _massspec_to_peak_objects(pred_merged_ms) if pred_merged_ms is not None else []
        )

        if exp_merged_peaks and pred_merged_peaks:
            # Average NCE across all contributing CEs (if available)
            nces: List[float] = []
            for ce_key in ce_keys_unique:
                try:
                    nce_val = getattr(user_specs[ce_key], "nce", None)
                except Exception:
                    nce_val = None
                if nce_val is not None:
                    try:
                        nces.append(float(nce_val))
                    except Exception:
                        pass
            nce_str = "/".join(f"{nce:.0f}%" for nce in nces) if nces else None

            ce_ev_list = [ev for ev in (_to_float(k) for k in ce_keys_unique) if ev is not None]
            ce_ev_list_sorted = sorted(ce_ev_list)
            ce_ev_str = "/".join(f"{ev:.0f}" for ev in ce_ev_list_sorted)

            if nce_str is not None:
                label = f"Merged CE {ce_ev_str} eV | NCE ≈ {nce_str}"
            else:
                label = f"Merged CE {ce_ev_str} eV"

            entries.append(
                {
                    "ceEvList": ce_ev_list_sorted,
                    "ceLabel": label,
                    "exp": {
                        "collision_energy_ev": None,
                        "nce": None,
                        "peaks": exp_merged_peaks,
                    },
                    "pred": {
                        "collision_energy_ev": None,
                        "peaks": pred_merged_peaks,
                    },
                }
            )

    return entries


def retrieve_candidates(
    formula: str,
    adduct: str,
    user_specs,
    ignore_precursor: bool = True,
    apply_denoise: bool = False,
    top_k: int = 50,
    stepped_mode: bool = False,
    stepped_ces: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Perform spectrum retrieval for a given formula and user spectra.
    ...
    """
    mgf_path = get_mgf_path_for_formula(formula)
    lib_grouped = load_library_spectra_grouped(mgf_path)

    # Precursor m/z for ignoring precursor peak if requested
    precursor_mz = common.formula_mass(formula) + common.ion2mass[adduct]
    ignore_mass_val = precursor_mz - 1 if ignore_precursor else None

    # Process the experimental spectra
    if isinstance(user_specs, CompositeMassSpec) or isinstance(user_specs, MassSpec):
        user_specs.process_spec_file(parentmass=precursor_mz, denoise=apply_denoise)
    else:
        raise TypeError(
            f"user_specs must be CompositeMassSpec or MassSpec, got {type(user_specs)}"
        )

    results: List[Dict[str, Any]] = []

    for struct_key, info in lib_grouped.items():
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

    # Sort by similarity descending and take top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# -----------------------------------------------------------------------------
# Flask application
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "ms-pred-secret-key"  # replace with something secure in production


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    formula = ""
    adduct = "[M+H]+"
    preloaded_spectra = []  # for rendering spectrum blocks from form data
    retrieval_data = None

    if request.method == "POST":
        formula = request.form.get("formula", "").strip()
        adduct = request.form.get("adduct", "[M+H]+").strip()

        spectrum_texts = request.form.getlist("spectrum_text")
        spectrum_nces = request.form.getlist("spectrum_nce")
        spectrum_evs = request.form.getlist("spectrum_ev")
        spectrum_nce_conts = request.form.getlist("spectrum_nce_cont")
        spectrum_modes = request.form.getlist("spectrum_mode")              # "single" or "stepped"
        spectrum_stepped_nces = request.form.getlist("spectrum_stepped_nces")  # e.g. "20,30,50" or ""

        ignore_precursor = request.form.get("ignore_precursor") == "on"
        apply_denoise = request.form.get("apply_denoise") == "on"

        ms_file = request.files.get("ms_file")

        # -------------------------------------------------------------
        # (A) Build preloaded_spectra from whatever the user submitted
        #     so the blocks stay on the page after "Run retrieval".
        # -------------------------------------------------------------
        precursor_mz = None
        if formula and adduct in common.ion2mass:
            try:
                precursor_mz = common.formula_mass(formula) + common.ion2mass[adduct]
            except Exception:
                precursor_mz = None

        preloaded_spectra = []

        num_blocks = max(
            len(spectrum_texts),
            len(spectrum_nces),
            len(spectrum_evs),
            len(spectrum_nce_conts),
            len(spectrum_modes),
            len(spectrum_stepped_nces),
        )

        for i in range(num_blocks):
            spec_text = (spectrum_texts[i] if i < len(spectrum_texts) else "").strip()
            if not spec_text:
                continue

            nce_str = (spectrum_nces[i] if i < len(spectrum_nces) else "").strip()
            ev_str = (spectrum_evs[i] if i < len(spectrum_evs) else "").strip()
            nce_cont_str = (spectrum_nce_conts[i] if i < len(spectrum_nce_conts) else "").strip()

            # dropdown bucket (for UI)
            try:
                nce_guess = int(round(float(nce_str)))
            except ValueError:
                nce_guess = 30

            # exact NCE (from hidden) if available
            nce_cont = None
            if nce_cont_str:
                try:
                    nce_cont = float(nce_cont_str)
                except ValueError:
                    nce_cont = None

            # exact eV (from hidden) if available
            ev_val = None
            if ev_str:
                try:
                    ev_val = float(ev_str)
                except ValueError:
                    ev_val = None

            # fallback: if ev_val missing but we have nce_cont, recompute
            if ev_val is None and precursor_mz is not None:
                try:
                    base_nce = nce_cont if nce_cont is not None else float(nce_str)
                    ev_val = float(common.nce_to_ev(base_nce, precursor_mz))
                except Exception:
                    ev_val = None

            # We intentionally do NOT recompute NCE from eV here.
            # If a continuous NCE value is available (e.g. from .ms upload),
            # it is passed through via the hidden field. Otherwise, NCE
            # remains unknown and we only display the bucketed dropdown.

            # acquisition mode: "single" or "stepped"
            mode = (spectrum_modes[i] if i < len(spectrum_modes) else "single").strip() or "single"

            stepped_nces_str = (
                spectrum_stepped_nces[i] if i < len(spectrum_stepped_nces) else ""
            )
            stepped_nces_str = stepped_nces_str.strip()

            preloaded_spectra.append(
                {
                    "nce_guess": nce_guess,
                    "ev": ev_val,
                    "text": spec_text,
                    "nce_cont": nce_cont,
                    "mode": mode,
                    "stepped_nces": stepped_nces_str,
                }
            )

        # -------------------------------------------------------------
        # (B) Retrieval logic (unchanged, except it now reuses the same
        #     spectrum_texts / spectrum_nces that we just saved above).
        # -------------------------------------------------------------
        try:
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

            stepped_mode = False
            stepped_ces = None
            user_specs = None

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

            results = retrieve_candidates(
                formula=formula,
                adduct=adduct,
                user_specs=user_specs,
                ignore_precursor=ignore_precursor,
                apply_denoise=apply_denoise,
                stepped_mode=stepped_mode,
                stepped_ces=stepped_ces,
            )

            # Build a JSON-friendly snapshot of everything needed for mirror plots.
            # This is per-request state that will be sent to the frontend, so the
            # backend does NOT need to reload .mgf or re-process the experimental
            # spectra.
            if results:
                exp_spectra_payload = serialize_user_specs_for_frontend(
                    user_specs=user_specs,
                    stepped_mode=stepped_mode,
                )

                retrieval_data = {
                    "formula": formula,
                    "adduct": adduct,
                    "stepped_mode": bool(stepped_mode),
                    "exp_spectra": exp_spectra_payload,
                    "hits": [
                        {
                            "formula": r.get("formula", ""),
                            "adduct": r.get("adduct", ""),
                            "smiles": r.get("smiles", ""),
                            "inchikey": r.get("inchikey", ""),
                            "pred_spectra": r.get("pred_spectra", []),
                        }
                        for r in results
                    ],
                }

            if not results:
                flash("No candidates found with matching collision energies.", "warning")

        except FileNotFoundError as e:
            flash(str(e), "danger")
        except Exception as e:
            flash(f"Error during retrieval: {e}", "danger")

    return render_template(
        "index.html",
        results=results,
        formula=formula,
        adduct=adduct,
        preloaded_spectra=preloaded_spectra,
        retrieval_data=retrieval_data,
    )


@app.route("/upload_ms", methods=["POST"])
def upload_ms():
    """
    Parse an uploaded .ms (SIRIUS format) file and return spectra as JSON
    so the frontend can populate spectrum blocks.
    """
    file = request.files.get("ms_file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    # Read file lines
    raw = file.read().decode("utf-8", errors="ignore").splitlines()
    meta, comp_ms = common.parse_spectra(raw)  # returns metadata, CompositeMassSpec

    # Try to get parentmass (precursor m/z) from metadata
    parentmass = None
    if meta is not None and "parentmass" in meta:
        try:
            parentmass = float(meta["parentmass"])
        except Exception:
            parentmass = None

    spectra_payload = []

    # comp_ms.ce_to_ms is a dict; iterating .items() exactly once
    for ce_key, ms_obj in comp_ms.items():
        # assume the labels are nce
        if parentmass is not None:
            ms_obj.nce_to_ev(parentmass)

        # ms_obj.collision_energy has already been normalized to a float-like
        ce_val = float(ms_obj.collision_energy)

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
            # Snap to {10, 20, 30, 40, 50}
            nce_guess = int(round(nce_cont / 10.0) * 10)
            nce_guess = max(10, min(50, nce_guess))
        except Exception:
            nce_guess = None
            nce_cont = None

        spectra_payload.append(
            {
                "collision_energy": ce_val,   # eV
                "spectrum_text": spec_text,   # "mz intensity" lines
                "nce_guess": nce_guess,       # bucketed NCE (may be None)
                "nce_cont": nce_cont,         # estimated NCE (may be None)
            }
        )

    return jsonify({"spectra": spectra_payload})



@app.route("/download_mgf")
def download_mgf():
    """
    Download the .mgf file for a given formula.
    Usage: /download_mgf?formula=CH9N3O9S4
    """
    formula = request.args.get("formula", "").strip()
    if not formula:
        flash("Chemical formula is required to download MGF.", "danger")
        return redirect(url_for("index"))

    mgf_path = get_mgf_path_for_formula(formula)
    if not mgf_path.exists():
        flash(f"MGF file not found for formula {formula}.", "danger")
        return redirect(url_for("index"))

    return send_file(
        mgf_path,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"{formula}.mgf",
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

    try:
        # Build fragment engine and get highlight information
        engine = fragmentation.FragmentEngine(smiles, mol_str_type="smiles")
        draw_dict = engine.get_draw_dict(frag_id_int)

        mol = draw_dict["mol"]
        hatoms = list(draw_dict.get("hatoms") or [])
        hbonds = list(draw_dict.get("hbonds") or [])

        # Draw highlighted substructure as SVG
        d2d = rdMolDraw2D.MolDraw2DSVG(300, 300)
        d2d.DrawMolecule(mol, highlightAtoms=hatoms, highlightBonds=hbonds)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()

        return jsonify({"svg": svg})
    except Exception as e:
        # Best-effort error message for debugging
        return jsonify({"error": f"Failed to draw fragment: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
