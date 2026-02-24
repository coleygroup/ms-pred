#!/usr/bin/env python3
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple
import threading
import time
import uuid
import os
import json
from dataclasses import dataclass, field

from flask import Flask, render_template, request, send_file, flash, redirect, url_for, abort, jsonify
import numpy as np

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
BASE_MGF_DIR = Path(os.environ.get("MSPRED_ATLAS_DIR", "/home/coley-group/atlas/"))
ADDUCT_TO_DIR = {
    '[M+H]+': BASE_MGF_DIR / 'h_plus_out_mgf',
    '[M-H]-': BASE_MGF_DIR / 'h_minus_out_mgf',
}

# Example spectrum
EXAMPLE_MS_PATH = (
    Path(__file__).resolve().parent.parent / "data/exp_specs/clinical/lpc19-0_standard.ms"
)

# Collision energy matching tolerance (in eV)
EV_TOLERANCE = 1.0

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def get_formula_subdir(formula: str, adduct='[M+H]+') -> Path:
    """
    Wrapper around common.get_formula_subdir to return a Path.
    """
    subdir = common.get_formula_subdir(formula)
    return ADDUCT_TO_DIR[adduct] / subdir


def get_mgf_path_for_formula(formula: str, adduct='[M+H]+') -> Path:
    """
    Get the .mgf path for a given chemical formula.
    """
    subdir = get_formula_subdir(formula, adduct)
    mgf_path = subdir / f"{formula}.mgf"
    return mgf_path


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
            ce_val = float(ce_str)
            if interested_ces is not None:
                if f'{ce_val:.0f}' not in interested_ces:
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
) -> List[Dict[str, Any]]:
    """
    Perform spectrum retrieval for a given formula and user spectra.
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

    mgf_path = get_mgf_path_for_formula(formula, adduct)
    lib_grouped = load_library_spectra_grouped(mgf_path, progress_cb, all_ces)

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
# Flask application
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]


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
                "collision_energy": ce_val,   # eV
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
