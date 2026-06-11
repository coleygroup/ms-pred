""" utils """
import sys
import copy
import logging
from pathlib import Path, PosixPath
import json
from itertools import groupby, islice
from typing import Tuple, List, Dict, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import hashlib
import torch
import math
import re
from collections import defaultdict

import ms_pred.common.chem_utils as chem_utils
from ms_pred.common.denoising_utils import electronic_denoising
from ms_pred import nn_utils

try:
    from pytorch_lightning.loggers import LightningLoggerBase as Logger
    from pytorch_lightning.loggers.base import rank_zero_experiment
except ImportError: # pytorch_lightning >= 1.9
    from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

NIST_COLLISION_ENERGY_MEAN = 40.260853377886264
NIST_COLLISION_ENERGY_STD = 31.604227557486197

_MANIFEST_GROUP = "__predspec_manifest__"
_MANIFEST_VERSION = 1
_MANIFEST_LAYOUT_VERSION = 2  # resizable datasets for incremental append

def get_data_dir(dataset_name: str) -> Path:
    return Path("data/spec_datasets") / dataset_name


class MassSpec:
    """
    Data structure for a predicted MS/MS spectrum
    """
    def __init__(self, collision_energy: Union[str, float], root_canonical_smiles=None, adduct=None, remark=None,
                 probs=None, brokens=None, masses=None, masses_no_adduct=None,
                 frag_form_vecs=None, frags=None, intens=None, int_frags=None,
                 binned_spec=None, binned_spec_sparse=None, num_bins=None, upper_limit=None, **kwargs):
        self._engine = None
        self._binned_inds = None
        self._binned_vals = None
        self._binned_spec_dense_cache = None
        self._mass_upper_limit = upper_limit
        self._num_bins = num_bins
        self._binned_pool_fn = None

        if isinstance(collision_energy, str) and 'collision' in collision_energy:
            self.collision_energy = chem_utils.get_collision_energy(collision_energy)
        else:
            self.collision_energy = collision_energy
        self.collision_energy = float(f'{float(self.collision_energy):.0f}')
        self.root_canonical_smiles = root_canonical_smiles
        self.adduct = adduct
        self.remark = remark
        def safe_assign(x, dtype):
            if isinstance(x, np.ndarray):
                x_np = x
            elif isinstance(x, torch.Tensor):
                x_np = x.cpu().numpy()
            elif isinstance(x, list) and len(x) > 0:
                x_np = np.array(x)
            else:
                return None
            if not np.issubdtype(x_np.dtype, dtype):
                if dtype == np.integer:
                    x_np = x_np.astype(np.integer)
                else:
                    raise TypeError(f'Input data does not have the correct data type, expected {dtype}, got {x_np.dtype}')
            return x_np
        self.probs = safe_assign(probs, np.floating)
        self.brokens = safe_assign(brokens, np.integer)
        self.masses = safe_assign(masses, np.floating)
        self.masses_no_adduct = safe_assign(masses_no_adduct, np.floating)
        self.frag_form_vecs = safe_assign(frag_form_vecs, np.integer)
        if frags is None and int_frags is not None:
            from ms_pred.magma.fragmentation import FragmentEngine
            self._engine = FragmentEngine(self.root_canonical_smiles, mol_str_canonicalized=True)
            bit_lists = [
                ((x >> np.arange(self._engine.natoms)) & 1).astype(bool)
                for x in int_frags
            ]
            frags = np.vstack(bit_lists)
        self.frags = safe_assign(frags, bool)
        self.intens = safe_assign(intens, np.floating)
        if binned_spec is not None and binned_spec_sparse is not None:
            raise ValueError("Specify only one of binned_spec or binned_spec_sparse")
        if binned_spec_sparse is not None:
            self._set_binned_spec_sparse(binned_spec_sparse, num_bins=num_bins, upper_limit=upper_limit)
        elif binned_spec is not None:
            self._set_binned_spec_dense(safe_assign(binned_spec, np.floating), upper_limit=upper_limit)
        for key in ("atoms_pulled_ptr", "atoms_pulled_data"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = safe_assign(kwargs[key], np.integer)
        self.meta = kwargs

    def _set_binned_spec_dense(self, binned_spec, upper_limit=None, pool_fn=None):
        if binned_spec is None:
            self._binned_inds = None
            self._binned_vals = None
            self._binned_spec_dense_cache = None
            self._binned_pool_fn = None
            return
        binned_spec = np.asarray(binned_spec, dtype=np.float32)
        if binned_spec.ndim != 1:
            raise ValueError(f"binned_spec must be 1D, got shape {binned_spec.shape}")
        inds = np.flatnonzero(binned_spec > 0).astype(np.uint32)
        vals = binned_spec[inds].astype(np.float32, copy=False)
        self._set_binned_spec_sparse((inds, vals), num_bins=len(binned_spec), upper_limit=upper_limit, pool_fn=pool_fn)

    def _set_binned_spec_sparse(self, binned_spec_sparse, num_bins=None, upper_limit=None, pool_fn=None):
        if isinstance(binned_spec_sparse, dict):
            inds = binned_spec_sparse.get("indices")
            vals = binned_spec_sparse.get("values")
        elif isinstance(binned_spec_sparse, tuple) and len(binned_spec_sparse) == 2:
            inds, vals = binned_spec_sparse
        else:
            arr = np.asarray(binned_spec_sparse)
            if arr.size == 0:
                inds = np.zeros(0, dtype=np.uint32)
                vals = np.zeros(0, dtype=np.float32)
            elif arr.ndim == 2 and arr.shape[1] == 2:
                inds, vals = arr[:, 0], arr[:, 1]
            else:
                raise ValueError("binned_spec_sparse must be (indices, values), a dict, or an [n, 2] array")

        inds = np.asarray(inds, dtype=np.uint32)
        vals = np.asarray(vals, dtype=np.float32)
        if inds.ndim != 1 or vals.ndim != 1 or len(inds) != len(vals):
            raise ValueError("Sparse binned indices and values must be 1D arrays with the same length")

        keep = vals > 0
        inds = inds[keep]
        vals = vals[keep]
        order = np.argsort(inds, kind="stable")
        inds = inds[order]
        vals = vals[order]
        if len(inds) > 1:
            unique_inds, inverse = np.unique(inds, return_inverse=True)
            if len(unique_inds) != len(inds):
                summed_vals = np.zeros(len(unique_inds), dtype=np.float32)
                np.add.at(summed_vals, inverse, vals)
                inds = unique_inds.astype(np.uint32, copy=False)
                vals = summed_vals

        if num_bins is None:
            num_bins = self._num_bins
        if num_bins is None:
            num_bins = int(inds.max() + 1) if len(inds) > 0 else 0
        if len(inds) > 0 and int(inds.max()) >= int(num_bins):
            raise ValueError("Sparse binned index exceeds num_bins")

        self._binned_inds = inds
        self._binned_vals = vals
        self._num_bins = int(num_bins)
        if upper_limit is not None:
            self._mass_upper_limit = upper_limit
        self._binned_pool_fn = pool_fn
        self._binned_spec_dense_cache = None

    @staticmethod
    def _bin_spec_sparse(spec, num_bins: int = 15000, upper_limit: int = 1500, pool_fn: str = "add"):
        if isinstance(spec, MassSpec) or hasattr(spec, "spec"):
            spec = spec.spec
        if spec is None or len(spec) == 0:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)

        spec = np.asarray(spec)
        mz = spec[:, 0]
        inten = spec[:, 1]
        scale = (num_bins - 1) / upper_limit
        bin_idx = np.floor(mz * scale).astype(np.int64) + 1
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        bin_idx = bin_idx[valid]
        inten = inten[valid].astype(np.float32, copy=False)
        if len(bin_idx) == 0:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)

        unique_inds, inverse = np.unique(bin_idx, return_inverse=True)
        if pool_fn == "add":
            vals = np.zeros(len(unique_inds), dtype=np.float32)
            np.add.at(vals, inverse, inten)
        elif pool_fn == "max":
            order = np.argsort(inverse, kind="stable")
            sorted_inverse = inverse[order]
            sorted_inten = inten[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
            vals = np.maximum.reduceat(sorted_inten, starts).astype(np.float32, copy=False)
        else:
            raise NotImplementedError()
        return unique_inds.astype(np.uint32, copy=False), vals.astype(np.float32, copy=False)

    def has_matching_binned_spec(self, mass_upper_limit=1500, num_bins=15000, pool_fn=None):
        return (
            self.has_binned_spec
            and self._num_bins == int(num_bins)
            and self._mass_upper_limit == mass_upper_limit
            and (pool_fn is None or self._binned_pool_fn is None or self._binned_pool_fn == pool_fn)
        )

    def ensure_binned_spectrum(self, mass_upper_limit=1500, num_bins=15000, pool_fn='add', force=False):
        if not force and self.has_matching_binned_spec(mass_upper_limit, num_bins, pool_fn):
            return
        if self.has_masses and self.has_intens:
            self._set_binned_spec_sparse(
                self._bin_spec_sparse(self, num_bins=num_bins, upper_limit=mass_upper_limit, pool_fn=pool_fn),
                num_bins=num_bins,
                upper_limit=mass_upper_limit,
                pool_fn=pool_fn,
            )
        else:
            raise ValueError('Spectrum object should have both masses and intens')

    @classmethod
    def from_instance(cls, other_instance, **kwargs):
        def clone_array_like(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, torch.Tensor):
                return value.clone()
            return value

        new_kwargs = {
            "collision_energy": other_instance.collision_energy,
            "root_canonical_smiles": other_instance.root_canonical_smiles,
            "adduct": other_instance.adduct,
            "remark": other_instance.remark,
            "probs": clone_array_like(other_instance.probs),
            "brokens": clone_array_like(other_instance.brokens),
            "masses": clone_array_like(other_instance.masses),
            "masses_no_adduct": clone_array_like(other_instance.masses_no_adduct),
            "frag_form_vecs": clone_array_like(other_instance.frag_form_vecs),
            "frags": clone_array_like(other_instance.frags),
            "intens": clone_array_like(other_instance.intens),
            "binned_spec_sparse": (
                clone_array_like(other_instance._binned_inds),
                clone_array_like(other_instance._binned_vals),
            ) if other_instance.has_binned_spec else None,
            "num_bins": other_instance._num_bins,
            "upper_limit": other_instance._mass_upper_limit,
            **copy.deepcopy(other_instance.meta),
        }
        new_kwargs.update(kwargs)
        return cls(**new_kwargs)

    def add_hydrogen_shift(self):
        """add hydrogen shift to masses and fragments"""
        h_pos = chem_utils.element_to_ind["H"]
        if not self.has_brokens:
            raise ValueError('self.brokens is required to add hydrogen shift')

        if self.has_masses:
            n = len(self.masses)
        elif self.has_masses_no_adduct:
            n = len(self.masses_no_adduct)
        elif self.has_frag_form_vecs:
            n = len(self.frag_form_vecs)
        else:
            raise ValueError('Cannot infer data size')
        assert len(self.brokens) == n
        assert self.intens is None

        new_masses, new_masses_no_adduct, new_frag_form_vecs, new_frags, new_probs = [], [], [], [], []
        for i in range(n):
            nbrokens = int(self.brokens[i])
            for hshift in range(-nbrokens, nbrokens + 1):
                if self.has_frag_form_vecs: # make sure not to create negative H in form
                    if self.frag_form_vecs[i][h_pos] < -hshift:
                        continue

                if self.has_masses:
                    new_masses.append(self.masses[i] + chem_utils.ELEMENT_TO_MASS['H'] * hshift)
                if self.has_masses_no_adduct:
                    new_masses_no_adduct.append(self.masses_no_adduct[i] + chem_utils.ELEMENT_TO_MASS['H'] * hshift)
                if self.has_frag_form_vecs:
                    vec = self.frag_form_vecs[i].astype(np.int32)
                    vec[h_pos] += hshift
                    new_frag_form_vecs.append(vec.astype(np.uint8))
                if self.has_frags:
                    new_frags.append(self.frags[i])
                if self.has_probs:
                    new_probs.append(self.probs[i])
        if self.has_masses:
            self.masses = new_masses
        if self.has_masses_no_adduct:
            self.masses_no_adduct = new_masses_no_adduct
        if self.has_frag_form_vecs:
            self.frag_form_vecs = new_frag_form_vecs
        if self.has_frags:
            self.frags = new_frags
        if self.has_probs:
            self.probs = new_probs

    @property
    def root_form(self):
        return chem_utils.form_from_smi(self.root_canonical_smiles) \
            if self.root_canonical_smiles is not None else None

    @property
    def frag_form(self):
        return [chem_utils.vec_to_formula(vec) for vec in self.frag_form_vecs] \
            if self.frag_form_vecs is not None else None

    @property
    def info(self):
        info = {
            "collision_energy": self.collision_energy,
            "root_canonical_smiles": self.root_canonical_smiles,
            "adduct": self.adduct,
            "remark": self.remark,
        }
        info.update(self.meta)
        info = {k: v for k, v in info.items() if v is not None}
        return info

    @property
    def parent_mass(self):
        if self.adduct is not None and self.root_canonical_smiles is not None:
            return chem_utils.ion2mass[self.adduct] + chem_utils.mass_from_smi(self.root_canonical_smiles)
        else:
            return None

    @property
    def inchi(self):
        return chem_utils.inchi_from_smiles(self.root_canonical_smiles)

    @property
    def inchikey(self):
        return chem_utils.inchikey_from_smiles(self.root_canonical_smiles)

    @property
    def formula(self):
        return chem_utils.form_from_smi(self.root_canonical_smiles)

    @property
    def int_frags(self):
        if self.has_frags:
            all_frags = []
            for bin_frag in self.frags:
                frag = 0
                for i, b in enumerate(bin_frag):
                    if b:
                        frag += 2 ** i
                all_frags.append(frag)
            return all_frags
        else:
            return None

    @property
    def has_probs(self):
        return self.probs is not None

    @property
    def has_masses(self):
        return self.masses is not None

    @property
    def has_masses_no_adduct(self):
        return self.masses_no_adduct is not None

    @property
    def has_intens(self):
        return self.intens is not None

    @property
    def has_brokens(self):
        return self.brokens is not None

    @property
    def has_frag_form_vecs(self):
        return self.frag_form_vecs is not None

    @property
    def has_frags(self):
        return self.frags is not None

    @property
    def max_add_hs(self):
        return self.brokens

    @property
    def max_remove_hs(self):
        if self.has_frag_form_vecs and self.has_brokens:
            nhs = self.frag_form_vecs[:, chem_utils.element_to_ind['H']]
            return np.minimum(self.brokens, nhs)
        else:
            return None

    def merged_spec(self, merge_method='sum') -> "MassSpec":
        spec_ar = self.spec
        if spec_ar is not None:
            merged_tup = self._merge_spec_to_tup(merge_method=merge_method)
            return self._merge_tup_to_spec(merged_tup)
        else:
            return None

    @property
    def spec(self):
        if self.has_masses and self.has_intens:
            return np.stack((self.masses, self.intens), axis=1)
        else:
            return None

    def get_atoms_pulled(self):
        ptr = self.meta.get("atoms_pulled_ptr")
        data = self.meta.get("atoms_pulled_data")
        if ptr is None or data is None:
            return None
        ptr = np.asarray(ptr, dtype=np.int64)
        data = np.asarray(data, dtype=np.int64)
        if len(ptr) == 0:
            return []
        return [data[ptr[i]:ptr[i + 1]].tolist() for i in range(len(ptr) - 1)]

    @property
    def binned_spec(self):
        if not self.has_binned_spec:
            self.ensure_binned_spectrum()  # use default parameters (0.1 bin)
        if self._binned_spec_dense_cache is not None:
            return self._binned_spec_dense_cache
        out = np.zeros(int(self._num_bins), dtype=np.float32)
        if len(self._binned_inds) > 0:
            out[self._binned_inds.astype(np.int64)] = self._binned_vals
        return out

    @property
    def binned_spec_sparse(self):
        if not self.has_binned_spec:
            self.ensure_binned_spectrum()  # use default parameters (0.1 bin)
        return self._binned_inds, self._binned_vals

    def binned_spec_dense(self, cache=False):
        dense = self.binned_spec
        if cache and self._binned_spec_dense_cache is None:
            self._binned_spec_dense_cache = dense
        return self._binned_spec_dense_cache if cache else dense

    def cache_binned_spec_dense(self):
        return self.binned_spec_dense(cache=True)

    def clear_binned_spec_dense_cache(self):
        self._binned_spec_dense_cache = None

    @property
    def has_binned_spec(self):
        return self._binned_inds is not None and self._binned_vals is not None

    @property
    def num_peaks(self):
        return np.sum(self.intens > 0)  # peaks must have intensities

    def __repr__(self):
        _repr = f'MassSpec ('
        _repr += ', '.join([f'{k}={v}' for k, v in self.info.items()])
        for key in ('probs', 'brokens', 'masses', 'masses_no_adduct', 'frag_form_vecs', 'frags', 'intens'):
            obj = getattr(self, key)
            if obj is not None:
                _repr += f', {key}={obj.shape}'
        if self.has_binned_spec:
            _repr += f', binned_spec_sparse=({len(self._binned_inds)}, {self._num_bins})'
        _repr += ')'
        return _repr

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise AttributeError(f'MassSpec object has no attribute {item}')

    def __contains__(self, item):
        return hasattr(self, item)

    def _merge_spec_to_tup(self, spec_tup_to_merge: dict=None, merge_method='sum'):
        if not spec_tup_to_merge:
            spec_tup_to_merge = {}
        if not self.has_masses or not self.has_intens:
            raise ValueError('both masses and intens must be non-empty')
        for idx in range(self.num_peaks):
            mz, inten = self.masses[idx], self.intens[idx]
            if self.has_frags:
                frag = self.frags[idx]
                frag_int = self.int_frags[idx]
            else:
                frag = None
                frag_int = 0
            mz_frag = f'{mz:.4f}_{frag_int}'
            cur_tup = spec_tup_to_merge.get(mz_frag)
            if cur_tup is None:
                spec_tup_to_merge[mz_frag] = [mz, inten, frag]
            else:
                if merge_method == 'sum':
                    cur_tup[1] += inten
                elif merge_method == 'max':
                    cur_tup[1] = max(inten, cur_tup[1])
                else:
                    raise ValueError(f'Unknown merge_method {merge_method}')
        return spec_tup_to_merge

    def _merge_tup_to_spec(self, spec_tup_to_merge: dict):
        merged_mz, merged_inten, merged_frag = [], [], []
        for tup in spec_tup_to_merge.values():
            merged_mz.append(tup[0])
            merged_inten.append(tup[1])
            merged_frag.append(tup[2])
        merged_mz = np.array(merged_mz)
        merged_inten = np.array(merged_inten)
        merged_frag = np.array(merged_frag)
        merged_inten = merged_inten / merged_inten.max()

        merged_spec = MassSpec(
            collision_energy='nan',
            root_canonical_smiles=self.root_canonical_smiles,
            masses=merged_mz,
            intens=merged_inten,
            frags=merged_frag if merged_frag[0] is not None else None,
            **self.meta
        )

        return merged_spec

    def inten_thresh(self, thresh=0.0001) -> "MassSpec":
        if not self.has_intens:
            raise ValueError('spectrum must have intensities!')

        new_masses, new_masses_no_adduct, new_probs, new_frags, new_frag_form_vecs = [], [], [], [], []
        new_brokens, new_intens = [], []
        for i in range(self.num_peaks):
            if self.intens[i] >= thresh:
                if self.has_masses:           new_masses.append(self.masses[i])
                if self.has_masses_no_adduct: new_masses_no_adduct.append(self.masses_no_adduct[i])
                if self.has_probs:            new_probs.append(self.probs[i])
                if self.has_frags:            new_frags.append(self.frags[i])
                if self.has_frag_form_vecs:   new_frag_form_vecs.append(self.frag_form_vecs[i])
                if self.has_brokens:          new_brokens.append(self.brokens[i])
                new_intens.append(self.intens[i])

        return MassSpec(
            collision_energy=self.collision_energy,
            root_canonical_smiles=self.root_canonical_smiles,
            adduct=self.adduct,
            remark=self.remark,
            probs=new_probs,
            brokens=new_brokens,
            masses=new_masses,
            masses_no_adduct=new_masses_no_adduct,
            frag_form_vecs=new_frag_form_vecs,
            frags=new_frags,
            intens=new_intens,
            **self.meta
        )

    def sort_peaks_by_mz(self):
        if not self.has_masses:
            raise ValueError('spectrum must have masses!')
        sortind = np.argsort(self.masses)

        self.masses = self.masses[sortind]
        if self.has_masses_no_adduct: self.masses_no_adduct = self.masses_no_adduct[sortind]
        if self.has_probs:            self.probs = self.probs[sortind]
        if self.has_frags:            self.frags = self.frags[sortind]
        if self.has_frag_form_vecs:   self.frag_form_vecs = self.frag_form_vecs[sortind]
        if self.has_brokens:          self.brokens = self.brokens[sortind]
        if self.has_intens:           self.intens = self.intens[sortind]

    @staticmethod
    def _standardize_ce(ce):
        return CompositeMassSpec._standardize_ce(ce)

    def nce_to_ev(self, parentmass=None):
        """Change NCE energies to eV"""
        if parentmass is None:
            parentmass = self.parent_mass
        assert parentmass is not None
        self.collision_energy = nce_to_ev(self.collision_energy, parentmass)

    def process_spec_file(self, parentmass=None, denoise=False, max_num_inten=20, inten_thresh=0.05):
        if parentmass is not None:
            meta = {'parentmass': parentmass}
        elif self.parent_mass is not None:
            meta = {'parentmass': self.parent_mass}
        else:
            meta = {}
        ce_key = self._standardize_ce(self.collision_energy)
        processed_specs = process_spec_file(meta, [(ce_key, self.spec)], merge_specs=False) or {}
        new_spec = processed_specs.get(ce_key)
        if new_spec is None:
            new_spec = np.empty((0, 2))
        self.masses, self.intens = new_spec[:, 0], new_spec[:, 1]

        if denoise: self.denoise(max_num_inten, inten_thresh)

        return self

    def denoise(self, max_num_inten=20, inten_thresh=0.05):
        """"""
        if self.has_masses and self.has_intens:
            new_spec = max_inten_spec(self.spec, max_num_inten=max_num_inten, inten_thresh=inten_thresh)
            self.masses, self.intens = new_spec[:, 0], new_spec[:, 1]
            return electronic_denoising(self)
        else:
            raise ValueError('Spectrum object should have both masses and intens')

    def bin_spectrum(self, mass_upper_limit=1500, num_bins=15000, pool_fn='add'):
        """turn spectrum into binned vectors, and store the binned spectrum (accessible via self.binned_spec)"""
        self.ensure_binned_spectrum(mass_upper_limit=mass_upper_limit, num_bins=num_bins, pool_fn=pool_fn, force=True)
        return self.binned_spec

    def _filtered_binned_sparse(self, ignore_mass=None):
        inds, vals = self.binned_spec_sparse
        if ignore_mass is None:
            return inds.astype(np.int64, copy=False), vals.astype(np.float32, copy=False)
        if self._num_bins is None or self._mass_upper_limit is None:
            raise ValueError("num_bins and upper_limit are required to filter binned spectra by mass")
        max_ind = int(ignore_mass * (self._num_bins / self._mass_upper_limit))
        keep = inds < max_ind
        return inds[keep].astype(np.int64, copy=False), vals[keep].astype(np.float32, copy=False)

    def _check_binned_compatibility(self, other):
        if self._num_bins != other._num_bins or self._mass_upper_limit != other._mass_upper_limit:
            raise ValueError(
                "Binned spectra must share num_bins and upper_limit for sparse similarity "
                f"(self={self._num_bins}, {self._mass_upper_limit}; "
                f"other={other._num_bins}, {other._mass_upper_limit})"
            )

    @staticmethod
    def _entropy_from_sparse_probs(vals):
        vals = vals[vals > 0]
        return -np.sum(vals * np.log(vals + 1e-22))

    @staticmethod
    def _cos_sim_sparse(self_inds, self_vals, other_inds, other_vals):
        if len(self_vals) == 0 or len(other_vals) == 0:
            return 0.0

        other_lookup = {int(ind): float(val) for ind, val in zip(other_inds, other_vals)}
        dot = sum(float(val) * other_lookup.get(int(ind), 0.0) for ind, val in zip(self_inds, self_vals))
        norm_self = np.sqrt(np.dot(self_vals, self_vals)) + 1e-22
        norm_other = np.sqrt(np.dot(other_vals, other_vals)) + 1e-22
        return dot / (norm_self * norm_other)

    @staticmethod
    def _entr_sim_sparse(self_inds, self_vals, other_inds, other_vals):
        self_sum = self_vals.sum()
        other_sum = other_vals.sum()
        if self_sum <= 0 or other_sum <= 0:
            return 0.0

        self_probs = self_vals / (self_sum + 1e-22)
        other_probs = other_vals / (other_sum + 1e-22)
        entropy_self = MassSpec._entropy_from_sparse_probs(self_probs)
        entropy_other = MassSpec._entropy_from_sparse_probs(other_probs)

        other_lookup = {}
        for ind, prob in zip(other_inds, other_probs):
            ind = int(ind)
            other_lookup[ind] = other_lookup.get(ind, 0.0) + float(prob)

        entropy_mix = 0.0
        seen = set()
        for ind, prob in zip(self_inds, self_probs):
            ind = int(ind)
            seen.add(ind)
            mix_prob = (float(prob) + other_lookup.get(ind, 0.0)) / 2
            entropy_mix -= mix_prob * np.log(mix_prob + 1e-22)
        for ind, prob in zip(other_inds, other_probs):
            ind = int(ind)
            if ind in seen:
                continue
            mix_prob = float(prob) / 2
            entropy_mix -= mix_prob * np.log(mix_prob + 1e-22)
        return 1 - (2 * entropy_mix - entropy_self - entropy_other) / np.log(4)

    def similarity(self, other, ignore_mass=None, metric='entropy'):
        if metric == 'entropy':
            return self.entr_sim(other, ignore_mass)
        elif metric == 'cosine':
            return self.cos_sim(other, ignore_mass)
        else:
            raise ValueError(f'Unknown metric {metric}')

    def cos_sim(self, other, ignore_mass=None):
        """
        Compute cosine similarity between two MassSpec object

        Args:
            other: compared MassSpec object
            ignore_mass: any peaks with a larger mass will be ignored
        """
        _ = self.binned_spec_sparse
        _ = other.binned_spec_sparse
        self._check_binned_compatibility(other)
        self_inds, self_vals = self._filtered_binned_sparse(ignore_mass)
        other_inds, other_vals = other._filtered_binned_sparse(ignore_mass)
        return self._cos_sim_sparse(self_inds, self_vals, other_inds, other_vals)

    def entr_sim(self, other, ignore_mass=None):
        """
        Compute entropy similarity between two MassSpec object

        Args:
            other: compared MassSpec object
            ignore_mass: any peaks with a larger mass will be ignored
        """
        _ = self.binned_spec_sparse
        _ = other.binned_spec_sparse
        self._check_binned_compatibility(other)
        self_inds, self_vals = self._filtered_binned_sparse(ignore_mass)
        other_inds, other_vals = other._filtered_binned_sparse(ignore_mass)
        return self._entr_sim_sparse(self_inds, self_vals, other_inds, other_vals)


class CompositeMassSpec:
    """
    Multiple mass spectra corresponding to the same molecule (e.g., different collision energies)
    If there are multiple spectra with the same collision energy, they will be merged.
    """
    def __init__(self, mass_spec_list: Union[List[MassSpec], Dict[str, MassSpec]]):
        self.ce_to_ms = {}
        self.ce_to_merged_ms = {}
        self.root_canonical_smiles = None
        if isinstance(mass_spec_list, dict):
            mass_spec_list = mass_spec_list.values()
        for ms_obj in mass_spec_list:
            ce = ms_obj.collision_energy
            standardized_ce = self._standardize_ce(ce)
            if standardized_ce in self.ce_to_ms:
                # Merge with existing
                existing_ms = self.ce_to_ms[standardized_ce]
                
                # Use merge_specs to properly bin and sum peaks
                specs_to_merge = {'old': existing_ms.spec, 'new': ms_obj.spec}
                merged_result = merge_specs(specs_to_merge, precision=4, merge_method='sum')
                merged_spec_ar = list(merged_result.values())[0] # extract the single merged spectrum
                
                new_masses = merged_spec_ar[:, 0]
                new_intens = merged_spec_ar[:, 1]
                
                # Create merged MassSpec
                merged_ms = MassSpec(
                    collision_energy=existing_ms.collision_energy,
                    masses=new_masses,
                    intens=new_intens,
                    root_canonical_smiles=existing_ms.root_canonical_smiles,
                    adduct=existing_ms.adduct,
                    remark=existing_ms.remark,
                    # Propagate other attributes if needed, assuming basic raw spectra here
                    **existing_ms.meta
                )
                self.ce_to_ms[standardized_ce] = merged_ms
            else:
                self.ce_to_ms[standardized_ce] = ms_obj

            if self.root_canonical_smiles is None:
                self.root_canonical_smiles = ms_obj.root_canonical_smiles
            else:
                assert self.root_canonical_smiles == ms_obj.root_canonical_smiles

    def __repr__(self):
        _repr = f'CompositeMassSpec [\n'
        _repr += ';\n'.join(['  ' + ms.__repr__() for ms in self.values()])
        _repr += ']'
        return _repr

    def __len__(self):
        return len(self.ce_to_ms)

    @staticmethod
    def _standardize_ce(ce: Union[str, float]):
        if isinstance(ce, str) and 'collision' in ce:
            ce = chem_utils.get_collision_energy(ce)
        ce_key = f'{float(ce):.0f}'
        return ce_key

    def keys(self):
        return self.ce_to_ms.keys()

    def values(self):
        return self.ce_to_ms.values()

    def items(self):
        return self.ce_to_ms.items()

    def __getitem__(self, item):
        return self.ce_to_ms[self._standardize_ce(item)]

    @property
    def num_peaks(self):
        return [v.num_peaks for v in self.values()]

    def process_spec_file(self, parentmass=None, denoise=False, max_num_inten=20, inten_thresh=0.05):
        self.ce_to_merged_ms = {}
        filtered_ms = {}
        for k, v in self.ce_to_ms.items():
            processed_ms = v.process_spec_file(parentmass, denoise, max_num_inten, inten_thresh)
            if processed_ms.num_peaks > 0:
                filtered_ms[k] = processed_ms
        self.ce_to_ms = filtered_ms

    def bin_spectrum(self, mass_upper_limit=1500, num_bins=15000, pool_fn='add'):
        for v in self.ce_to_ms.values():
            v.bin_spectrum(mass_upper_limit, num_bins, pool_fn)

    def ensure_binned_spectrum(self, mass_upper_limit=1500, num_bins=15000, pool_fn='add', force=False):
        for v in self.ce_to_ms.values():
            v.ensure_binned_spectrum(
                mass_upper_limit=mass_upper_limit,
                num_bins=num_bins,
                pool_fn=pool_fn,
                force=force,
            )

    def nce_to_ev(self, parentmass=None):
        self.ce_to_merged_ms = {}
        new_ce_to_ms = {}
        for v in self.ce_to_ms.values():
            v.nce_to_ev(parentmass)
            new_ce_to_ms[self._standardize_ce(v.collision_energy)] = v
        self.ce_to_ms = new_ce_to_ms

    def merge_spectra(self, energies: list=None, merge_method='sum') -> MassSpec:
        if energies is None: # merge all
            energies = list(self.keys())
        merged_key = '_'.join(sorted(energies)) + '_' + merge_method
        if merged_key not in self.ce_to_merged_ms:
            mz_frag_to_tup = {}
            for ce in energies:
                spec_data = self.ce_to_ms[ce]
                mz_frag_to_tup = spec_data._merge_spec_to_tup(mz_frag_to_tup, merge_method)
            self.ce_to_merged_ms[merged_key] = spec_data._merge_tup_to_spec(mz_frag_to_tup)
        return self.ce_to_merged_ms[merged_key]

    def entr_sim(self, *args, **kwargs):
        return self.similarity(*args, **kwargs, metric='entropy')

    def cos_sim(self, *args, **kwargs):
        return self.similarity(*args, **kwargs, metric='cosine')

    def similarity(self, other, ignore_mass=None, merge_method='unmerged', stepped_ce=None, metric='entropy',
                   aggregate='mean', return_ce=False):
        """
        merge_method: how ``other`` spectrum is merged, from 'unmerged', 'stepped', 'unknown'.
            if 'unmerged', ``other`` is CompositeMassSpec, only compare spectra with matched collision energy
            if 'stepped', ``other`` is MassSpec, means stepped collision energy, merge spectra at energies in ``stepped_ce``
            if 'unknown', ``other`` is MassSpec, match to all spectra in self and return the best match
            if 'merged', ``other`` is CompositeMassSpec, merge both spectra and then compare
        """
        if aggregate == 'mean' or aggregate == 'avg':
            agg_func = np.mean
        elif aggregate == 'sum' or aggregate == 'add':
            agg_func = np.sum
        elif aggregate == 'none':
            agg_func = lambda x: x
        else:
            raise ValueError(f'Unknown aggregation method {aggregate}')

        if merge_method == 'unmerged':
            common_ces = set(other.keys()) & set(self.keys())
            if len(common_ces) == 0:
                raise ValueError(f'No matched collision energy self: {", ".join(self.keys())}, other: {", ".join(other.keys())}')
            sims = []
            for ce in common_ces:
                sim = self[ce].similarity(other[ce], ignore_mass, metric)
                sims.append(sim)
            if return_ce:
                return agg_func(sims), [float(_) for _ in common_ces]
            else:
                return agg_func(sims)
        elif merge_method == 'stepped':
            if stepped_ce is not None:
                for ce in stepped_ce:
                    if not ce in self.keys():
                        raise ValueError(f'Collision energy does not match. Stepped energies: {stepped_ce}, '
                                         f'compared energies: {self.keys()}')
            merged_spec = self.merge_spectra(stepped_ce)
            if isinstance(other, CompositeMassSpec) and len(other) == 1:
                other = list(other.values())[0]
            assert isinstance(other, MassSpec)
            sim = merged_spec.similarity(other, ignore_mass, metric)
            if return_ce:
                return sim, [float(_) for _ in stepped_ce]
            else:
                return sim
        elif merge_method == 'unknown':
            sims = []
            for ce in self.keys():
                sim = self[ce].similarity(other, ignore_mass, metric)
                sims.append(sim)
            if return_ce:
                best_ce = list(self.keys())[np.argmax(sims)]
                return np.max(sims), float(best_ce)
            else:
                return np.max(sims)
        elif merge_method == 'merged':
            merged_self = self.merge_spectra()
            merged_other = other.merge_spectra()
            sim = merged_self.similarity(merged_other, ignore_mass, metric)
            if return_ce:
                raise RuntimeWarning('return_ce=True is not supported. No collision energy returned.')
            return sim
        else:
            raise ValueError(f'Unknown merge method {merge_method}')

def _normalize_str_list(strings):
    # HDF5 string datasets should be written from plain Python strings, not
    # numpy object arrays of bytes, which h5py cannot always convert.
    return ["" if s is None else str(s) for s in strings]

def _decode_str_array_py(arr):
    # arr may be dtype object bytes; convert to Python str; empty => None
    out = []
    for b in arr.tolist():
        if b is None:
            out.append(None)
            continue
        if isinstance(b, str):
            s = b
        else:
            s = b.decode("utf-8")
        out.append(None if s == "" else s)
    return out

def _create_str_dataset(grp, name, values, maxshape, chunks):
    dt = h5py.string_dtype(encoding="utf-8")
    ds = grp.create_dataset(
        name,
        shape=(len(values),),
        dtype=dt,
        maxshape=maxshape,
        chunks=chunks,
    )
    if len(values) > 0:
        ds[:] = _normalize_str_list(values)
    return ds

def _is_h5_writable(h5f: h5py.File) -> bool:
    # h5py File.mode examples: 'r', 'r+', 'w', 'w-', 'x', 'a'
    # Writable modes: r+, w, w-, x, a
    return h5f.mode in ("r+", "w", "w-", "x", "a")

def _parse_leaf_path(rel_path: str):
    """
    Old layout leaf groups look like:
      name/collision XX
      name/<remark>/collision XX

    Return: (name, remark_or_None, collision_key_str)
    """
    parts = rel_path.split("/")
    if len(parts) == 2:
        name, ce = parts[0], parts[1]
        remark = None
    elif len(parts) == 3:
        name, remark, ce = parts[0], parts[1], parts[2]
    else:
        raise ValueError(f"Unexpected leaf path format: {rel_path}")
    return name, remark if remark != "" else None, ce

def _build_manifest_from_old_structure(h5f: h5py.File):
    """
    Traverse once, collect leaf groups tagged with attrs["override"].
    Returns dict with arrays:
      - leaf_path: full leaf group path relative to root (e.g., 'C10H10/.../collision 40')
      - name
      - remark (None or str)
      - ce_key (e.g., 'collision 40')
    """
    leaf_paths = []

    def visitor(name, obj):
        # name is path relative to the group called on (we call on root)
        # obj can be Dataset or Group
        if isinstance(obj, h5py.Group):
            # leaf "data group" marker in old code
            if obj.attrs and "override" in obj.attrs:
                leaf_paths.append(name)

    h5f.visititems(visitor)

    names = []
    remarks = []
    ce_keys = []
    for lp in leaf_paths:
        n, r, ce = _parse_leaf_path(lp)
        names.append(n)
        remarks.append(r)
        ce_keys.append(ce)

    return {
        "version": _MANIFEST_VERSION,
        "leaf_path": leaf_paths,
        "name": names,
        "remark": remarks,      # list[str|None]
        "ce_key": ce_keys,      # list[str]
    }

def _embed_manifest_into_h5(h5f: h5py.File, manifest: dict, h5_path: Path):
    """
    Store manifest under a dedicated group in the HDF5 file.
    Safe to overwrite existing manifest group.
    """
    if _MANIFEST_GROUP in h5f:
        del h5f[_MANIFEST_GROUP]
    grp = h5f.create_group(_MANIFEST_GROUP)

    # Attach metadata
    grp.attrs["version"] = int(manifest["version"])
    grp.attrs["layout_version"] = int(_MANIFEST_LAYOUT_VERSION)

    # Create RESIZABLE datasets so we can append inside write().
    # Use chunking for efficient append.
    n = len(manifest["leaf_path"])
    chunk = (max(1, min(4096, n)),)
    _create_str_dataset(grp, "leaf_path", manifest["leaf_path"], maxshape=(None,), chunks=chunk)
    _create_str_dataset(grp, "name", manifest["name"], maxshape=(None,), chunks=chunk)
    _create_str_dataset(grp, "remark", manifest["remark"], maxshape=(None,), chunks=chunk)
    _create_str_dataset(grp, "ce_key", manifest["ce_key"], maxshape=(None,), chunks=chunk)
    grp.attrs["count"] = int(n)

def _load_embedded_manifest(h5f: h5py.File, h5_path: Path):
    if _MANIFEST_GROUP not in h5f:
        return None

    grp = h5f[_MANIFEST_GROUP]
    try:
        version = int(grp.attrs.get("version", -1))
        if version != _MANIFEST_VERSION:
            return None

        leaf_path = _decode_str_array_py(grp["leaf_path"][()])
        names = _decode_str_array_py(grp["name"][()])
        remarks = _decode_str_array_py(grp["remark"][()])
        ce_keys = _decode_str_array_py(grp["ce_key"][()])

        return {
            "version": version,
            "leaf_path": leaf_path,
            "name": names,
            "remark": remarks,
            "ce_key": ce_keys,
        }
    except Exception:
        return None

def _manifest_can_append(grp: h5py.Group) -> bool:
    """
    Return True if manifest datasets are present and resizable (maxshape None).
    Older manifests may exist but not be resizable.
    """
    try:
        for k in ("leaf_path", "name", "remark", "ce_key"):
            ds = grp[k]
            # must be 1D and resizable along axis 0
            if ds.ndim != 1:
                return False
            if ds.maxshape is None:
                return False
            if ds.maxshape[0] is not None:
                return False
        return True
    except Exception:
        return False

def _manifest_append_one(h5f: h5py.File, leaf_path: str, name: str, remark, ce_key: str):
    """
    Append a single entry into the embedded manifest group.
    Assumes file is writable.
    """
    if remark is None:
        remark = ""
    # Ensure group exists and is appendable
    if _MANIFEST_GROUP not in h5f:
        grp = h5f.create_group(_MANIFEST_GROUP)
        grp.attrs["version"] = int(_MANIFEST_VERSION)
        grp.attrs["layout_version"] = int(_MANIFEST_LAYOUT_VERSION)
        # start empty, chunked, resizable
        chunk = (4096,)
        for k in ("leaf_path", "name", "remark", "ce_key"):
            _create_str_dataset(grp, k, [], maxshape=(None,), chunks=chunk)
        grp.attrs["count"] = 0
    grp = h5f[_MANIFEST_GROUP]

    # If not appendable (old layout), rebuild to new layout first.
    if not _manifest_can_append(grp):
        manifest = _build_manifest_from_old_structure(h5f)
        _embed_manifest_into_h5(h5f, manifest, Path(getattr(h5f, "filename", "")))
        grp = h5f[_MANIFEST_GROUP]

    # Append
    idx = int(grp.attrs.get("count", grp["leaf_path"].shape[0]))
    new_n = idx + 1
    for k, val in (
        ("leaf_path", leaf_path),
        ("name", name),
        ("remark", remark),
        ("ce_key", ce_key),
    ):
        ds = grp[k]
        ds.resize((new_n,))
        ds[idx] = str(val)
    grp.attrs["count"] = int(new_n)

class PredSpecDB:
    """
    Data structure for predicted spectrum database
    """
    _SPECIAL_ROOT_GROUPS = {_MANIFEST_GROUP}

    @staticmethod
    def _path_sort_key(path: Path):
        stem = path.stem
        shard_match = re.search(r"_shard(\d+)(?:_chunk_\d+)?$", stem)
        chunk_match = re.search(r"_chunk_(\d+)$", stem)
        shard_idx = int(shard_match.group(1)) if shard_match else -1
        chunk_idx = int(chunk_match.group(1)) if chunk_match else -1
        return (shard_idx, chunk_idx, path.name)

    @classmethod
    def _resolve_h5_paths(cls, h5_path: Path, mode: str, num_h5s: int):
        if num_h5s > 1:
            return [
                h5_path.parent / (h5_path.stem + f'_chunk_{i}' + h5_path.suffix)
                for i in range(num_h5s)
            ]

        if mode == 'w':
            return [h5_path]

        shard_chunk_paths = sorted(
            h5_path.parent.glob(f"{h5_path.stem}_shard*_chunk_*{h5_path.suffix}"),
            key=cls._path_sort_key,
        )
        if shard_chunk_paths:
            return shard_chunk_paths

        shard_paths = sorted(
            h5_path.parent.glob(f"{h5_path.stem}_shard*{h5_path.suffix}"),
            key=cls._path_sort_key,
        )
        if shard_paths:
            return shard_paths

        chunk_paths = sorted(
            h5_path.parent.glob(f"{h5_path.stem}_chunk_*{h5_path.suffix}"),
            key=cls._path_sort_key,
        )
        if chunk_paths:
            return chunk_paths

        return [h5_path]

    def __init__(self, h5_path, mode="r", num_h5s=1,
                 has_probs=True, has_brokens=True, has_masses=False, has_masses_no_adduct=True, has_frag_form_vecs=True,
                 has_frags=True, has_intens=False, has_pulled_atoms=False, has_binned_spec=False,
                 h5_persistent=None):
        """
        Args:
            h5_persistent: if False, the h5 object(s) will be loaded on-demand
                           (default: True for write mode, False for read mode)
        """
        h5_path = Path(h5_path)
        self.all_h5_paths = self._resolve_h5_paths(h5_path, mode, num_h5s)

        self.mode = mode
        self.h5_persistent = h5_persistent
        self._name_to_h5_idx = {}
        h5_dataset_0 = HDF5Dataset(self.all_h5_paths[0], self.mode)
        if self.mode == 'w':  # create new file
            self.has_probs   = has_probs
            self.has_brokens = has_brokens
            self.has_masses  = has_masses
            self.has_masses_no_adduct = has_masses_no_adduct
            self.has_frag_form_vecs = has_frag_form_vecs
            self.has_frags   = has_frags
            self.has_intens  = has_intens
            self.has_binned_spec = has_binned_spec
            self.root_key_dict = {
                "probs": self.has_probs,
                "masses": self.has_masses,
                "masses_no_adduct": self.has_masses_no_adduct,
                "intens": self.has_intens,
                "brokens": self.has_brokens,
                "frag_form_vecs": self.has_frag_form_vecs,
                "frags": self.has_frags,
                "binned_spec": self.has_binned_spec,
            }
            h5_dataset_0.update_attr('.', self.root_key_dict)
            if self.h5_persistent is None:
                self.h5_persistent = True  # default persistent H5 objects for write mode
        elif self.mode == 'r' or self.mode == 'r+' or self.mode == 'a':
            self.root_key_dict = h5_dataset_0.read_attr('.')
            safe_root_key_get = lambda x: self.root_key_dict[x] if x in self.root_key_dict else None
            self.has_probs            = safe_root_key_get('probs')
            self.has_masses           = safe_root_key_get('masses')
            self.has_masses_no_adduct = safe_root_key_get('masses_no_adduct')
            self.has_intens           = safe_root_key_get('intens')
            self.has_brokens          = safe_root_key_get('brokens')
            self.has_frag_form_vecs   = safe_root_key_get('frag_form_vecs')
            self.has_frags            = safe_root_key_get('frags')
            self.has_binned_spec      = safe_root_key_get('binned_spec')
            if self.h5_persistent is None:
                self.h5_persistent = False  # default non-persistent H5 objects for read mode (to support parallel read)
        else:
            raise ValueError(f'Unknown mode={self.mode}')

        if self.h5_persistent:
            self.h5datasets = [h5_dataset_0] + [HDF5Dataset(p, self.mode) for p in self.all_h5_paths[1:]]
        else:
            self.h5datasets = None

    @staticmethod
    def _binned_index_u8_len(num_bins):
        return max(1, (max(1, int(num_bins) - 1).bit_length() + 7) // 8)

    @classmethod
    def _encode_binned_indices(cls, indices, num_bins):
        u8_len = cls._binned_index_u8_len(num_bins)
        encoded = indices.astype('<u4', copy=False).view(np.uint8).reshape(-1, 4)[:, :u8_len]
        return encoded, u8_len

    @staticmethod
    def _decode_binned_indices(encoded, u8_len):
        padded = np.zeros((encoded.shape[0], 4), dtype=np.uint8)
        padded[:, :u8_len] = encoded
        return padded.view('<u4').reshape(-1).astype(np.int64)

    def _read_binned_spec(self, h5_dataset, full_name, key_dict, fdata, udata, float_col, uint_col):
        if "binned_u8_len" not in key_dict or fdata is None or udata is None:
            raise ValueError(f"Sparse binned spectrum payload missing for {full_name}")

        binned_rows = int(key_dict.get("binned_rows", 0))
        u8_len = int(key_dict["binned_u8_len"])
        if binned_rows == 0:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)
        indices = self._decode_binned_indices(udata[:binned_rows, uint_col:uint_col + u8_len], u8_len)
        values = fdata[:binned_rows, float_col].astype(np.float32, copy=False)
        return indices.astype(np.uint32, copy=False), values

    def write(self, name, spec: MassSpec, replace_name_by_formula=False):
        """write one spectrum"""
        key_dict = {
            "probs": spec.has_probs,
            "masses": spec.has_masses,
            "masses_no_adduct": spec.has_masses_no_adduct,
            "intens": spec.has_intens,
            "brokens": spec.has_brokens,
            "frag_form_vecs": spec.has_frag_form_vecs,
            "frags": spec.has_frags,
            "binned_spec": spec.has_binned_spec,
        }
        if replace_name_by_formula and spec.root_canonical_smiles is not None:
            name = chem_utils.form_from_smi(spec.root_canonical_smiles)

        h5_dataset = self._get_h5_dataset(name)
        full_name  = self._get_full_name(name, spec.collision_energy, spec.remark)

        if full_name in h5_dataset:
            logging.warning(f'{full_name} already exists, skipping...')
            return

        float_arrs = []
        if spec.has_probs: float_arrs.append(spec.probs.astype(np.float32))
        if spec.has_masses: float_arrs.append(spec.masses.astype(np.float32))
        if spec.has_masses_no_adduct: float_arrs.append(spec.masses_no_adduct.astype(np.float32))
        if spec.has_intens: float_arrs.append(spec.intens.astype(np.float32))

        uint_arrs = []
        if spec.has_brokens: uint_arrs.append(spec.brokens.astype(np.uint8)[:, None])
        if spec.has_frag_form_vecs: uint_arrs.append(spec.frag_form_vecs.astype(np.uint8))
        if spec.has_frags: uint_arrs.append(nn_utils.encode_bin_to_uint8(spec.frags).astype(np.uint8))
        main_rows = 0
        if len(float_arrs) > 0:
            main_rows = len(float_arrs[0])
        elif len(uint_arrs) > 0:
            main_rows = uint_arrs[0].shape[0]

        binned_rows = 0
        binned_u8_len = 0
        binned_vals = None
        binned_inds_u8 = None
        if spec.has_binned_spec:
            binned_inds, binned_vals = spec.binned_spec_sparse
            binned_inds = binned_inds.astype(np.uint32, copy=False)
            binned_vals = binned_vals.astype(np.float32, copy=False)
            binned_inds_u8, binned_u8_len = self._encode_binned_indices(binned_inds, spec._num_bins)
            binned_rows = len(binned_inds)

        num_rows = max(main_rows, binned_rows)

        if len(float_arrs) > 0 or spec.has_binned_spec:
            num_float_cols = len(float_arrs) + (1 if spec.has_binned_spec else 0)
            fdata = np.zeros((num_rows, num_float_cols), dtype=np.float32)
            if len(float_arrs) > 0:
                fdata[:main_rows, :len(float_arrs)] = np.stack(float_arrs, axis=1)
            if spec.has_binned_spec and binned_rows > 0:
                fdata[:binned_rows, len(float_arrs)] = binned_vals
            h5_dataset.write_data(full_name + '/f', fdata)

        if len(uint_arrs) > 0 or spec.has_binned_spec:
            base_uint_cols = sum(arr.shape[1] for arr in uint_arrs)
            num_uint_cols = base_uint_cols + (binned_u8_len if spec.has_binned_spec else 0)
            udata = np.zeros((num_rows, num_uint_cols), dtype=np.uint8)
            cur_col = 0
            for arr in uint_arrs:
                next_col = cur_col + arr.shape[1]
                udata[:main_rows, cur_col:next_col] = arr
                cur_col = next_col
            if spec.has_binned_spec and binned_rows > 0:
                udata[:binned_rows, cur_col:cur_col + binned_u8_len] = binned_inds_u8
            h5_dataset.write_data(full_name + '/u', udata)

        if not all([self.root_key_dict[k] == v for k, v in key_dict.items()]):
            # h5 group will have a different attribute if any attribute is different from root
            key_dict["override"] = True
            h5_dataset.update_attr(full_name, key_dict)
        else:
            h5_dataset.update_attr(full_name, {"override": False})
        attr_updates = {"main_rows": main_rows}
        if spec.has_frags:
            attr_updates["frag_bits"] = spec.frags.shape[-1]
        if spec.has_binned_spec:
            binned_attr_updates = {
                "num_bins": spec._num_bins,
                "upper_limit": spec._mass_upper_limit,
                "binned_rows": binned_rows,
                "binned_u8_len": binned_u8_len,
            }
            attr_updates.update({k: v for k, v in binned_attr_updates.items() if v is not None})
        h5_dataset.update_attr(full_name, attr_updates)
        h5_dataset.update_attr(full_name, spec.info)

        # ---- update embedded manifest incrementally ----
        # The leaf group path in old/new layout is exactly full_name
        # e.g. "C10H10/remarkA/collision 40" or "C10H10/collision 40"
        try:
            if _is_h5_writable(h5_dataset.h5_obj):
                ce_key = self._get_collision_str(spec.collision_energy)
                _manifest_append_one(
                    h5_dataset.h5_obj,
                    leaf_path = full_name,
                    name = name,
                    remark = spec.remark,
                    ce_key = ce_key,
                )
        except Exception:
            # Fail-open: do not break writes if manifest update fails.
            pass

    def read(self, name, collision_energy=None, remark=None, ignore_keys=None):
        """read a specific spectrum"""
        h5_dataset = self._get_h5_dataset(name)
        try:
            return self._read_from_dataset(
                h5_dataset,
                name=name,
                collision_energy=collision_energy,
                remark=remark,
                ignore_keys=ignore_keys,
            )
        finally:
            if not self.h5_persistent:
                h5_dataset.close()

    def _read_from_dataset(self, h5_dataset, name, collision_energy=None, remark=None, ignore_keys=None):
        """read a specific spectrum from a specific HDF5 dataset"""
        full_name  = self._get_full_name(name, collision_energy, remark)

        key_dict = h5_dataset.read_attr(full_name)
        if not key_dict["override"]:
            key_dict.update(self.root_key_dict)

        del key_dict["override"]

        def in_key_dict(x):
            if ignore_keys is not None:
                assert is_iterable(ignore_keys)
                if x in ignore_keys:
                    return False
            return x in key_dict and key_dict[x]

        spec_dict = {}
        spec_dict.update(key_dict)
        spec_dict.pop("pulled_atoms", None)
        try:
            fdata = h5_dataset.read_data(full_name + '/f')
        except:
            #case of no f data
            fdata = None
        main_rows = int(key_dict.get("main_rows", fdata.shape[0] if fdata is not None else 0))
        cur_idx = 0
        for key in ("probs", "masses", "masses_no_adduct", "intens"):
            if in_key_dict(key):
                spec_dict[key] = fdata[:main_rows, cur_idx]
                cur_idx += 1
        try:
            udata = h5_dataset.read_data(full_name + '/u')
        except:
            #case of no u data
            udata = None
        ucur_idx = 0
        if in_key_dict("brokens"):
            spec_dict["brokens"] = udata[:main_rows, ucur_idx]
            ucur_idx += 1
        if in_key_dict("frag_form_vecs"):
            spec_dict["frag_form_vecs"] = udata[:main_rows, ucur_idx:ucur_idx+chem_utils.ELEMENT_DIM]
            ucur_idx += chem_utils.ELEMENT_DIM
        if in_key_dict("frags"):
            frag_u8_len = math.ceil(key_dict["frag_bits"] / 8)
            spec_dict["frags"] = nn_utils.decode_bin_from_uint8(udata[:main_rows, ucur_idx:ucur_idx+frag_u8_len], key_dict["frag_bits"])
            ucur_idx += frag_u8_len
        if in_key_dict("binned_spec"):
            spec_dict["binned_spec_sparse"] = self._read_binned_spec(h5_dataset, full_name, key_dict, fdata, udata, cur_idx, ucur_idx)
            spec_dict.pop("binned_spec", None)

        return MassSpec(**spec_dict)

    def read_from_name(self, name):
        """read all entries with the same name (all remarks, all collision energies)"""
        dataset_entries = self._get_entries_by_dataset(name)
        all_spec_dict = {}
        has_remark = False
        try:
            for h5_dataset, entries in dataset_entries:
                for c, r in entries:
                    spec_obj = self._read_from_dataset(h5_dataset, name, c, r)
                    c_key = f'collision {c}'
                    if r is None:
                        all_spec_dict[c_key] = spec_obj
                    else:
                        if r not in all_spec_dict:
                            all_spec_dict[r] = {}
                        all_spec_dict[r][c_key] = spec_obj
                        has_remark = True
        finally:
            if not self.h5_persistent:
                for h5_dataset, _ in dataset_entries:
                    h5_dataset.close()
        return all_spec_dict, has_remark

    def _get_chunk_index(self, name):
        if len(self.all_h5_paths) == 1:
            return 0
        else:
            return int(str_to_hash(name), base=16) % len(self.all_h5_paths)

    def _get_full_name(self, name, collision_energy, remark):
        full_name = name
        if remark is not None:
            full_name += '/' + str(remark)
        if collision_energy is not None:
            full_name += '/' + self._get_collision_str(collision_energy)
        return full_name

    @staticmethod
    def _get_collision_str(collision_energy):
        if collision_energy is not None:
            return f'collision {float(collision_energy):.0f}'
        else:
            return f'collision nan'

    def _get_h5_dataset(self, name):
        cur_i = self._name_to_h5_idx.get(name)
        if cur_i is None:
            cur_i = self._get_chunk_index(name)

        def _open_dataset(idx):
            if self.h5_persistent:
                return self.h5datasets[idx]
            return HDF5Dataset(self.all_h5_paths[idx], self.mode)

        if self.mode == 'w' or len(self.all_h5_paths) == 1:
            return _open_dataset(cur_i)

        candidate_indices = [cur_i] + [i for i in range(len(self.all_h5_paths)) if i != cur_i]
        for idx in candidate_indices:
            h5_dataset = _open_dataset(idx)
            if name in h5_dataset:
                self._name_to_h5_idx[name] = idx
                return h5_dataset
            if not self.h5_persistent:
                h5_dataset.close()

        return _open_dataset(cur_i)

    def get_all_names(self):
        if self.h5_persistent:
            all_h5s = self.h5datasets
        else:
            all_h5s = [HDF5Dataset(p, self.mode) for p in self.all_h5_paths]
        all_names = [h5.get_all_names() for h5 in all_h5s]
        all_names = [i for j in all_names for i in j if i not in self._SPECIAL_ROOT_GROUPS]
        return all_names

    def _iter_h5_datasets_with_name(self, name):
        if self.h5_persistent:
            all_h5s = self.h5datasets
        else:
            all_h5s = [HDF5Dataset(p, self.mode) for p in self.all_h5_paths]

        found = []
        for idx, h5_dataset in enumerate(all_h5s):
            if name in h5_dataset:
                found.append((idx, h5_dataset))

        if len(found) == 1:
            self._name_to_h5_idx[name] = found[0][0]
        elif len(found) > 1:
            self._name_to_h5_idx.pop(name, None)

        return found

    def _enumerate_entry_pairs(self, h5_dataset, name):
        if name not in h5_dataset.h5_obj:  # no entry
            return []

        root_obj = h5_dataset.h5_obj[name]
        if isinstance(root_obj, h5py.Dataset):
            # Legacy MAGMa trees store each entry as a dataset payload rather than
            # a nested PredSpecDB group. Let callers fall back to HDF5Dataset.
            return []

        all_paths = []
        from collections import deque
        queue = deque()

        for key in root_obj.keys():
            queue.append((root_obj[key], key))

        while queue:
            obj, path = queue.popleft()
            if obj.attrs and "override" in obj.attrs:
                all_paths.append(path)
            else:
                for key in obj.keys():
                    queue.append((obj[key], f"{path}/{key}"))

        entry_pairs = []
        for path in all_paths:
            keys = path.split('/')
            if len(keys) == 1:
                entry_pairs.append((chem_utils.get_collision_energy(keys[0]), None))
            elif len(keys) == 2:
                entry_pairs.append((chem_utils.get_collision_energy(keys[1]), keys[0]))
            else:
                raise ValueError(f'HDF5 path is not accepted: {path}')

        return entry_pairs

    def _get_entries_by_dataset(self, name):
        dataset_entries = []
        for _, h5_dataset in self._iter_h5_datasets_with_name(name):
            entry_pairs = self._enumerate_entry_pairs(h5_dataset, name)
            if entry_pairs:
                dataset_entries.append((h5_dataset, entry_pairs))
        return dataset_entries

    def get_entries(self, name, collision_energy=None):
        """return collision energies and remarks under each name"""
        dataset_entries = self._get_entries_by_dataset(name)
        try:
            entry_pairs = []
            for _, entries in dataset_entries:
                entry_pairs.extend(entries)
        finally:
            if not self.h5_persistent:
                for h5_dataset, _ in dataset_entries:
                    h5_dataset.close()

        if collision_energy is not None:
            entry_pairs = [
                (ce, remark)
                for ce, remark in entry_pairs
                if float(collision_energy) == float(ce)
            ]

        deduped_pairs = list(dict.fromkeys(entry_pairs))
        colli_engs = [ce for ce, _ in deduped_pairs]
        remarks = [remark for _, remark in deduped_pairs]
        return colli_engs, remarks

    def _ensure_manifest_for_h5(self, h5_dataset: "HDF5Dataset"):
        """
        Return a manifest dict for this HDF5Dataset.
        Priority:
          1) embedded manifest in HDF5 (if valid)
          2) sidecar manifest file (if valid)
          3) build from old structure, then cache (embed if possible else sidecar)
        """
        h5_path = Path(h5_dataset.path)

        # 1) Load from embedded
        manifest = _load_embedded_manifest(h5_dataset.h5_obj, h5_path)
        if manifest is not None:
            return manifest

        # 2) Build from missing structure
        manifest = _build_manifest_from_old_structure(h5_dataset.h5_obj)

        # Cache it
        try:
            if _is_h5_writable(h5_dataset.h5_obj):
                _embed_manifest_into_h5(h5_dataset.h5_obj, manifest, h5_path)
                h5_dataset.h5_obj.flush()
            else:
                print('writing manifest into h5 object failed. Please open h5 object in \'r+\' or \'a\' mode')
        except Exception:
            # If caching fails, still return the in-memory manifest
            pass

        return manifest

    def _iter_manifest_grouped(self, h5_dataset: "HDF5Dataset"):
        """
        Yield groups of entries: (name, remark, [ce_key...]) for a single HDF5 file.
        """
        manifest = self._ensure_manifest_for_h5(h5_dataset)
        # Group collision energies by (name, remark)
        buckets = defaultdict(list)
        for n, r, ce in zip(manifest["name"], manifest["remark"], manifest["ce_key"]):
            buckets[(n, r)].append(ce)
        for (n, r), ces in buckets.items():
            # Keep stable ordering for determinism (collision energies are strings 'collision 40')
            # Note: you can do a numeric sort if you want, but not required.
            yield n, r, sorted(ces)

    class _AllSpecsIterable:
        def __init__(self, db: "PredSpecDB", h5_list, ignore_keys):
            self.db = db
            self.h5_list = h5_list
            self.ignore_keys = ignore_keys

            # Precompute group counts using manifest (O(#entries), very cheap)
            self._groups = []
            for h5 in h5_list:
                for name, remark, ce_keys in db._iter_manifest_grouped(h5):
                    self._groups.append((h5, name, remark, ce_keys))

        def __len__(self):
            return len(self._groups)

        def __iter__(self):
            for i in range(self.__len__()):
                yield self.__getitem__(i)

        def __getitem__(self, index):
            n = len(self)

            # --- int indexing ---
            if isinstance(index, int):
                if index < 0:
                    index += n
                if index < 0 or index >= n:
                    raise IndexError(f"index {index} out of range for length {n}")
                return self.__get_one_item(index)

            # --- slice indexing ---
            if isinstance(index, slice):
                # Range handles all slice semantics, including negative step.
                rng = range(*index.indices(n))
                # Return a list, like normal sequences do.
                return [self.__get_one_item(i) for i in rng]

            # --- optional: fancy indexing (list/tuple/ndarray of ints) ---
            if isinstance(index, (list, tuple)):
                return [self[i] for i in index]

            raise TypeError(
                f"indices must be int or slice (or list/tuple of ints), not {type(index).__name__}"
            )

        def __get_one_item(self, index):
            h5, name, remark, ce_keys = self._groups[index]
            colli_spec_dict = {}
            for ce_key in ce_keys:
                ce_val = chem_utils.get_collision_energy(ce_key)
                colli_spec_dict[ce_key] = self.db.read(name, ce_val, remark, self.ignore_keys)

            cms = CompositeMassSpec(colli_spec_dict)

            if remark is None:
                return name, cms
            else:
                return name, remark, cms

    def get_all_specs(self, ignore_keys=None):
        """
        return all spectra in the database

        Yields:
            - If remarks exist for a name: (name, remark, CompositeMassSpec)
            - Else: (name, CompositeMassSpec)
        """
        if self.h5_persistent:
            all_h5s = self.h5datasets
        else:
            all_h5s = [HDF5Dataset(p, self.mode) for p in self.all_h5_paths]

        return PredSpecDB._AllSpecsIterable(self, all_h5s, ignore_keys)

    def close(self):
        if self.h5datasets is None:
            return
        for ds in self.h5datasets:
            if self.mode != 'r':
                ds.flush()
            ds.close()


class HDF5Dataset:
    """
    A dataset as a HDF5 file
    """
    def __init__(self, path, mode="r"):
        self.path = Path(path)
        self.h5_obj = h5py.File(path, mode=mode)
        self.attrs = self.h5_obj.attrs

    def __getitem__(self, idx):
        return self.h5_obj[idx]

    def __setitem__(self, key, value):
        self.h5_obj[key] = value

    def __contains__(self, idx):
        return idx in self.h5_obj

    def get_all_names(self):
        return self.h5_obj.keys()

    def read_str(self, name, encoding='utf-8') -> str:
        # Some metadata pipelines may store HDF5 dataset keys as numeric IDs.
        # Normalize to a string key before indexing h5py groups.
        if isinstance(name, bytes):
            name = name.decode(encoding)
        elif not isinstance(name, str):
            name = str(name)

        if '/' in name:  # has group
            groupname, name = name.rsplit('/', 1)
            grp = self.h5_obj[groupname]
        else:
            grp = self.h5_obj
        str_obj = grp[name][0]
        if type(str_obj) is not bytes:
            raise TypeError(f'Wrong type of {name}')
        return str_obj.decode(encoding)

    def write_str(self, name, data):
        if '/' in name:  # has group
            groupname, name = name.rsplit('/', 1)
            grp = self.h5_obj.require_group(groupname)
        else:
            grp = self.h5_obj
        dt = h5py.special_dtype(vlen=str)
        ds = grp.create_dataset(name, (1,), dtype=dt, compression="gzip")
        ds[0] = data

    def write_dict(self, dict):
        """dict entries: {filename: data}"""
        for filename, data in dict.items():
            self.write_str(filename, data)

    def write_list_of_tuples(self, list_of_tuples):
        """each tuple is (filename, data)"""
        for tup in list_of_tuples:
            if tup is None:
                continue
            self.write_str(tup[0], tup[1])

    def read_data(self, name) -> np.ndarray:
        """read a numpy array object"""
        return self.h5_obj[name][:]

    def write_data(self, name, data):
        """write a numpy array object"""
        self.h5_obj.create_dataset(name, data=data, dtype=data.dtype)

    def read_attr(self, name) -> dict:
        """read attribute of name as a dict"""

        return {k: v for k, v in self.h5_obj[name].attrs.items()}

    def update_attr(self, name, inp_dict):
        """write inp_dict to name's attribute"""
        cur_obj = self.h5_obj[name].attrs
        for k, v in inp_dict.items():
            cur_obj[k] = v

    def close(self):
        self.h5_obj.close()

    def flush(self):
        self.h5_obj.flush()


def setup_logger(save_dir, log_name="output.log", debug=False, custom_label=""):
    """Create output directory"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_file = save_dir / log_name

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(level)

    # Define basic logger
    logging.basicConfig(
        level=level,
        format=custom_label + "%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            file_handler,
        ],
        force=True
    )

    # log all uncaught exceptions
    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        logging.getLogger().error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = log_uncaught_exceptions

    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))



class ConsoleLogger(Logger):
    """Custom console logger class"""

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self):
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    @rank_zero_experiment
    def version(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        ## No need to log hparams
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):

        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status):
        pass


# Parsing


def parse_spectra(spectra_file: [str, list]) -> Tuple[dict, CompositeMassSpec]:
    """parse_spectra.

    Parses spectra in the SIRIUS format and returns

    Args:
        spectra_file (str or list): Name of spectra file to parse or lines of parsed spectra
    Return:
        Tuple[dict, List[Tuple[str, np.ndarray]]]: metadata and list of spectra
            tuples containing name and array
    """
    if type(spectra_file) is str or type(spectra_file) is PosixPath:
        lines = [i.strip() for i in open(spectra_file, "r").readlines()]
    elif type(spectra_file) is list:
        lines = [i.strip() for i in spectra_file]
    else:
        raise ValueError(f'type of variable spectra_file not understood, got {type(spectra_file)}')
    
    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        # Get spectra
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            # Check if spectra is empty
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                ms_obj = MassSpec(spectra_header, masses=peak_data[:, 0], intens=peak_data[:, 1])
                spectras.append(ms_obj)
        # Get meta data
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end

            metadata.update(entries)
        group_num += 1

    if type(spectra_file) is str:
        metadata["_FILE_PATH"] = spectra_file
        metadata["_FILE"] = Path(spectra_file).stem
    return metadata, CompositeMassSpec(spectras)


def spec_to_ms_str(
    spec: List[Tuple[str, np.ndarray]], essential_keys: dict, comments: dict = {}
) -> str:
    """spec_to_ms_str.

    Turn spec ars and info dicts into str for output file


    Args:
        spec (List[Tuple[str, np.ndarray]]): spec
        essential_keys (dict): essential_keys
        comments (dict): comments

    Returns:
        str:
    """

    def pair_rows(rows):
        return "\n".join([f"{i} {j}" for i, j in rows])

    header = "\n".join(f">{k} {v}" for k, v in essential_keys.items())
    comments = "\n".join(f"#{k} {v}" for k, v in essential_keys.items())
    spec_strs = [f">{name}\n{pair_rows(ar)}" for name, ar in spec]
    spec_str = "\n\n".join(spec_strs)
    output = f"{header}\n{comments}\n\n{spec_str}"
    return output


def parse_spectra_mgf(
    mgf_file: str, max_num = None
) -> List[Tuple[dict, np.ndarray]]:
    """parse_spectra_mgf.

    Parses spectra in the MGF file formate, with

    Args:
        mgf_file (str) : str
        max_num (Optional[int]): If set, only parse this many
    Return:
        List[Tuple[dict, List[Tuple[str, np.ndarray]]]]: metadata and list of spectra
            tuples containing name and array
    """

    key = lambda x: x.strip() == "BEGIN IONS"
    parsed_spectra = []
    with open(mgf_file, "r") as fp:

        for (is_header, group) in tqdm(groupby(fp, key)):

            if is_header:
                continue

            meta = dict()
            # Note: Sometimes we have multiple scans
            # This mgf has them collapsed
            cur_spectra = []
            group = list(group)
            for line in group:
                line = line.strip()
                if not line:
                    pass
                elif line == "END IONS" or line == "BEGIN IONS":
                    pass
                elif "=" in line:
                    k, v = [i.strip() for i in line.split("=", 1)]
                    meta[k] = v
                else:
                    mz, intens = line.split()
                    cur_spectra.append((float(mz), float(intens)))

            if len(cur_spectra) > 0:
                cur_spectra = np.vstack(cur_spectra)
                parsed_spectra.append((meta, cur_spectra))
            else:
                pass

            if max_num is not None and len(parsed_spectra) > max_num:
                break
        return parsed_spectra


def parse_cfm_out(spectra_file: str, max_merge=False) -> Tuple[dict, pd.DataFrame]:
    """parse_cfm_out.

    Args:
        out_file (str): out_file
        max_merge (bool): If true, merge across energies

    Returns:
        dict, pd.DataFrame:
    """
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]
    specs, keys = "\n".join(lines).split("\n\n")

    # Step 1: Process keys
    key_dict = {}
    for row in keys.split("\n"):
        row = row.strip()
        num, mass, smi = row.split()
        key_dict[num] = dict(mass=mass, smi=smi)

    lines = specs.split("\n")
    my_iterator = groupby(lines, lambda line: line.startswith("#"))
    meta_groups, spec_groups = [(i[0], list(i[1])) for i in my_iterator]

    # Step 2: extract meta
    meta_data = {}
    for i in meta_groups[1]:
        if "=" not in i:
            continue
        key, val = i.split("=", 1)
        meta_data[key[1:]] = val

    # Step 3: extract specs
    new_iter = groupby(spec_groups[1], lambda x: x.startswith("energy"))
    all_specs = {list(i)[0]: list(next(new_iter)[1]) for _, i in new_iter}

    # Combine all
    full_spec = []
    for spec_key, spec_vals in all_specs.items():
        key_to_amt = {}
        for spec_val in spec_vals:
            spec_val = spec_val.replace(")", "")
            mass, inten, rest = spec_val.split(" ", 2)
            num, amts = rest.split(" (")
            num_to_amt = dict(zip(num.split(), map(float, amts.split())))
            key_to_amt.update(num_to_amt)

        # Spec format is going to be
        for k, amt in key_to_amt.items():
            k_info = key_dict.get(k)
            mass = k_info["mass"]
            smi = k_info["smi"]
            chem_form = chem_utils.form_from_smi(smi)
            new_info = dict(
                smi=smi, mass=mass, form=chem_form, inten=amt, energy=spec_key
            )
            full_spec.append(new_info)
    full_spec = pd.DataFrame(full_spec)
    if max_merge:
        full_spec = full_spec.groupby("smi").max().reset_index()

    # Safe sub h
    sub_h = lambda x: chem_utils.formula_difference(x, "H") if "H" in x else x
    less_h = [sub_h(i) for i in full_spec["form"].values]
    full_spec["form_no_h"] = less_h
    full_spec["formula_mass"] = [chem_utils.formula_mass(i) for i in full_spec["form"].values]
    full_spec["ionization"] = "[M+H]+"
    return meta_data, full_spec


def merge_specs(specs_list, precision=4, merge_method='sum'):
    mz_to_inten_pair = {}
    new_tuples = []
    for spec in specs_list.values():
        for tup in spec:
            mz, inten = tup
            mz_ind = np.round(mz, precision)
            cur_pair = mz_to_inten_pair.get(mz_ind)
            if cur_pair is None:
                mz_to_inten_pair[mz_ind] = tup
                new_tuples.append(tup)
            else:
                if merge_method == 'sum':
                    cur_pair[1] += inten  # sum merging
                elif merge_method == 'max':
                    cur_pair[1] = max(cur_pair[1], inten)  # max merging
                else:
                    raise ValueError(f'Unknown merge_method {merge_method}')

    merged_spec = np.vstack(new_tuples)
    merged_spec = merged_spec[merged_spec[:, 1] > 0]
    if len(merged_spec) == 0:
        return
    merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()

    return {'nan': merged_spec}

def merge_intens(spec_dict):
    merged_intens = np.zeros_like(next(iter(spec_dict.values())))
    for spec in spec_dict.values():
        merged_intens += spec
    merged_intens = merged_intens / merged_intens.max()
    return {'nan': merged_intens}


def merge_mz(mzs, ppm=20):
    if not isinstance(mzs, float) and mzs is not None:
        if (max(mzs) - min(mzs)) / max(mzs) * 1e6 > ppm:
            raise ValueError(f'mass difference is larger than threshold ppm={ppm}. Got {mzs}')
        mz = np.mean(mzs).item()
        return mz
    else:  # is float
        return mzs

def process_spec_file(meta, tuples, precision=4, merge_specs=True, exclude_parent=False):
    """process_spec_file."""

    parent_mass = meta.get("parentmass", None)
    if parent_mass is None:
        print(f"missing parentmass for spec")
        parent_mass = 1000000

    parent_mass = float(parent_mass)

    # First norm spectra
    # First norm spectra
    from collections import defaultdict
    fused_tuples = defaultdict(list)
    for ce, x in tuples:
        if x.size > 0:
            fused_tuples[ce].append(x)

    if len(fused_tuples) == 0:
        return

    if merge_specs:
        mz_to_inten_pair = {}
        new_tuples = []
        for i_list in fused_tuples.values():
            for i in i_list: # iterate across grouped spectra 
                for tup in i: # iterate across spectrum
                    mz, inten = tup
                    mz_ind = np.round(mz, precision)
                    cur_pair = mz_to_inten_pair.get(mz_ind)
                    if cur_pair is None:
                        mz_to_inten_pair[mz_ind] = tup
                        new_tuples.append(tup)
                    elif inten > cur_pair[1]:
                        cur_pair[1] = inten # max merging
                    else:
                        pass

        merged_spec = np.vstack(new_tuples)
        if exclude_parent:
            merged_spec = merged_spec[merged_spec[:, 0] <= (parent_mass - 1)]
        else:
            merged_spec = merged_spec[merged_spec[:, 0] <= (parent_mass + 1)]
        merged_spec = merged_spec[merged_spec[:, 1] > 0]
        if len(merged_spec) == 0:
            return
        merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()

        # Sqrt intensities here
        merged_spec[:, 1] = np.sqrt(merged_spec[:, 1])
        return merged_spec
    else:
        new_specs = {}
        for k, v_list in fused_tuples.items():
            new_spec = np.vstack(v_list)
            new_spec = new_spec[new_spec[:, 0] <= (parent_mass + 1)]
            new_spec = new_spec[new_spec[:, 1] > 0]
            if len(new_spec) == 0:
                continue
            new_spec[:, 1] = new_spec[:, 1] / new_spec[:, 1].max()

            # Sqrt intensities here
            new_spec[:, 1] = np.sqrt(new_spec[:, 1])

            new_specs[k] = new_spec
        return new_specs


def bin_from_file(spec_file, num_bins, upper_limit) -> Tuple[dict, np.ndarray]:
    """bin_from_file.
    """
    return bin_from_str(open(spec_file, 'r').read(), num_bins, upper_limit)


def bin_from_str(spec_str, num_bins, upper_limit) -> Tuple[dict, np.ndarray]:
    """bin_from_str
    Args:
        spec_str:
        num_bins:
        upper_limit:

    Returns:
        Tuple[dict, np.ndarray]:
    """

    loaded_json = json.loads(spec_str)
    if loaded_json["output_tbl"] is None:
        return {}, None

    # Load with adduct involved
    mz = loaded_json["output_tbl"]["mono_mass"]
    inten = loaded_json["output_tbl"]["ms2_inten"]

    # Don't renorm; already procesed prior!
    spec_ar = np.vstack([mz, inten]).transpose(1, 0)
    binned = bin_spectra([spec_ar], num_bins, upper_limit)
    # normed = common.norm_spectrum(binned)
    avged = binned.mean(0)
    return {}, avged


def max_inten_spec(spec, max_num_inten: int = 60, inten_thresh: float = 0):
    """max_inten_spec.

    Args:
        spec: 2D spectra array
        max_num_inten: Max number of peaks
        inten_thresh: Min intensity to alloow in returned peak

    Return:
        Spec filtered down


    """
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]

    # Make sure to only take max of each formula
    # Sort by intensity and select top subpeaks
    new_sort_order = np.argsort(spec_intens)[::-1]
    if max_num_inten is not None:
        new_sort_order = new_sort_order[:max_num_inten]

    spec_masses = spec_masses[new_sort_order]
    spec_intens = spec_intens[new_sort_order]

    spec_mask = spec_intens > inten_thresh
    spec_masses = spec_masses[spec_mask]
    spec_intens = spec_intens[spec_mask]
    spec = np.vstack([spec_masses, spec_intens]).transpose(1, 0)
    return spec


def norm_spectrum(binned_spec: np.ndarray) -> np.ndarray:
    """norm_spectrum.

    Normalizes each spectral channel to have norm 1
    This change is made in place

    Args:
        binned_spec (np.ndarray) : Vector of spectras

    Return:
        np.ndarray where each channel has max(1)
    """

    spec_maxes = binned_spec.max(1)

    non_zero_max = spec_maxes > 0

    spec_maxes = spec_maxes[non_zero_max]
    binned_spec[non_zero_max] = binned_spec[non_zero_max] / spec_maxes.reshape(-1, 1)

    # Add in sqrt
    binned_spec = np.sqrt(binned_spec)

    return binned_spec


def bin_spectra(
    spectras: List[np.ndarray],
    num_bins: int = 15000,
    upper_limit: int = 1500,
    pool_fn: str = "max",
) -> np.ndarray:
    """bin_spectra.

    Args:
        spectras (List[np.ndarray]): Input list of spectra tuples
        num_bins (int): Number of discrete bins from [0, upper_limit)
        upper_limit (int): Max m/z to consider featurizing
        pool_fn (str): Pooling function to use for binning (max or add)

    Return:
        np.ndarray of shape [channels, num_bins]
    """
    binned_spec = []
    scale = (num_bins - 1) / upper_limit

    for spec in spectras:
        if isinstance(spec, MassSpec) or hasattr(spec, "spec"):
            spec = spec.spec

        mz = spec[:, 0]
        inten = spec[:, 1]

        bin_idx = np.floor(mz * scale).astype(np.int32) + 1
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        bin_idx = bin_idx[valid]
        inten = inten[valid]

        if pool_fn == "add":
            out = np.bincount(bin_idx, weights=inten, minlength=num_bins).astype(np.float32)
        elif pool_fn == "max":
            out = np.zeros(num_bins, dtype=np.float32)
            np.maximum.at(out, bin_idx, inten)
        else:
            raise NotImplementedError()

        binned_spec.append(out)

    return np.stack(binned_spec, axis=0)


def digitize_ar(
    ar: np.ndarray, num_bins: int = 15000, upper_limit: int = 1500
) -> np.ndarray:
    """digitize_ar.

    Args:
        ar (np.ndarray): ar
        num_bins (int): Num bins
        upper_limit (int): upper lim

    Return:
        np ndarray containing indices
    """
    bins = np.linspace(0, upper_limit, num=num_bins)
    return np.digitize(ar, bins=bins)


def bin_mass_results(
    mass,
    mass_bins=[
        (0, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, 600),
        (600, 700),
        (700, 2000),
    ],
):
    """bin_mass_results.

    Use to stratify results

    Args:
        mass:
        mass_bins:
    """
    for i, j in mass_bins:
        m_str = f"{i} - {j}"
        if mass <= j and mass > i:
            return m_str


def bin_peak_results(
    spec,
    peak_bins=[
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30),
        (30, 40),
        (40, 500),
    ],
    reduction = 'mean',  # mean / max / min
):
    """bin_peak_results.

    Use to stratify results
    """
    def count_peaks(sp):
        if isinstance(sp, tuple) and len(sp) == 2:
            return np.sum(np.asarray(sp[1]) > 0)

        sp = np.asarray(sp)
        if sp.size == 0:
            return 0
        if sp.ndim == 1:
            return np.sum(sp > 0)
        if sp.ndim == 2 and sp.shape[1] >= 2:
            return np.sum(sp[:, 1] > 0)
        return np.sum(sp > 0)

    num_peaks = [count_peaks(sp) for sp in spec.values()]
    reduction_func = getattr(np, reduction)
    num_peaks = reduction_func(num_peaks)
    for i, j in peak_bins:
        m_str = f"({i}, {j}]"
        if num_peaks <= j and num_peaks > i:
            return m_str

def bin_collision_results(
    collision_energy,
    bins=[
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (50, 100),
        (100, 1000),
    ],
):
    """bin_collision_results.

    Use to stratify results
    """
    collision_energy = float(collision_energy)
    if f'{collision_energy:.0f}' == 'nan':
        return "null"
    for i, j in bins:
        m_str = f"{i} - {j}"
        if collision_energy <= j and collision_energy > i:
            return m_str


def batches(it, chunk_size: int):
    """Consume an iterable in batches of size chunk_size""" ""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])


def batches_num_chunks(it, num_chunks: int):
    """Consume an iterable in batches of size chunk_size""" ""
    chunk_size = len(it) // num_chunks + 1
    return batches(it, chunk_size)


def build_mgf_str(
    meta_spec_list: List[Tuple[dict, np.ndarray]],
    sort_peaks=True,
    parent_mass_keys=["PEPMASS", "parentmass", "PRECURSOR_MZ"],
) -> str:
    """build_mgf_str.

    Args:
        meta_spec_list (List[Tuple[dict, List[Tuple[str, np.ndarray]]]]): meta_spec_list

    Returns:
        str:
    """
    entries = []
    for meta, spec_ar in meta_spec_list:
        str_rows = ["BEGIN IONS"]

        for k in ["TITLE", "SEQ"]:
            if k in meta:
                str_rows.append(f"{k}={meta[k]}")
                meta.pop(k)

        # Try to add precusor mass
        for i in parent_mass_keys:
            if i in meta:
                pep_mass = float(meta.get(i, -100))
                str_rows.append(f"PEPMASS={pep_mass}")
                break

        for k, v in meta.items():
            if k not in parent_mass_keys:
                str_rows.append(f"{k.upper().replace(' ', '_')}={v}")

        if sort_peaks:
            spec_ar = np.vstack([i for i in sorted(spec_ar, key=lambda x: x[0])])
        else:
            spec_ar = np.array(spec_ar)
        str_rows.append(f"Num peaks={len(spec_ar)}")
        str_rows.extend([f"{i:.4f} {j:.4f}" for i, j in spec_ar])
        str_rows.append("END IONS")

        str_out = "\n".join(str_rows)
        entries.append(str_out)

    full_out = "\n\n".join(entries)
    return full_out


def np_stack_padding(it, axis=0):

    def resize(row, size):
        new = np.array(row)
        new.resize(size)
        return new

    # find longest row length
    max_shape = [max(i) for i in zip(*[j.shape for j in it])]
    mat = np.stack([resize(row, max_shape) for row in it], axis=axis)
    return mat

def nce_to_ev(nce, precursor_mz):
    if type(nce) is str:
        output_type = 'str'
        if '.' in nce:  # decimal points
            decimal_num = len(nce.strip().split('.')[-1])
        else:
            decimal_num = 0
        nce = float(nce)
    elif type(nce) is int:
        output_type = 'int'
    elif type(nce) is float:
        output_type = 'float'
    else:
        raise TypeError(f'Input NCE type {type(nce)} is not understood')

    ev = nce * precursor_mz / 500

    if output_type == 'str':
        format_str = '{' + f':.{decimal_num}f' + '}'
        return format_str.format(ev)
    elif output_type == 'int':
        return int(round(ev))
    else:
        return float(ev)


def ev_to_nce(ev, precursor_mz):
    nce = ev * 500 / precursor_mz
    return nce


def md5(fname, chunk_size=4096):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def str_to_hash(inp_str, digest_size=16):
    return hashlib.blake2b(inp_str.encode("ascii"), digest_size=digest_size).hexdigest()


def rm_collision_str(key: str) -> str:
    """remove `_collision VALUE` from the string"""
    keys = key.split('_collision')
    if len(keys) == 2:
        return keys[0]
    elif len(keys) == 1:
        return key
    else:
        raise ValueError(f'Unrecognized key: {key}')


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False
