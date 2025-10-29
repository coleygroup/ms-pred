""" utils """
import sys
import copy
import logging
from pathlib import Path, PosixPath
import json
from itertools import groupby, islice
from typing import Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import hashlib
import torch
import math

import ms_pred.common.chem_utils as chem_utils
from ms_pred import nn_utils

try:
    from pytorch_lightning.loggers import LightningLoggerBase as Logger
    from pytorch_lightning.loggers.base import rank_zero_experiment
except ImportError: # pytorch_lightning >= 1.9
    from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

NIST_COLLISION_ENERGY_MEAN = 40.260853377886264
NIST_COLLISION_ENERGY_STD = 31.604227557486197


def get_data_dir(dataset_name: str) -> Path:
    return Path("data/spec_datasets") / dataset_name


class MassSpec:
    """
    Data structure for a predicted MS/MS spectrum
    """
    def __init__(self, collision_energy, root_canonical_smiles=None, adduct=None, remark=None,
                 probs=None, brokens=None, masses=None, masses_no_adduct=None,
                 frag_form_vecs=None, frags=None, intens=None, pulled_atoms=None,
                 **kwargs):
        self.collision_energy = collision_energy
        self.root_canonical_smiles = root_canonical_smiles
        self.adduct = adduct
        self.remark = remark
        def safe_assign(x):
            if isinstance(x, np.ndarray):
                return x
            elif isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            else:
                return None
        self.probs = safe_assign(probs)
        self.brokens = safe_assign(brokens)
        self.masses = safe_assign(masses)
        self.masses_no_adduct = safe_assign(masses_no_adduct)
        self.frag_form_vecs = safe_assign(frag_form_vecs)
        self.frags = safe_assign(frags)
        self.intens = safe_assign(intens)
        self.pulled_atoms = safe_assign(pulled_atoms)
        self.meta = kwargs

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
    def int_frags(self):
        all_frags = []
        for bin_frag in self.frags:
            frag = 0
            for i, b in enumerate(bin_frag):
                if b:
                    frag += 2 ** i
            all_frags.append(frag)
        return all_frags

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
    def has_pulled_atoms(self):
        return self.pulled_atoms is not None

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

    def __repr__(self):
        _repr = f'MassSpec ('
        _repr += ', '.join([f'{k}={v}' for k, v in self.info.items()])
        for key in ('probs', 'brokens', 'masses', 'masses_no_adduct', 'frag_form_vecs', 'frags', 'intens'):
            obj = getattr(self, key)
            if obj is not None:
                _repr += f', {key}={obj.shape}'
        _repr += ')'
        return _repr

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise AttributeError(f'MassSpec object has no attribute {item}')

    def __contains__(self, item):
        return hasattr(self, item)


class PredSpecDB:
    """
    Data structure for predicted spectrum database
    """
    def __init__(self, h5_path, mode="r", num_h5s=1,
                 has_probs=True, has_brokens=True, has_masses=False, has_masses_no_adduct=True, has_frag_form_vecs=True,
                 has_frags=True, has_intens=False, has_pulled_atoms=False,
                 h5_persistent=None):
        h5_path = Path(h5_path)
        if num_h5s > 1:
            self.all_h5_paths = [h5_path.parent / (h5_path.stem + f'_chunk_{i}' + h5_path.suffix) for i in range(num_h5s)]
        else:
            self.all_h5_paths = [h5_path]

        self.mode = mode
        self.h5_persistent = h5_persistent
        h5_dataset_0 = HDF5Dataset(self.all_h5_paths[0], self.mode)
        if self.mode == 'w':  # create new file
            self.has_probs   = has_probs
            self.has_brokens = has_brokens
            self.has_masses  = has_masses
            self.has_masses_no_adduct = has_masses_no_adduct
            self.has_frag_form_vecs = has_frag_form_vecs
            self.has_frags   = has_frags
            self.has_intens  = has_intens
            self.has_pulled_atoms = has_pulled_atoms
            self.root_key_dict = {
                "probs": self.has_probs,
                "masses": self.has_masses,
                "masses_no_adduct": self.has_masses_no_adduct,
                "intens": self.has_intens,
                "brokens": self.has_brokens,
                "frag_form_vecs": self.has_frag_form_vecs,
                "frags": self.has_frags,
                "pulled_atoms": self.has_pulled_atoms,
            }
            h5_dataset_0.update_attr('.', self.root_key_dict)
            if self.h5_persistent is None:
                self.h5_persistent = True  # default persistent H5 objects for write mode
        else:
            self.root_key_dict = h5_dataset_0.read_attr('.')
            safe_root_key_get = lambda x: self.root_key_dict[x] if x in self.root_key_dict else None
            self.has_probs            = safe_root_key_get('probs')
            self.has_masses           = safe_root_key_get('masses')
            self.has_masses_no_adduct = safe_root_key_get('masses_no_adduct')
            self.has_intens           = safe_root_key_get('intens')
            self.has_brokens          = safe_root_key_get('brokens')
            self.has_frag_form_vecs   = safe_root_key_get('frag_form_vecs')
            self.has_frags            = safe_root_key_get('frags')
            self.has_pulled_atoms     = safe_root_key_get('pulled_atoms')
            if self.h5_persistent is None:
                self.h5_persistent = False  # default non-persistent H5 objects for read mode (for parallel read)

        if self.h5_persistent:
            self.h5datasets = [h5_dataset_0] + [HDF5Dataset(p, self.mode) for p in self.all_h5_paths[1:]]
        else:
            self.h5datasets = None

    def write(self, name, spec: MassSpec):
        """write one spectrum"""
        key_dict = {
            "probs": spec.has_probs,
            "masses": spec.has_masses,
            "masses_no_adduct": spec.has_masses_no_adduct,
            "intens": spec.has_intens,
            "brokens": spec.has_brokens,
            "frag_form_vecs": spec.has_frag_form_vecs,
            "frags": spec.has_frags,
            "pulled_atoms": spec.has_pulled_atoms,
        }
        h5_dataset = self._get_h5_dataset(name)
        full_name  = self._get_full_name(name, spec.collision_energy, spec.remark)

        float_arrs = []
        if spec.has_probs: float_arrs.append(spec.probs.astype(np.float32))
        if spec.has_masses: float_arrs.append(spec.masses.astype(np.float32))
        if spec.has_masses_no_adduct: float_arrs.append(spec.masses_no_adduct.astype(np.float32))
        if spec.has_intens: float_arrs.append(spec.intens.astype(np.float32))
        if len(float_arrs) > 0:
            h5_dataset.write_data(full_name + '/f', np.stack(float_arrs, axis=1))

        uint_arrs = []
        if spec.has_brokens: uint_arrs.append(spec.brokens.astype(np.uint8)[:, None])
        if spec.has_frag_form_vecs: uint_arrs.append(spec.frag_form_vecs.astype(np.uint8))
        if spec.has_frags: uint_arrs.append(nn_utils.encode_bin_to_uint8(spec.frags).astype(np.uint8))
        if spec.has_pulled_atoms: uint_arrs.append(nn_utils.encode_bin_to_uint8(spec.pulled_atoms).astype(np.uint8))
        if len(uint_arrs) > 0:
            h5_dataset.write_data(full_name + '/u', np.concatenate(uint_arrs, axis=1))

        if not all([self.root_key_dict[k] == v for k, v in key_dict.items()]):
            # h5 group will have a different attribute if any attribute is different from root
            key_dict["override"] = True
            h5_dataset.update_attr(full_name, key_dict)
        else:
            h5_dataset.update_attr(full_name, {"override": False})
        if spec.has_frags:
            h5_dataset.update_attr(full_name, {"frag_bits": spec.frags.shape[-1]})
        h5_dataset.update_attr(full_name, spec.info)

    def read(self, name, collision_energy=None, remark=None):
        """read a specific spectrum"""
        h5_dataset = self._get_h5_dataset(name)
        full_name  = self._get_full_name(name, collision_energy, remark)

        key_dict = h5_dataset.read_attr(full_name)
        if not key_dict["override"]:
            key_dict.update(self.root_key_dict)

        del key_dict["override"]

        in_key_dict = lambda x: x in key_dict and key_dict[x]

        spec_dict = {}
        spec_dict.update(key_dict)
        fdata = h5_dataset.read_data(full_name + '/f')
        cur_idx = 0
        for key in ("probs", "masses", "masses_no_adduct", "intens"):
            if in_key_dict(key):
                spec_dict[key] = fdata[:, cur_idx]
                cur_idx += 1

        udata = h5_dataset.read_data(full_name + '/u')
        cur_idx = 0
        if in_key_dict("brokens"):
            spec_dict["brokens"] = udata[:, cur_idx]
            cur_idx += 1
        if in_key_dict("frag_form_vecs"):
            spec_dict["frag_form_vecs"] = udata[:, cur_idx:cur_idx+chem_utils.ELEMENT_DIM]
            cur_idx += chem_utils.ELEMENT_DIM
        if in_key_dict("frags"):
            frag_u8_len = math.ceil(key_dict["frag_bits"] / 8)
            spec_dict["frags"] = nn_utils.decode_bin_from_uint8(udata[:, cur_idx:cur_idx+frag_u8_len], key_dict["frag_bits"])
            cur_idx += frag_u8_len
        if in_key_dict("pulled_atoms"):
            assert "frag_bits" in key_dict
            spec_dict["pulled_atoms"] = nn_utils.decode_bin_from_uint8(udata[:, cur_idx:], key_dict["frag_bits"])

        return MassSpec(**spec_dict)

    def read_all(self, name):
        """read all entries with the same name (all remarks, all collision energies)"""
        colli_engs, remarks = self.get_entries(name)
        all_spec_dict = {}
        for c, r in zip(colli_engs, remarks):
            spec_obj = self.read(name, c, r)
            c_key = f'collision {c}'
            if r is None:
                all_spec_dict[c_key] = spec_obj
            else:
                if r not in all_spec_dict:
                    all_spec_dict[r] = {}
                all_spec_dict[r][c_key] = spec_obj
        return all_spec_dict

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
        cur_i = self._get_chunk_index(name)
        if self.h5_persistent:
            h5_dataset = self.h5datasets[cur_i]
        else:
            h5_dataset = HDF5Dataset(self.all_h5_paths[cur_i], self.mode)
        return h5_dataset

    def get_all_names(self):
        if self.h5_persistent:
            all_h5s = self.h5datasets
        else:
            all_h5s = [HDF5Dataset(p, self.mode) for p in self.all_h5_paths]
        all_names = [h5.get_all_names() for h5 in all_h5s]
        all_names = [i for j in all_names for i in j]
        return all_names

    def get_entries(self, name, collision_energy=None):
        """return collision energies and remarks under each name"""
        def _enumerate_path(obj):  # recursive call
            if obj.attrs and "override" in obj.attrs:  # reached data group
                return None
            else:
                paths = []
                for key in obj.keys():
                    next_paths = _enumerate_path(obj[key])
                    if next_paths is None:
                        paths.append(key)
                    else:
                        for next_p in next_paths:
                            paths.append(key + '/' + next_p)
                return paths
        h5_dataset = self._get_h5_dataset(name)
        if not name in h5_dataset.h5_obj:  # no entry
            return [], []
        root_obj = h5_dataset.h5_obj[name]
        all_paths = _enumerate_path(root_obj)
        colli_engs, remarks = [], []
        for path in all_paths:
            keys = path.split('/')
            if len(keys) == 1:  # collision energy
                remarks.append(None)
                get_ce = chem_utils.get_collision_energy(keys[0])
                if collision_energy is None or float(collision_energy) == float(get_ce):
                    colli_engs.append(get_ce)
            elif len(keys) == 2:  # collision energy + remark
                get_ce = chem_utils.get_collision_energy(keys[1])
                if collision_energy is None or float(collision_energy) == float(get_ce):
                    remarks.append(keys[0])
                    colli_engs.append(get_ce)
            else:
                raise ValueError(f'HDF5 path is not accepted: {path}')
        return colli_engs, remarks

    def close(self):
        for ds in self.h5datasets:
            ds.close()


class HDF5Dataset:
    """
    A dataset as a HDF5 file
    """
    def __init__(self, path, mode="r"):
        self.path = path
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
    )

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


def parse_spectra(spectra_file: [str, list]) -> Tuple[dict, List[Tuple[str, np.ndarray]]]:
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
                # Add new tuple
                spectras.append((spectra_header, peak_data))
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
    return metadata, spectras


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
) -> List[Tuple[dict, List[Tuple[str, np.ndarray]]]]:
    """parse_spectr_mgf.

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
            spectra = []
            # Note: Sometimes we have multiple scans
            # This mgf has them collapsed
            cur_spectra_name = "spec"
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
                spectra.append((cur_spectra_name, cur_spectra))
                parsed_spectra.append((meta, spectra))
            else:
                pass
                # print("no spectra found for group: ", "".join(group))

            if max_num is not None and len(parsed_spectra) > max_num:
                # print("Breaking")
                break
        return parsed_spectra


def parse_spectra_mgf(
    mgf_file: str, max_num = None
) -> List[Tuple[dict, List[Tuple[str, np.ndarray]]]]:
    """parse_spectr_mgf.

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
            spectra = []
            # Note: Sometimes we have multiple scans
            # This mgf has them collapsed
            cur_spectra_name = "spec"
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
                spectra.append((cur_spectra_name, cur_spectra))
                parsed_spectra.append((meta, spectra))
            else:
                pass
                # print("no spectra found for group: ", "".join(group))

            if max_num is not None and len(parsed_spectra) > max_num:
                # print("Breaking")
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
    fused_tuples = {ce: x for ce, x in tuples if x.size > 0}

    if len(fused_tuples) == 0:
        return

    if merge_specs:
        mz_to_inten_pair = {}
        new_tuples = []
        for i in fused_tuples.values():
            for tup in i:
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
        for k, v in fused_tuples.items():
            new_spec = np.vstack(v)
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
    if pool_fn == "add":
        pool = lambda x, y: x + y
    elif pool_fn == "max":
        pool = lambda x, y: max(x, y)
    else:
        raise NotImplementedError()

    bins = np.linspace(0, upper_limit, num=num_bins)
    binned_spec = np.zeros((len(spectras), len(bins)))
    for spec_index, spec in enumerate(spectras):

        # Convert to digitized spectra
        digitized_mz = np.digitize(spec[:, 0], bins=bins)

        # Remove all spectral peaks out of range
        in_range = digitized_mz < len(bins)
        digitized_mz, spec = digitized_mz[in_range], spec[in_range, :]

        # Add the current peaks to the spectra
        # Use a loop rather than vectorize because certain bins have conflicts
        # based upon resolution
        for bin_index, spec_val in zip(digitized_mz, spec[:, 1]):
            cur_val = binned_spec[spec_index, bin_index]
            binned_spec[spec_index, bin_index] = pool(spec_val, cur_val)

    return binned_spec


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
    binned_spec = True,
    reduction = 'mean',  # mean / max / min
):
    """bin_peak_results.

    Use to stratify results
    """
    if binned_spec:
        num_peaks = [np.sum(sp > 0) for sp in spec.values()]
    else:
        num_peaks = [np.sum(sp[:, 1] > 0)  for sp in spec.values()]
    reduction_func = eval('np.' + reduction)
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
    meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]],
    merge_charges=True,
    parent_mass_keys=["PEPMASS", "parentmass", "PRECURSOR_MZ"],
) -> str:
    """build_mgf_str.

    Args:
        meta_spec_list (List[Tuple[dict, List[Tuple[str, np.ndarray]]]]): meta_spec_list

    Returns:
        str:
    """
    entries = []
    for meta, spec in tqdm(meta_spec_list):
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

        if merge_charges:
            spec_ar = np.vstack([i[1] for i in spec])
            spec_ar = np.vstack([i for i in sorted(spec_ar, key=lambda x: x[0])])
        else:
            raise NotImplementedError()
        str_rows.append(f"Num peaks={len(spec_ar)}")
        str_rows.extend([f"{i} {j}" for i, j in spec_ar])
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
