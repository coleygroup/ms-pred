import logging
import json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

import ms_pred.common as common


class BinnedDataset(Dataset):
    """BinnedDataset."""

    def __init__(
        self,
        df,
        data_dir,
        num_bins,
        num_workers=0,
        upper_limit=1500,
        form_dir_name: str = "subform_50",
        use_ray=False,
        **kwargs,
    ):
        """__init__ _summary_

        Args:
            df (_type_): _description_
            data_dir (_type_): _description_
            num_bins (_type_): _description_
            num_workers (int, optional): _description_. Defaults to 0.
            upper_limit (int, optional): _description_. Defaults to 1500.
            form_dir_name (str, optional): _description_. Defaults to "subform_50".
            use_ray (bool, optional): _description_. Defaults to False.
        """
        self.df = df
        self.num_bins = num_bins
        self.num_workers = num_workers
        self.upper_limit = upper_limit
        self.bins = np.linspace(0, self.upper_limit, self.num_bins)
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        # Read in all molecules
        self.smiles = self.df["smiles"].values
        if self.num_workers == 0:
            self.fps = [common.get_morgan_fp_smi_wt(i) for i in self.smiles]
        else:
            self.fps = common.chunked_parallel(
                self.smiles,
                common.get_morgan_fp_smi_wt,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=600,
                max_retries=3,
                use_ray=use_ray,
            )

        # Extract
        fps, weights = zip(*[(i, j) for i, j in self.fps])
        self.fps = np.vstack(fps)
        self.weights = np.array(weights)

        # Read in all specs
        self.spec_names = self.df["spec"].values

        spec_files = [
            (data_dir / "subformulae" / f"{form_dir_name}" / f"{spec_name}.json")
            for spec_name in self.spec_names
        ]
        process_spec_file = lambda x: common.bin_from_file(x, num_bins=num_bins, upper_limit=upper_limit)
        if self.num_workers == 0:
            spec_outputs = [process_spec_file(i) for i in spec_files]
        else:
            spec_outputs = common.chunked_parallel(
                spec_files,
                process_spec_file,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )
        self.metas, self.spec_ars = zip(*spec_outputs)
        mask = np.array([i is not None for i in self.spec_ars])
        logging.info(f"Could not find tables for {np.sum(~mask)} spec")

        self.spec_names = np.array(self.spec_names)[mask]
        self.metas = np.array(self.metas)[mask]
        self.spec_ars = np.array(self.spec_ars, dtype=object)[mask]
        self.spec_ars = np.vstack(self.spec_ars).astype(float)
        self.fps = np.array(self.fps)[mask]
        self.weights = np.array(self.weights)[mask]
        self.df = self.df[mask]

        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        name = self.spec_names[idx]
        meta = self.metas[idx]
        ar = self.spec_ars[idx]
        fp = self.fps[idx]
        adduct = self.adducts[idx]
        full_weight = self.weights[idx]
        outdict = {
            "name": name,
            "binned": ar,
            "full_weight": full_weight,
            "fp": fp,
            "adduct": adduct,
            "_meta": meta,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return BinnedDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["name"] for j in input_list]
        spec_ars = [j["binned"] for j in input_list]
        fp_ars = [j["fp"] for j in input_list]
        full_weight = [j["full_weight"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]

        # Now pad everything else to the max channel dim
        spectra_tensors = torch.stack([torch.tensor(spectra) for spectra in spec_ars])
        fp_tensors = torch.stack([torch.tensor(fp) for fp in fp_ars])
        full_weight = torch.FloatTensor(full_weight)
        adducts = torch.FloatTensor(adducts)
        return_dict = {
            "spectra": spectra_tensors,
            "fps": fp_tensors,
            "names": names,
            "adducts": adducts,
            "full_weight": full_weight,
        }
        return return_dict


class MolDataset(Dataset):
    """MolDataset."""

    def __init__(self, df, num_workers: int = 0, **kwargs):

        self.df = df
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)
        self.num_workers = num_workers

        # Read in all molecules
        self.smiles = self.df["smiles"].values
        self.spec_names = ["" for i in self.smiles]
        if "spec" in list(self.df.keys()):
            self.spec_names = self.df["spec"].values

        if self.num_workers == 0:
            self.fps = [common.get_morgan_fp_smi_wt(i) for i in self.smiles]
        else:
            self.fps = common.chunked_parallel(
                self.smiles,
                common.get_morgan_fp_smi_wt,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
        # Extract
        fps, weights = zip(*[(i, j) for i, j in self.fps if i is not None])
        self.fps = np.vstack(fps)
        self.weights = np.vstack(weights).squeeze()
        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        fp = self.fps[idx]
        full_weight = self.weights[idx]
        spec_name = self.spec_names[idx]
        adduct = self.adducts[idx]

        outdict = {
            "smi": smi,
            "fp": fp,
            "adduct": adduct,
            "full_weight": full_weight,
            "spec_name": spec_name,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return MolDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["smi"] for j in input_list]
        spec_names = [j["spec_name"] for j in input_list]
        fp_ars = [j["fp"] for j in input_list]
        full_weight = [j["full_weight"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]

        adducts = torch.FloatTensor(adducts)

        # Now pad everything else to the max channel dim
        fp_tensors = torch.stack([torch.tensor(fp) for fp in fp_ars])
        full_weight = torch.FloatTensor(full_weight)
        return_dict = {
            "fps": fp_tensors,
            "spec_names": spec_names,
            "names": names,
            "full_weight": full_weight,
            "adducts": adducts,
        }
        return return_dict
