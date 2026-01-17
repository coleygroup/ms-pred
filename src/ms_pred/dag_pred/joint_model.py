""" joint_model. """
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import torch

import ms_pred.common as common
import ms_pred.dag_pred.gen_model as gen_model
import ms_pred.dag_pred.inten_model as inten_model
import ms_pred.dag_pred.dag_data as dag_data
from ms_pred import nn_utils
from ms_pred.magma.fragmentation import FRAGMENT_ENGINE_PARAMS
MAX_BROKEN_BONDS = FRAGMENT_ENGINE_PARAMS['max_broken_bonds']

class JointModel(pl.LightningModule):
    def __init__(
        self,
        gen_model_obj: gen_model.FragGNN,
        inten_model_obj: inten_model.IntenGNN,
    ):
        """__init__.

        Args:
            gen_model_obj (gen_model.FragGNN): gen_model_obj
            inten_model_obj (inten_model.IntenGNN): inten_model_obj
        """

        super().__init__()
        self.gen_model_obj = gen_model_obj
        self.inten_model_obj = inten_model_obj
        self.inten_collate_fn = dag_data.IntenPredDataset.get_collate_fn()

        root_enc_gen = self.gen_model_obj.root_encode
        pe_embed_gen = self.gen_model_obj.pe_embed_k
        add_hs_gen = self.gen_model_obj.add_hs
        embed_elem_group_gen = self.gen_model_obj.embed_elem_group

        root_enc_inten = self.inten_model_obj.root_encode
        pe_embed_inten = self.inten_model_obj.pe_embed_k
        add_hs_inten = self.inten_model_obj.add_hs
        embed_elem_group_inten = self.inten_model_obj.embed_elem_group

        self.gen_tp = dag_data.TreeProcessor(
            root_encode=root_enc_gen, pe_embed_k=pe_embed_gen, add_hs=add_hs_gen, embed_elem_group=embed_elem_group_gen,
        )

        self.inten_tp = dag_data.TreeProcessor(
            root_encode=root_enc_inten, pe_embed_k=pe_embed_inten, add_hs=add_hs_inten, embed_elem_group=embed_elem_group_inten,
        )

    @classmethod
    def from_checkpoints(cls, gen_checkpoint, inten_checkpoint):
        """from_checkpoints.

        Args:
            gen_checkpoint
            inten_checkpoint
        """

        gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_checkpoint)
        inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_checkpoint)
        return cls(gen_model_obj, inten_model_obj)

    def predict_mol(
        self,
        smi: str,
        collision_eng: float,
        precursor_mz: float,
        adduct: str,
        threshold: float,
        device: str,
        max_nodes: int,
        binned_out: bool = False,
        adduct_shift: bool = False,
        canonical_root_smi: bool = False,
    ) -> dict:
        """predict_mol.

        Args:
            smi (str): smi
            adduct
            threshold (float): threshold
            device (str): device
            max_nodes (int): max_nodes
            binned_out
        """
        self.eval()
        self.freeze()

        # Run tree gen model
        # Defines exact tree
        root_smi = smi
        if type(root_smi) is str:
            batched_input = False
            root_smi = [root_smi]
            collision_eng = [collision_eng]
            precursor_mz = [precursor_mz]
            adduct = [adduct]
        else:
            batched_input = True
        batch_size = len(root_smi)
        if not canonical_root_smi:
            root_smi = [common.smiles_from_inchi(common.inchi_from_smiles(_)) for _ in root_smi] # canonical smiles

        frag_preds, root_reprs = self.gen_model_obj.predict_mol(
            root_smi=root_smi,
            collision_eng=collision_eng,
            precursor_mz=precursor_mz,
            adduct=adduct,
            threshold=threshold,
            device=device,
            max_nodes=max_nodes,
            canonical_root_smi=True,
        )

        num_frags = frag_preds['nfrags']
        max_nfrags = torch.max(num_frags)
        ind_maps = torch.arange(batch_size, device=device).repeat_interleave(num_frags)

        rep_roots, _, __ = nn_utils.slice_batched_graph(root_reprs, ind_maps)
        frag_mask = nn_utils.pack_padded_tensor(frag_preds['frags'], num_frags)
        frag_graphs = nn_utils.batched_mask_subgraph(
            rep_roots,
            nn_utils.pack_padded_tensor(frag_mask, frag_preds['natoms'].repeat_interleave(num_frags)),
            frag_mask.sum(dim=-1),
        )

        broken_bonds = frag_preds['brokens']
        nhs = frag_preds['frag_form_vecs'][:, :, common.element_to_ind['H']]
        max_add_hs = broken_bonds[:, :max_nfrags]
        max_remove_hs = torch.min(broken_bonds, nhs)[:, :max_nfrags]

        mass_offset = torch.tensor(
            [(common.ion2mass[add], -common.ELECTRON_MASS) if common.is_positive_adduct(add[-1]) else
             (common.ion2mass[add],  common.ELECTRON_MASS)
             for add in adduct], device=device)[:, :, None] + \
            (torch.arange(MAX_BROKEN_BONDS * 2 + 1, device=device)[None, None, :] - MAX_BROKEN_BONDS) * \
                      common.CHEM_MASSES[common.element_to_ind['H']].item()
        masses = frag_preds['masses_no_adduct'][:, :max_nfrags, None, None] + mass_offset[:, None, :, :]

        to_tensor = lambda x: torch.tensor(x, device=device, dtype=torch.float) if x is not None else x
        adducts = to_tensor([common.ion2onehot_pos[a] for a in adduct])
        collision_engs = to_tensor(collision_eng)
        precursor_mzs = to_tensor(precursor_mz)
        root_forms = frag_preds['root_form_vec']
        frag_forms = frag_preds['frag_form_vecs'][:, :max_nfrags]

        # IDs to use to recapitulate
        inten_preds = self.inten_model_obj.predict(
            graphs=frag_graphs,
            root_reprs=root_reprs,
            ind_maps=ind_maps,
            num_frags=num_frags,
            max_breaks=broken_bonds,
            max_add_hs=max_add_hs,
            max_remove_hs=max_remove_hs,
            masses=masses,
            root_forms=root_forms,
            frag_forms=frag_forms,
            binned_out=binned_out,
            adducts=adducts,
            collision_engs=collision_engs,
            precursor_mzs=precursor_mzs,
        )

        if binned_out:
            out = inten_preds
        else:
            out = {"spec": [], "frag": []}
            num_shifts = len(masses[0, 0, :, :].reshape(-1))  # number of shifts,
                                                              # (1 + h_shift * 2) * 2 if include_unshifted_mz==True
            if not self.inten_model_obj.include_unshifted_mz:
                masses = masses[:, :, :1, :].contiguous()  # only keep m/z with adduct shift

            for i, (inten_pred, mass, nfrags) in \
                    enumerate(zip(inten_preds["spec"], masses, num_frags)):
                out_mass = mass[:nfrags].reshape(-1)
                out_inten = inten_pred.reshape(-1)
                natoms = frag_preds['natoms'][i]
                out_frag = frag_preds['frags'][i, :nfrags, :natoms].repeat_interleave(num_shifts, dim=0)

                # add to output dict
                out["spec"].append(torch.stack((out_mass, out_inten), dim=1))
                out["frag"].append(out_frag)

        if batched_input:
            return out
        else:
            return {k: v[0] for k, v in out.items()}
