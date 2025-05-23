""" joint_model. """
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import torch

import ms_pred.common as common
import ms_pred.dag_pred.gen_model as gen_model
import ms_pred.dag_pred.inten_model as inten_model
import ms_pred.dag_pred.dag_data as dag_data


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

        frag_tree = self.gen_model_obj.predict_mol(
            root_smi=root_smi,
            collision_eng=collision_eng,
            precursor_mz=precursor_mz,
            adduct=adduct,
            threshold=threshold,
            device=device,
            max_nodes=max_nodes,
            canonical_root_smi=True,
        )
        processed_trees = []
        out_trees = []
        for r_smi, colli_eng, adct, p_mz, tree in zip(root_smi, collision_eng, adduct, precursor_mz, frag_tree):
            tree = {
                "root_canonical_smiles": r_smi,
                "name": "",
                "collision_energy": colli_eng,
                "frags": tree,
                "adduct": adct
            }

            processed_tree = self.inten_tp.process_tree_inten_pred(tree)

            # Save for output wrangle
            out_tree = processed_tree["tree"]
            processed_tree = processed_tree["dgl_tree"]

            processed_tree["adduct"] = common.ion2onehot_pos[adct]
            processed_tree["name"] = ""
            processed_tree["precursor"] = p_mz
            processed_trees.append(processed_tree)
            out_trees.append(out_tree)
        batch = self.inten_collate_fn(processed_trees)
        inten_frag_ids = batch["inten_frag_ids"]

        safe_device = lambda x: x.to(device) if x is not None else x

        frag_graphs = safe_device(batch["frag_graphs"])
        root_reprs = safe_device(batch["root_reprs"])
        ind_maps = safe_device(batch["inds"])
        num_frags = safe_device(batch["num_frags"])
        broken_bonds = safe_device(batch["broken_bonds"])
        max_remove_hs = safe_device(batch["max_remove_hs"])
        max_add_hs = safe_device(batch["max_add_hs"])
        masses = safe_device(batch["masses"])  # b x n x 2 x 13

        assert adduct_shift, 'adduct shift must be enforced'

        adducts = safe_device(batch["adducts"]).to(device)
        collision_engs = safe_device(batch["collision_engs"]).to(device)
        precursor_mzs = safe_device(batch["precursor_mzs"]).to(device)
        root_forms = safe_device(batch["root_form_vecs"])
        frag_forms = safe_device(batch["frag_form_vecs"])

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
            for inten_pred, mass, inten_frag_id, out_tree, n in \
                    zip(inten_preds["spec"], masses.cpu().numpy(), inten_frag_ids, out_trees, num_frags.cpu().numpy()):
                out_mass = mass[:n].reshape(-1)
                out_inten = inten_pred.reshape(-1)
                out_frag = []
                for id in inten_frag_id:
                    out_frag += [out_tree["frags"][id]["frag"]] * num_shifts

                # merge duplicated mass + frag combinations
                seen_mass = []
                seen_frag = []
                seen_inten = []
                for i in range(len(out_mass)):
                    same_mass_ids = np.nonzero(np.abs(out_mass[i] - np.array(seen_mass)) < 0.0001)
                    same_frag_ids = np.nonzero(np.array(seen_frag) == out_frag[i])
                    same_mass_frag_ids = np.intersect1d(same_mass_ids, same_frag_ids)
                    if len(same_mass_frag_ids) > 0:  # this mass + frag has been recorded
                        assert len(same_mass_frag_ids) == 1  # the entry should be unique
                        seen_inten[same_mass_frag_ids.item()] += out_inten[i]  # merge intensity
                    else:  # this mass + frag is new
                        seen_mass.append(out_mass[i])
                        seen_frag.append(out_frag[i])
                        seen_inten.append(out_inten[i])

                # add to output dict
                out["spec"].append(np.stack((np.array(seen_mass), np.array(seen_inten)), axis=1))
                out["frag"].append(seen_frag)

        if batched_input:
            return out
        else:
            return {k: v[0] for k, v in out.items()}
