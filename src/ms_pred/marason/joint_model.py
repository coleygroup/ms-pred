""" joint_model. """
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import torch

import ms_pred.common as common
import ms_pred.marason.gen_model as gen_model
import ms_pred.marason.inten_model as inten_model
import ms_pred.marason.dag_data as dag_data


class JointModel(pl.LightningModule):
    def __init__(
        self,
        gen_model_obj: gen_model.FragGNN,
        inten_model_obj: inten_model.IntenGNN,
        db=None, 
        ref_engs=None, 
        ref_specs=None,
        add_ref=False,
        max_ref_count = 10
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
        self.max_ref_count = max_ref_count
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
        self.db=db
        self.ref_engs=ref_engs
        self.ref_specs=ref_specs
        self.add_ref=add_ref

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
        instrument: str = None,
        binned_out: bool = False,
        adduct_shift: bool = False,
        min_distances = None,
        valid_ref_counts = None,
        closests = None,
        canonical_root_smi: bool = False,
        name: str = None,
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
            instrument = [instrument]
        else:
            batched_input = True
        batch_size = len(root_smi)
        if not canonical_root_smi:
            # root_smi = [common.smiles_from_inchi(common.inchi_from_smiles(_)) for _ in root_smi] # canonical smiles
            root_smi = [common.rm_stereo(smi) for smi in root_smi]
            root_smi = [common.smiles_from_inchi(common.inchi_from_smiles(_)) for _ in root_smi] # canonical smiles
            # use to filter
            valid_mask = [r_smi != None for r_smi in root_smi]
            # use np.arrays to reprocess:
            if sum(valid_mask) < batch_size:
                print("['joint_model.py']: Some SMILES could not be canonicalized via inchi: ", [(smi[i], name[i]) for i in range(batch_size) if not valid_mask[i]])
                print("['joint_model.py']:", )
                root_smi = tuple(np.array(root_smi)[valid_mask].tolist())
                collision_eng = tuple(np.array(collision_eng)[valid_mask].tolist())
                precursor_mz = tuple(np.array(precursor_mz)[valid_mask].tolist())
                adduct = tuple(np.array(adduct)[valid_mask].tolist())
                instrument = tuple(np.array(instrument)[valid_mask].tolist())

        frag_tree = self.gen_model_obj.predict_mol(
            root_smi=root_smi,
            collision_eng=collision_eng,
            precursor_mz=precursor_mz,
            adduct=adduct,
            instrument=instrument,
            threshold=threshold,
            device=device,
            max_nodes=max_nodes,
            canonical_root_smi=True,
        )
        processed_trees = []
        out_trees = []
        for (r_smi, colli_eng, adct, instrument, p_mz, tree, min_distance, valid_ref_count, closest) in zip(root_smi, collision_eng, adduct, instrument, precursor_mz, frag_tree, min_distances, valid_ref_counts, closests):
            tree = {"root_canonical_smiles": r_smi, "name": "", "collision_energy": colli_eng, "frags": tree, "adduct": adct, "instrument": instrument}

            processed_tree = self.inten_tp.process_tree_inten_pred(tree)

            # Save for output wrangle
            out_tree = processed_tree["tree"]
            processed_tree = processed_tree["dgl_tree"]
            processed_tree['instrument'] = common.instrument2onehot_pos[instrument]
            processed_tree["adduct"] = common.ion2onehot_pos[adct]
            processed_tree["name"] = ""
            processed_tree["precursor"] = p_mz
            if self.add_ref:
                valid_ref_count = min(valid_ref_count, self.max_ref_count)
                data = self.db[closest[0]]
                spec_indices = closest[:valid_ref_count]
                ref_engs = self.ref_engs[spec_indices]
                ref_specs = self.ref_specs[spec_indices, :].toarray()
                processed_tree["distance"] = min_distance
                processed_tree["ref"] = data
                processed_tree["ref_count"] = valid_ref_count
                processed_tree["ref_collision_engs"] = ref_engs
                processed_tree["ref_inten_targs"] = ref_specs
            else:
                processed_tree["distance"] = processed_tree["ref"] = processed_tree["ref_count"] = processed_tree["ref_collision_engs"] = processed_tree["ref_inten_targs"] = None
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
        masses = safe_device(batch["masses"])

        assert adduct_shift, 'adduct shift must be enforced'

        adducts = safe_device(batch["adducts"])
        if self.inten_model_obj.embed_instrument:
            instruments = safe_device(batch["instruments"]).to(device)
            closest_instruments=instruments
        collision_engs = safe_device(batch["collision_engs"])
        root_forms = safe_device(batch["root_form_vecs"])
        frag_forms = safe_device(batch["frag_form_vecs"])
        frag_morgans = safe_device(batch["frag_morgans"])
        dag_graphs = safe_device(batch["dag_graphs"])


        closest_graphs = safe_device(batch["closest_frag_graphs"])
        closest_root_repr = safe_device(batch["closest_root_reprs"])
        closest_ind_maps = safe_device(batch["closest_inds"])
        closest_num_frags = safe_device(batch["closest_num_frags"])
        closest_broken = safe_device(batch["closest_broken_bonds"])
        closest_adducts = adducts
        closest_max_remove_hs = safe_device(batch["closest_max_remove_hs"])
        closest_max_add_hs = safe_device(batch["closest_max_add_hs"])
        closest_masses = safe_device(batch["closest_masses"])
        closest_root_forms = safe_device(batch["closest_root_form_vecs"])
        closest_frag_forms = safe_device(batch["closest_frag_form_vecs"])
        closest_frag_morgans = safe_device(batch["closest_frag_morgans"])
        closest_dag_graphs = safe_device(batch["closest_dag_graphs"])
        closest_inten_targs = safe_device(batch["closest_inten_targs"])


        distances = safe_device(batch["distances"])
        ref_collision_engs = safe_device(batch["ref_collision_engs"])
        ref_inten_targs = safe_device(batch["ref_inten_targs"])
        ref_counts = safe_device(batch["ref_counts"])

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
            instruments=instruments if self.inten_model_obj.embed_instrument else None,
            collision_engs=collision_engs,
            frag_morgans=frag_morgans,
            dag_graphs=dag_graphs,
            closest_graphs = closest_graphs,
            closest_root_repr = closest_root_repr,
            closest_ind_maps = closest_ind_maps,
            closest_num_frags = closest_num_frags,
            closest_broken=closest_broken,
            closest_adducts=closest_adducts,
            closest_instruments=closest_instruments if self.inten_model_obj.embed_instrument else None,
            closest_max_remove_hs=closest_max_remove_hs,
            closest_max_add_hs=closest_max_add_hs,
            closest_masses=closest_masses,
            closest_root_forms=closest_root_forms,
            closest_frag_forms=closest_frag_forms,
            closest_frag_morgans = closest_frag_morgans,
            closest_dag_graphs=closest_dag_graphs,
            closest_inten_targs=closest_inten_targs,
            distances = distances,
            ref_collision_engs=ref_collision_engs,
            ref_inten_targs=ref_inten_targs,
            ref_counts=ref_counts
        )

        if binned_out:
            out = inten_preds
        else:
            out = {"spec": [], "frag": []}
            if not self.inten_model_obj.include_unshifted_mz:
                masses = masses[:, :, :1, :].contiguous()  # only keep m/z with adduct shift
            for inten_pred, mass, inten_frag_id, out_tree, n in \
                    zip(inten_preds["spec"], masses.cpu().numpy(), inten_frag_ids, out_trees, num_frags.cpu().numpy()):
                out["spec"].append(np.stack((mass[:n].reshape(-1), inten_pred.reshape(-1)), axis=1))
                out_frags = out_tree["frags"]
                out["frag"].append([out_frags[id]["frag"] for id in inten_frag_id])

        if batched_input:
            if not canonical_root_smi and sum(valid_mask) < batch_size:
                rebatched_out = dict()
                rebatched_out["spec"] = []
                for elem in valid_mask:
                    if elem:
                        rebatched_out['spec'].append(out['spec'].pop(0))
                    else:
                        rebatched_out['spec'].append(np.zeros((15000,)))
                return rebatched_out
            else:
                return out
        else:
            return {k: v[0] for k, v in out.items()}
