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
        self.embed_instrument = self.inten_model_obj.embed_instrument

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

        def empty_single_output():
            out = {
                "spec": np.zeros((0, 2), dtype=np.float32),
                "frag": np.zeros((0, 0), dtype=bool),
            }
            if binned_out:
                out["binned_spec"] = np.zeros(
                    (self.inten_model_obj.inten_buckets.shape[-1],), dtype=np.float32
                )
            return out

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
            if instrument is None:
                instrument = [None] * len(root_smi)
        batch_size = len(root_smi)
        min_distances = [None] * batch_size if min_distances is None else list(min_distances)
        valid_ref_counts = [None] * batch_size if valid_ref_counts is None else list(valid_ref_counts)
        closests = [None] * batch_size if closests is None else list(closests)
        name = [None] * batch_size if name is None else list(name)
        if not canonical_root_smi:
            root_smi = [common.rm_stereo(smi) for smi in root_smi]
            root_smi = [common.smiles_from_inchi(common.inchi_from_smiles(_)) for _ in root_smi]
            valid_mask = np.array([r_smi is not None for r_smi in root_smi], dtype=bool)
            if sum(valid_mask) < batch_size:
                print("['joint_model.py']: Some SMILES could not be canonicalized via inchi: ", [(smi[i], name[i]) for i in range(batch_size) if not valid_mask[i]])
                print("['joint_model.py']:", )
                if valid_mask.sum() == 0:
                    if batched_input:
                        empty_batch = {"spec": [], "frag": []}
                        if binned_out:
                            empty_batch["binned_spec"] = []
                        for _ in range(batch_size):
                            empty_out = empty_single_output()
                            empty_batch["spec"].append(empty_out["spec"])
                            empty_batch["frag"].append(empty_out["frag"])
                            if binned_out:
                                empty_batch["binned_spec"].append(empty_out["binned_spec"])
                        return empty_batch
                    return empty_single_output()

                root_smi = np.array(root_smi, dtype=object)[valid_mask].tolist()
                collision_eng = np.array(collision_eng, dtype=object)[valid_mask].tolist()
                precursor_mz = np.array(precursor_mz, dtype=object)[valid_mask].tolist()
                adduct = np.array(adduct, dtype=object)[valid_mask].tolist()
                instrument = np.array(instrument, dtype=object)[valid_mask].tolist()
                min_distances = np.array(min_distances, dtype=object)[valid_mask].tolist()
                valid_ref_counts = np.array(valid_ref_counts, dtype=object)[valid_mask].tolist()
                closests = np.array(closests, dtype=object)[valid_mask].tolist()
                name = np.array(name, dtype=object)[valid_mask].tolist()
        else:
            valid_mask = np.ones(batch_size, dtype=bool)

        frag_preds, _ = self.gen_model_obj.predict_mol(
            root_smi=root_smi,
            collision_eng=collision_eng,
            precursor_mz=precursor_mz,
            adduct=adduct,
            instrument=instrument if self.gen_model_obj.embed_instrument else None,
            threshold=threshold,
            device=device,
            max_nodes=max_nodes,
            canonical_root_smi=True,
        )
        processed_trees = []
        num_frags = frag_preds["nfrags"].detach().cpu().tolist()
        for batch_ind, (r_smi, colli_eng, adct, inst, p_mz, min_distance, valid_ref_count, closest) in enumerate(
            zip(root_smi, collision_eng, adduct, instrument, precursor_mz, min_distances, valid_ref_counts, closests)
        ):
            n_frags = int(num_frags[batch_ind])
            pred_ms = common.MassSpec(
                root_canonical_smiles=r_smi,
                collision_energy=colli_eng,
                adduct=adct,
                probs=frag_preds["probs"][batch_ind, :n_frags].detach().cpu().numpy(),
                brokens=frag_preds["brokens"][batch_ind, :n_frags].detach().cpu().numpy().astype(np.int64),
                masses_no_adduct=frag_preds["masses_no_adduct"][batch_ind, :n_frags].detach().cpu().numpy(),
                frag_form_vecs=frag_preds["frag_form_vecs"][batch_ind, :n_frags].detach().cpu().numpy().round().astype(np.int64),
                frags=frag_preds["frags"][batch_ind, :n_frags].detach().cpu().numpy(),
            )
            processed_tree = self.inten_tp.process_tree_inten_pred(pred_ms)["dgl_tree"]
            processed_tree["instrument"] = common.instrument2onehot_pos[inst]
            processed_tree["adduct"] = common.ion2onehot_pos[adct]
            processed_tree["name"] = ""
            processed_tree["precursor"] = p_mz
            if self.add_ref and closest is not None and valid_ref_count is not None and len(closest) > 0:
                valid_ref_count = min(int(valid_ref_count), self.max_ref_count)
                data = self.db[int(closest[0])]
                spec_indices = np.asarray(closest[:valid_ref_count], dtype=np.int64)
                ref_engs = self.ref_engs[spec_indices]
                ref_specs = self.ref_specs[spec_indices, :].toarray()
                processed_tree["distance"] = min_distance
                processed_tree["ref"] = data
                processed_tree["ref_count"] = valid_ref_count
                processed_tree["ref_collision_engs"] = ref_engs
                processed_tree["ref_inten_targs"] = ref_specs
            else:
                processed_tree["distance"] = None
                processed_tree["ref"] = None
                processed_tree["ref_count"] = None
                processed_tree["ref_collision_engs"] = None
                processed_tree["ref_inten_targs"] = None
            processed_trees.append(processed_tree)
        batch = self.inten_collate_fn(processed_trees)

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
        instruments = safe_device(batch["instruments"]) if self.embed_instrument else None
        closest_instruments = instruments
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

        inten_preds = self.inten_model_obj.forward(
            graphs=frag_graphs,
            root_repr=root_reprs,
            ind_maps=ind_maps,
            num_frags=num_frags,
            broken=broken_bonds,
            collision_engs=collision_engs,
            adducts=adducts,
            instruments=instruments if self.inten_model_obj.embed_instrument else None,
            frag_morgans=frag_morgans,
            max_add_hs=max_add_hs,
            max_remove_hs=max_remove_hs,
            masses=masses,
            root_forms=root_forms,
            frag_forms=frag_forms,
            dag_graphs=dag_graphs,
            closest_graphs=closest_graphs,
            closest_root_repr=closest_root_repr,
            closest_ind_maps=closest_ind_maps,
            closest_num_frags=closest_num_frags,
            closest_broken=closest_broken,
            closest_adducts=closest_adducts,
            closest_instruments=closest_instruments if self.inten_model_obj.embed_instrument else None,
            closest_max_remove_hs=closest_max_remove_hs,
            closest_max_add_hs=closest_max_add_hs,
            closest_masses=closest_masses,
            closest_root_forms=closest_root_forms,
            closest_frag_forms=closest_frag_forms,
            closest_frag_morgans=closest_frag_morgans,
            closest_dag_graphs=closest_dag_graphs,
            closest_inten_targs=closest_inten_targs,
            distances=distances,
            ref_collision_engs=ref_collision_engs,
            ref_inten_targs=ref_inten_targs,
            ref_counts=ref_counts,
        )

        output = inten_preds["output"][:, :, 0]
        output_binned = inten_preds["output_binned"][:, 0, :]
        out = {"spec": [], "frag": []}
        num_shifts = len(masses[0, 0, :, :].reshape(-1))
        if not self.inten_model_obj.include_unshifted_mz:
            masses = masses[:, :, :1, :].contiguous()
        for batch_ind, (inten_pred, mass, n) in enumerate(
            zip(output.detach().cpu().numpy(), masses.detach().cpu().numpy(), num_frags.detach().cpu().numpy())
        ):
            out["spec"].append(np.stack((mass[:n].reshape(-1), inten_pred[:n].reshape(-1)), axis=1))
            out["frag"].append(
                frag_preds["frags"][batch_ind, :n]
                .repeat_interleave(num_shifts, dim=0)
                .detach()
                .cpu()
                .numpy()
            )
        if binned_out:
            out["binned_spec"] = [i.detach().cpu().numpy() for i in output_binned]

        if batched_input:
            if not canonical_root_smi and sum(valid_mask) < batch_size:
                rebatched_out = {"spec": [], "frag": []}
                if binned_out:
                    rebatched_out["binned_spec"] = []
                valid_ind = 0
                for elem in valid_mask:
                    if elem:
                        rebatched_out["spec"].append(out["spec"][valid_ind])
                        rebatched_out["frag"].append(out["frag"][valid_ind])
                        if binned_out:
                            rebatched_out["binned_spec"].append(out["binned_spec"][valid_ind])
                        valid_ind += 1
                    else:
                        empty_out = empty_single_output()
                        rebatched_out["spec"].append(empty_out["spec"])
                        rebatched_out["frag"].append(empty_out["frag"])
                        if binned_out:
                            rebatched_out["binned_spec"].append(empty_out["binned_spec"])
                return rebatched_out
            else:
                return out
        else:
            return {k: v[0] for k, v in out.items()}
