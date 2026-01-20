"""DAG Gen model """
import numpy as np
from typing import Tuple
import torch
import pytorch_lightning as pl
import torch.nn as nn
import dgl
import dgl.nn as dgl_nn


import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.dag_pred.dag_data as dag_data


class FragGNN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        set_layers: int = 2,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0,
        dropout: float = 0,
        mpnn_type: str = "GGNN",
        pool_op: str = "avg",
        node_feats: int = common.ELEMENT_DIM + common.MAX_H,
        pe_embed_k: int = 0,
        max_broken: int = fragmentation.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        root_encode: str = "gnn",
        inject_early: bool = False,
        warmup: int = 1000,
        embed_adduct=False,
        embed_collision=False,
        embed_instrument=False,
        embed_elem_group=False,
        encode_forms: bool = False,
        add_hs: bool = False,
        **kwargs,
    ):
        """__init__ _summary_

        Args:
            hidden_size (int): _description_
            layers (int, optional): _description_. Defaults to 2.
            set_layers (int, optional): _description_. Defaults to 2.
            learning_rate (float, optional): _description_. Defaults to 7e-4.
            lr_decay_rate (float, optional): _description_. Defaults to 1.0.
            weight_decay (float, optional): _description_. Defaults to 0.
            dropout (float, optional): _description_. Defaults to 0.
            mpnn_type (str, optional): _description_. Defaults to "GGNN".
            pool_op (str, optional): _description_. Defaults to "avg".
            node_feats (int, optional): _description_. Defaults to common.ELEMENT_DIM+common.MAX_H.
            pe_embed_k (int, optional): _description_. Defaults to 0.
            max_broken (int, optional): _description_. Defaults to fragmentation.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"].
            root_encode (str, optional): _description_. Defaults to "gnn".
            inject_early (bool, optional): _description_. Defaults to False.
            warmup (int, optional): _description_. Defaults to 1000.
            embed_adduct (bool, optional): _description_. Defaults to False.
            embed_collision (bool, optional): _description_. Defaults to False.
            embed_elem_group (bool, optional): _description_. Defaults to False.
            encode_forms (bool, optional): _description_. Defaults to False.
            add_hs (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
            NotImplementedError: _description_
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.root_encode = root_encode
        self.pe_embed_k = pe_embed_k
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_instrument = embed_instrument
        self.embed_elem_group = embed_elem_group
        self.encode_forms = encode_forms
        self.add_hs = add_hs

        self.tree_processor = dag_data.TreeProcessor(
            root_encode=root_encode, pe_embed_k=pe_embed_k, add_hs=self.add_hs, embed_elem_group=self.embed_elem_group,
        )
        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim

            # Account for diffs
            self.formula_in_dim *= 2

        self.pool_op = pool_op
        self.inject_early = inject_early

        self.layers = layers
        self.mpnn_type = mpnn_type
        self.set_layers = set_layers

        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.dropout = dropout

        self.max_broken = max_broken + 1
        self.broken_onehot = torch.nn.Parameter(torch.eye(self.max_broken))
        self.broken_onehot.requires_grad = False
        self.broken_clamp = max_broken

        edge_feats = fragmentation.MAX_BONDS

        orig_node_feats = node_feats
        if self.inject_early:
            node_feats = node_feats + self.hidden_size

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(set(common.ion2onehot_pos.values()))
            onehot_types = torch.eye(adduct_types)
            if self.embed_elem_group:
                adduct_modes = len(set([j for i in common.ion_pos2extra_multihot.values() for j in i]))
                multihot_modes = torch.zeros((adduct_types, adduct_modes))
                for i in range(adduct_types):
                    for j in common.ion_pos2extra_multihot[i]:
                        multihot_modes[i, j] = 1
                adduct_embedder = torch.cat((onehot_types, multihot_modes), dim=-1)
                self.adduct_embedder = nn.Parameter(adduct_embedder.float())
                self.adduct_embedder.requires_grad = False
                adduct_shift = adduct_types + adduct_modes
            else:
                self.adduct_embedder = nn.Parameter(onehot_types.float())
                self.adduct_embedder.requires_grad = False
                adduct_shift = adduct_types

        collision_shift = 0
        if self.embed_collision:
            pe_dim = common.COLLISION_PE_DIM
            pe_scalar = common.COLLISION_PE_SCALAR
            pe_power = 2 * torch.arange(pe_dim // 2) / pe_dim
            self.collision_embedder_denominators = nn.Parameter(torch.pow(pe_scalar, pe_power))
            self.collision_embedder_denominators.requires_grad = False
            collision_shift = pe_dim

            # Not used: Compute the merged collision embedding as the mean of all energies 0 - 100 eV
            # collision_eng_steps = torch.arange(0, 100, 0.01)
            # self.collision_embed_merged = nn.Parameter(torch.cat(
            #     (torch.sin(collision_eng_steps.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
            #      torch.cos(collision_eng_steps.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
            #     dim=1
            # ).mean(dim=0))
            # self.collision_embed_merged.requires_grad = False

            # All-zero for collision == nan
            self.collision_embed_merged = nn.Parameter(torch.zeros(pe_dim))
            self.collision_embed_merged.requires_grad = False

        instrument_shift = 0
        if self.embed_instrument:
            instrument_types = len(set(common.instrument2onehot_pos.values()))
            onehot_instrument = torch.eye(instrument_types)
            self.instrument_embedder = nn.Parameter(onehot_instrument.float())
            self.instrument_embedder.requires_grad = False
            instrument_shift = instrument_types


        # Define network
        self.gnn = nn_utils.MoleculeGNN(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.layers,
            set_transform_layers=self.set_layers,
            mpnn_type=self.mpnn_type,
            gnn_node_feats=node_feats + adduct_shift + collision_shift + instrument_shift,
            gnn_edge_feats=edge_feats,
            dropout=self.dropout,
        )

        if self.root_encode == "gnn":
            self.root_module = self.gnn

            # if inject early, need separate root and child GNN's
            if self.inject_early:
                self.root_module = nn_utils.MoleculeGNN(
                    hidden_size=self.hidden_size,
                    num_step_message_passing=self.layers,
                    set_transform_layers=self.set_layers,
                    mpnn_type=self.mpnn_type,
                    gnn_node_feats=orig_node_feats + adduct_shift,
                    gnn_edge_feats=edge_feats,
                    dropout=self.dropout,
                )
        elif self.root_encode == "fp":
            self.root_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                output_size=None,
                dropout=self.dropout,
                use_residuals=True,
                num_layers=1,
            )
        else:
            raise ValueError()

        # MLP layer to take representations from the pooling layer
        # And predict a single scalar value at each of them
        # I.e., Go from size B x 2h -> B x 1
        self.output_map = nn_utils.MLPBlocks(
            input_size=self.hidden_size * 3 + self.max_broken + self.formula_in_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            dropout=self.dropout,
            num_layers=1,
            use_residuals=True,
        )

        if self.pool_op == "avg":
            self.pool = dgl_nn.AvgPooling()
        elif self.pool_op == "attn":
            self.pool = dgl_nn.GlobalAttentionPooling(nn.Linear(hidden_size, 1))
        else:
            raise NotImplementedError()

        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(
        self,
        graphs,
        root_repr,
        ind_maps,
        broken,
        collision_engs,
        precursor_mzs,
        adducts,
        root_forms=None,
        frag_forms=None,
    ):
        """forward _summary_

        Args:
            graphs (_type_): _description_
            root_repr (_type_): _description_
            ind_maps (_type_): _description_
            broken (_type_): _description_
            collision_engs (_type_): _description_
            precursor_mzs (_type_): _description_
            adducts (_type_): _description_
            root_forms (_type_, optional): _description_. Defaults to None.
            frag_forms (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        embed_adducts = self.adduct_embedder[adducts.long()]
        if self.root_encode == "fp":
            root_embeddings = self.root_module(root_repr)
            raise NotImplementedError()
        elif self.root_encode == "gnn":
            with root_repr.local_scope():
                if self.embed_adduct:
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_repr.batch_num_nodes(), 0
                    )
                    ndata = root_repr.ndata["h"]
                    ndata = torch.cat([ndata, embed_adducts_expand], -1)
                    root_repr.ndata["h"] = ndata
                if self.embed_collision:
                    embed_collision = torch.cat(
                        (torch.sin(collision_engs.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                         torch.cos(collision_engs.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                        dim=1
                    )
                    embed_collision = torch.where(  # handle entries without collision energy (== nan)
                        torch.isnan(embed_collision), self.collision_embed_merged.unsqueeze(0), embed_collision
                    )
                    embed_collision_expand = embed_collision.repeat_interleave(
                        root_repr.batch_num_nodes(), 0
                    )
                    ndata = root_repr.ndata["h"]
                    ndata = torch.cat([ndata, embed_collision_expand], -1)
                    root_repr.ndata["h"] = ndata
                root_embeddings = self.root_module(root_repr)
                root_embeddings = self.pool(root_repr, root_embeddings)
        else:
            pass

        # Line up the features to be parallel between fragment avgs and root
        # graphs
        ext_root = root_embeddings[ind_maps]

        # Extend the root further to cover each individual atom
        ext_root_atoms = torch.repeat_interleave(
            ext_root, graphs.batch_num_nodes(), dim=0
        )
        concat_list = [graphs.ndata["h"]]

        if self.inject_early:
            concat_list.append(ext_root_atoms)

        if self.embed_adduct:
            adducts_mapped = embed_adducts[ind_maps]
            adducts_exp = torch.repeat_interleave(
                adducts_mapped, graphs.batch_num_nodes(), dim=0
            )
            concat_list.append(adducts_exp)

        if self.embed_collision:
            collision_mapped = embed_collision[ind_maps]
            collision_exp = torch.repeat_interleave(
                collision_mapped, graphs.batch_num_nodes(), dim=0
            )
            concat_list.append(collision_exp)

        if self.embed_instrument:                    
            embed_instruments_exp = embed_instruments.repeat_interleave(
                root_repr.batch_num_nodes(), 0
            )
            concat_list.append(embed_instruments_exp)

        with graphs.local_scope():
            graphs.ndata["h"] = torch.cat(concat_list, -1).float()

            frag_embeddings = self.gnn(graphs)

            # Average embed the full root molecules and fragments
            avg_frags = self.pool(graphs, frag_embeddings)

        # Extend the avg of each fragment
        ext_frag_atoms = torch.repeat_interleave(
            avg_frags, graphs.batch_num_nodes(), dim=0
        )

        exp_num = graphs.batch_num_nodes()
        # Do the same with the avg fragments

        broken = torch.clamp(broken, max=self.broken_clamp)
        ext_frag_broken = torch.repeat_interleave(broken, exp_num, dim=0)
        broken_onehots = self.broken_onehot[ext_frag_broken.long()]

        mlp_cat_vec = [
            ext_root_atoms,
            ext_root_atoms - ext_frag_atoms,
            frag_embeddings,
            broken_onehots,
        ]
        if self.encode_forms:
            root_exp = root_forms[ind_maps]
            diffs = root_exp - frag_forms
            form_encodings = self.embedder(frag_forms)
            diff_encodings = self.embedder(diffs)
            form_atom_exp = torch.repeat_interleave(form_encodings, exp_num, dim=0)
            diff_atom_exp = torch.repeat_interleave(diff_encodings, exp_num, dim=0)

            mlp_cat_vec.extend([form_atom_exp, diff_atom_exp])

        hidden = torch.cat(
            mlp_cat_vec,
            dim=1,
        )

        output = self.output_map(hidden)
        output = self.sigmoid(output)
        padded_out = nn_utils.pad_packed_tensor(output, graphs.batch_num_nodes(), 0)
        padded_out = torch.squeeze(padded_out, -1)
        return padded_out

    def loss_fn(self, outputs, targets, natoms):
        """loss_fn.

        Args:
            outputs: Outputs after sigmoid fucntion
            targets: Target binary vals
            natoms: Number of atoms in each atom to consider padding

        """
        targets = targets.float()
        loss = self.bce_loss(outputs, targets)
        #loss = loss * (0.5 + 0.5 * targets)
        is_valid = (
            torch.arange(loss.shape[1], device=loss.device)[None, :] < natoms[:, None]
        )
        pooled_loss = torch.sum(loss * is_valid) / torch.sum(natoms)
        return pooled_loss

    def _common_step(self, batch, name="train"):
        pred_leaving = self.forward(
            batch["frag_graphs"],
            batch["root_reprs"],
            batch["inds"],
            broken=batch["broken_bonds"],
            adducts=batch["adducts"],
            instruments=batch["instruments"],
            collision_engs=batch["collision_engs"],
            precursor_mzs=batch["precursor_mzs"],
            root_forms=batch["root_form_vecs"],
            frag_forms=batch["frag_form_vecs"],
        )
        loss = self.loss_fn(pred_leaving, batch["targ_atoms"], batch["frag_atoms"])
        self.log(
            f"{name}_loss", loss.item(), on_epoch=True, batch_size=len(batch["names"])
        )
        return loss

    def training_step(self, batch, batch_idx):
        """training_step."""
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        """validation_step."""
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
        """test_step."""
        return self._common_step(batch, name="test")

    def configure_optimizers(self):
        """configure_optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }
        return ret

    def predict_mol(
        self,
        root_smi: str,
        collision_eng,
        precursor_mz,
        adduct,
        instrument,
        threshold=0,
        device: str = "cpu",
        max_nodes: int = None,
        decode_final_step: bool = True,
        canonical_root_smi: bool = False,
    ) -> Tuple[dict, dgl.DGLGraph]:
        """prdict_mol.

        Predict a new fragmentation tree from a starting root molecule
        autoregressively. First a new fragment is added to the
        frag_hash_to_entry dict and also put on the stack. Then it is
        fragmented and its "atoms_pulled" and "left_pred" are updated
        accordingly. The resulting new fragments are added to the hash.

        Args:
            root_smi (smi)
            threshold: Leaving probability
            device: Device
            max_nodes (int): Max number to include
            decode_final_step (bool): if False, do not decode the final
              auto-regressive step. Instead, process it later by multi-
              processing workers
            canonical_root_smi (bool): if the root_smi is canonicalized

        Return:
            Dictionary containing results, root graph object
        """
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
        assert batch_size > 0

        # Step 1: Get a fragmentation engine for root mol
        engine = [fragmentation.FragmentEngine(rsmi, mol_str_canonicalized=canonical_root_smi) for rsmi in root_smi]
        max_depth = engine[0].max_tree_depth  # all max_depth should be the same
        root_frag = [e.get_root_frag() for e in engine]
        root_form = [common.form_from_smi(rsmi) for rsmi in root_smi]
        root_form_vec = torch.FloatTensor(np.array([common.formula_to_dense(rf) for rf in root_form])).to(device)
        adducts = torch.LongTensor([common.ion2onehot_pos[a] if type(a) is str else a for a in adduct]).to(device)
        collision_engs = torch.FloatTensor(collision_eng).to(device)
        precursor_mzs = torch.FloatTensor(precursor_mz).to(device)
        instruments = torch.LongTensor([common.instrument2onehot_pos[i] if type(i) is str else i for i in instrument]).to(device)

        # Step 2: Featurize the root molecule
        root_graph_dict = [self.tree_processor.featurize_frag(frag=rf, engine=e, add_random_walk=False)  # add random walk feature later in batched
                           for rf, e in zip(root_frag, engine)]

        root_repr = None
        if self.root_encode == "gnn":
            root_repr = dgl.batch([rg["graph"] for rg in root_graph_dict]).to(device)
            self.tree_processor.add_pe_embed(root_repr)  # add random walk feature
        elif self.root_encode == "fp":
            root_fp = torch.from_numpy(np.array([common.get_morgan_fp_smi(rsmi) for rsmi in root_smi]))
            root_repr = root_fp.float().to(device)
        depth = 0

        mol_batch_ids = torch.tensor([i for i in range(len(root_frag))], device=device, dtype=torch.long)
        natoms = torch.tensor([i.natoms for i in engine], device=device, dtype=torch.long)
        max_atoms = max(natoms)
        frags_bin = torch.zeros((batch_size, max_atoms), device=device, dtype=torch.bool)
        frags_bin[torch.arange(max_atoms, device=device)[None, :] < natoms[:, None]] = 1
        broken_nums = torch.zeros(batch_size, device=device, dtype=torch.float)
        frag_form_vecs = root_form_vec
        accu_prob = torch.ones_like(broken_nums)

        all_pred_prob = torch.zeros((batch_size, max_nodes), device=device, dtype=torch.float)  # accumulated probability
        all_pred_broken_bond = torch.zeros((batch_size, max_nodes), device=device, dtype=torch.float)  # broken bond order
        all_pred_frags = torch.zeros((batch_size, max_nodes, max_atoms), device=device, dtype=torch.bool)  # frag in binary
        all_pred_frag_mass = torch.zeros((batch_size, max_nodes), device=device, dtype=torch.float)
        all_pred_frag_form_vecs = torch.zeros((batch_size, max_nodes, root_form_vec.shape[-1]), device=device, dtype=torch.float)
        all_pred_prob[:, 0] = accu_prob
        all_pred_frags[:, 0, :] = frags_bin
        all_pred_frag_mass[:, 0] = nn_utils.form_vec_to_mass(root_form_vec)
        all_pred_frag_form_vecs[:, 0, :] = root_form_vec
        graphs = [root_graph_dict[i]["graph"].subgraph(frag[:n]) for i, frag, n in zip(mol_batch_ids, frags_bin, natoms)]
        frag_batch = dgl.batch(graphs).to(device)
        all_pred_graph_spectrum_hash = torch.zeros((batch_size, max_nodes, frag_batch.ndata['h'].shape[1]), device=device, dtype=torch.float)  # graph hash
        all_pred_graph_spectrum_hash[:, 0, :] = nn_utils.msg_passing_frag_graph_hash(frag_batch)

        # Step 3: Run the autoregressive gen loop
        with torch.no_grad():
            # Note: we don't fragment at the final depth
            while depth < max_depth:
                self.tree_processor.add_pe_embed(frag_batch)

                pred_leaving = self.forward(
                    graphs=frag_batch,
                    root_repr=root_repr,
                    ind_maps=mol_batch_ids,
                    broken=broken_nums,  # torch.ones_like(inds) * depth,
                    collision_engs=collision_engs,
                    precursor_mzs=precursor_mzs,
                    adducts=adducts,
                    instruments=instruments,
                    root_forms=root_form_vec,
                    frag_forms=frag_form_vecs,
                )

                cur_max_atoms = pred_leaving.shape[1]
                accu_prob = accu_prob.unsqueeze(-1) * pred_leaving  # accumulated probs of current prediction
                graph_batch_sizes = torch.bincount(mol_batch_ids, minlength=batch_size)
                accu_prob = nn_utils.pad_packed_tensor(accu_prob, graph_batch_sizes, 0).view(batch_size, -1)

                # select top-max_nodes probabilities for graph processing
                temp_prev_prob = torch.cat((all_pred_prob, accu_prob), dim=1)  # all probs (including prev ones)
                #prev_kept_indices = torch.argsort(temp_prev_prob, dim=1, descending=True)[:, :max_nodes]
                prev_kept_indices = torch.argsort(temp_prev_prob, dim=1, descending=True)[:, :max_nodes * 2]

                # indices selected for node removal
                select_mask = (prev_kept_indices >= max_nodes) & \
                              (temp_prev_prob[torch.arange(batch_size, device=device).unsqueeze(-1), prev_kept_indices] > threshold)
                select_batch_idx, pos_idx = select_mask.nonzero(as_tuple=True)
                sel_idx = (prev_kept_indices[select_batch_idx, pos_idx] - max_nodes)
                sel_prob = accu_prob[select_batch_idx, sel_idx]
                frag_batch_idx = (torch.cumsum(graph_batch_sizes, dim=0) - graph_batch_sizes).repeat_interleave(
                    torch.bincount(select_batch_idx, minlength=batch_size)) + sel_idx // cur_max_atoms
                broken_nums_per_frag = broken_nums[frag_batch_idx]

                if len(sel_idx) == 0:  # no new fragments generated
                    break

                # Remove atoms, and find connected components as fragments
                new_frag_batch, broken_bond_orders, new_info = \
                    nn_utils.batch_remove_single_atoms(
                        frag_batch, frag_batch_idx, sel_idx % cur_max_atoms,
                        {"sel_prob": sel_prob, "mol_batch_id": select_batch_idx, "broken_bonds": broken_nums_per_frag}
                    )
                if new_frag_batch is None:  # no new fragments generated
                    break
                broken_bond_orders += new_info["broken_bonds"]  # sum num of broken bonds with previous breakages

                count_per_mol_batch = torch.bincount(new_info["mol_batch_id"], minlength=batch_size)

                # Generate frag ids from DGL node labels
                frag_pos = torch.cat([torch.arange(n, device=device)
                                      for n in count_per_mol_batch], dim=0)  # (num_frags,)
                # assign frag id from DGL node n_id by GPU
                node_ids = new_frag_batch.ndata['n_id']  # (total_nodes,)
                batch_num_nodes = new_frag_batch.batch_num_nodes()  # list of length num_frags
                frag_ids = torch.repeat_interleave(
                    torch.arange(len(batch_num_nodes), device=device),
                    batch_num_nodes
                )  # (total_nodes,)
                mol_per_node = new_info["mol_batch_id"][frag_ids]  # (total_nodes,)
                pos_per_node = frag_pos[frag_ids]  # (total_nodes,)
                temp_new_frag_bin = torch.zeros((batch_size, count_per_mol_batch.max(), max_atoms), device=device, dtype=torch.bool)
                temp_new_frag_bin[mol_per_node, pos_per_node, node_ids] = True
                temp_new_frag_bin = torch.cat((all_pred_frags, temp_new_frag_bin), dim=1)
                temp_new_frag_dec = nn_utils.bin2dec(temp_new_frag_bin)

                # concat and calculate the probability again
                new_accu_prob = nn_utils.pad_packed_tensor(new_info["sel_prob"], count_per_mol_batch, 0)  # B x max(N_frags)
                temp_new_prob = torch.cat((all_pred_prob, new_accu_prob), dim=1)

                # compute fragment hash
                if self.root_encode == "gnn":
                    self.tree_processor.rm_pe_embed(new_frag_batch)
                new_frag_hash = nn_utils.msg_passing_frag_graph_hash(new_frag_batch)
                temp_frag_hash = torch.cat((all_pred_graph_spectrum_hash,
                                            nn_utils.pad_packed_tensor(new_frag_hash, count_per_mol_batch, 0)), dim=1)

                # only keep unique fragments
                tmp_max_frag = max_nodes + count_per_mol_batch.max()
                frag_id_batch_idx = torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, tmp_max_frag)
                _, uniq_frag_inv_idx = nn_utils.np_like_unique(
                    torch.cat((frag_id_batch_idx.unsqueeze(-1),
                               temp_frag_hash), dim=-1).view(-1, temp_frag_hash.shape[-1] + 1), dim=0)
                uniq_frag_mask = torch.zeros_like(temp_new_prob, dtype=torch.bool)
                uniq_frag_mask[uniq_frag_inv_idx // tmp_max_frag, uniq_frag_inv_idx % tmp_max_frag] = True
                temp_new_prob[
                    ~uniq_frag_mask &
                    (temp_new_prob > 0) #&
                    # ~((temp_new_prob == 0) & (torch.arange(tmp_max_frag, device=device).unsqueeze(0) < max_nodes))
                ] = -1  # for new nodes, only unique frags are considered

                new_kept_indices = torch.argsort(temp_new_prob, dim=1, descending=True)[:, :max_nodes]
                all_pred_prob = torch.gather(temp_new_prob, 1, new_kept_indices)

                # update all_pred_frags
                temp_batch_idx = torch.arange(batch_size, device=device)  # (B,)
                temp_batch_idx = temp_batch_idx.unsqueeze(1).expand(-1, new_kept_indices.shape[1])  # (B, K)
                all_pred_frags = temp_new_frag_bin[temp_batch_idx, new_kept_indices]

                # Same for broken bond orders
                new_bb = nn_utils.pad_packed_tensor(broken_bond_orders, count_per_mol_batch, 0)
                temp_new_bb = torch.cat((all_pred_broken_bond, new_bb), dim=1)

                # take the minimal broken bonds among same fragments
                # _, uniq_frag_inv_pos = torch.unique(
                #     torch.stack((frag_id_batch_idx, temp_new_frag_dec), dim=-1).view(-1, 2),
                #     dim=0, return_inverse=True)
                # bb_uniq_frag = torch.full((uniq_frag_inv_pos.max() + 1,), self.max_broken, device=device, dtype=torch.float).\
                #     scatter_reduce_(0, uniq_frag_inv_pos, temp_new_bb.view(-1), 'min')
                # temp_new_bb = torch.gather(bb_uniq_frag, 0, uniq_frag_inv_pos).reshape(batch_size, -1)

                all_pred_broken_bond = torch.gather(temp_new_bb, 1, new_kept_indices)

                # flatten new_kept_indices
                temp_mask = (new_kept_indices >= max_nodes) & (all_pred_prob > threshold)
                temp_idx, pos_idx = temp_mask.nonzero(as_tuple=True)
                temp_offset = torch.cumsum(count_per_mol_batch, 0) - count_per_mol_batch
                new_kept_indices_flat = new_kept_indices[temp_idx, pos_idx] - max_nodes + \
                                        temp_offset.repeat_interleave(torch.bincount(temp_idx, minlength=batch_size))

                if len(new_kept_indices_flat) == 0:  # no new fragments selected
                    break

                # update frag_batch and mol_batch_ids
                frag_batch, _, __ = nn_utils.slice_batched_graph(new_frag_batch, new_kept_indices_flat)
                mol_batch_ids = new_info["mol_batch_id"][new_kept_indices_flat]
                broken_nums = broken_bond_orders[new_kept_indices_flat]
                accu_prob = new_info["sel_prob"][new_kept_indices_flat]

                # frag_form_vecs; update frag mass, frag_form_vecs, graph_spectrum_hash
                frag_form_vecs = nn_utils.frag_to_form_vec(frag_batch, self.add_hs, self.embed_elem_group)
                temp_rerank_mask = new_kept_indices < max_nodes
                temp_rerank_idx, rerank_pos_idx = temp_rerank_mask.nonzero(as_tuple=True)
                all_pred_frag_mass[temp_rerank_mask] = all_pred_frag_mass[temp_rerank_idx, new_kept_indices[temp_rerank_mask]]
                all_pred_frag_mass[temp_idx, pos_idx] = nn_utils.form_vec_to_mass(frag_form_vecs)
                all_pred_frag_form_vecs[temp_rerank_mask] = all_pred_frag_form_vecs[temp_rerank_idx, new_kept_indices[temp_rerank_mask]]
                all_pred_frag_form_vecs[temp_idx, pos_idx] = frag_form_vecs
                all_pred_graph_spectrum_hash[temp_rerank_mask] = all_pred_graph_spectrum_hash[temp_rerank_idx, new_kept_indices[temp_rerank_mask]]
                all_pred_graph_spectrum_hash[temp_idx, pos_idx] = new_frag_hash[new_kept_indices_flat]

                # next step
                depth += 1

        if self.root_encode == "gnn":
            self.tree_processor.rm_pe_embed(root_repr)

        return_dict = {
            "frags": all_pred_frags,
            "probs": all_pred_prob,
            "brokens": all_pred_broken_bond,
            "masses_no_adduct": all_pred_frag_mass,
            "frag_form_vecs": all_pred_frag_form_vecs,
            "root_form_vec": root_form_vec,
            "natoms": natoms,
            "nfrags": torch.sum(all_pred_prob > threshold, dim=-1)
        }

        if not batched_input:
            return_dict = {k: v[0] for k, v in return_dict.items()}
        return return_dict, root_repr

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):  # fix lightning API mismatch for torch>=2.0
        scheduler.step()
