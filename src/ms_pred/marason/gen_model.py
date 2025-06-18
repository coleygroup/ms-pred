"""DAG Gen model """
import numpy as np
from typing import List
import json
import torch
import pytorch_lightning as pl
import torch.nn as nn
import dgl
import dgl.nn as dgl_nn


import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.magma.run_magma as magma
import ms_pred.marason.dag_data as dag_data


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
        max_broken: int = magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
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
            max_broken (int, optional): _description_. Defaults to magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"].
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
                    gnn_node_feats=orig_node_feats + adduct_shift, # TODO: why not ev or instrument?
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
        adducts=None,
        instruments=None,
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
            instruments (_type_): _description_
            root_forms (_type_, optional): _description_. Defaults to None.
            frag_forms (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if self.embed_adduct:
            embed_adducts = self.adduct_embedder[adducts.long()]
        if self.embed_instrument:
            embed_instruments = self.instrument_embedder[instruments.long()]
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

                if self.embed_instrument:

                    # TODO: should I account for nan:
                    # self.instrument_nansub = nn.Parameter(torch.zeros(len(set(common.instrument2onehot_pos.values()))))
                    # self.instrument_nansub.requires_grad = False

                    # embed_instruments = torch.where(torch.isnan(embed_instruments), 
                    #                                 self.instrument_nansub.unsqueeze(0),
                    #                                 embed_instruments
                    # )
                    embed_instruments_expand = embed_instruments.repeat_interleave(
                        root_repr.batch_num_nodes(), 0
                    )

                    ndata = root_repr.ndata["h"]
                    ndata = torch.cat([ndata, embed_instruments_expand], -1)
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
            instruments_mapped = embed_instruments[ind_maps]
            instruments_exp = torch.repeat_interleave(
                instruments_mapped, graphs.batch_num_nodes(), dim=0
            )
            concat_list.append(instruments_exp)

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
            collision_engs=batch["collision_engs"],
            precursor_mzs=batch["precursor_mzs"],
            instruments=batch["instruments"] if "instruments" in batch else None,
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
        adduct=None,
        instrument=None, 
        threshold=0,
        device: str = "cpu",
        max_nodes: int = None,
        decode_final_step: bool = True,
        canonical_root_smi: bool = False,
    ) -> List[dict]:
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

        Return:
            Dictionary containing results
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
        if self.embed_adduct:
            adducts = torch.LongTensor([common.ion2onehot_pos[a] if type(a) is str else a for a in adduct]).to(device)
        collision_engs = torch.FloatTensor(collision_eng).to(device)
        if self.embed_instrument:
            instruments = torch.LongTensor([common.instrument2onehot_pos[i] if type(i) is str else i for i in instrument]).to(device)
        precursor_mzs = torch.FloatTensor(precursor_mz).to(device)

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

        form_to_min_score = [{} for _ in range(batch_size)]
        frag_hash_to_entry = [{} for _ in range(batch_size)]
        frag_to_hash = [{} for _ in range(batch_size)]
        stack = [[rf] for rf in root_frag]
        depth = 0
        root_hash = [e.wl_hash(rf) for e, rf in zip(engine, root_frag)]
        for f2h, rf, rh in zip(frag_to_hash, root_frag, root_hash):
            f2h[rf] = rh
        root_score = [e.score_fragment(rf)[1] for e, rf in zip(engine, root_frag)]
        id_ = [0 for _ in range(batch_size)]
        # TODO: Compute as in fragment engine
        root_entry = [{
            "frag": int(rf),
            "frag_hash": rh,
            "parents": [],
            "atoms_pulled": [],
            "left_pred": [],
            "max_broken": 0,
            "tree_depth": 0,
            "id": 0,
            "prob_gen": 1,
            "score": rs,
        } for rf, rh, rs in zip(root_frag, root_hash, root_score)]
        id_ = [i+1 for i in id_]
        for e, re, rf, rh, f2ms, fh2e in zip(engine, root_entry, root_frag, root_hash, form_to_min_score, frag_hash_to_entry):
            re.update(e.atom_pass_stats(rf, depth=0))
            f2ms[re["form"]] = re["score"]
            fh2e[rh] = re

        # Step 3: Run the autoregressive gen loop
        with torch.no_grad():

            # Note: we don't fragment at the final depth
            while depth < max_depth:
                # Convert all new frags to graphs (stack is to run next)
                new_info_dicts = []
                new_graphs = []
                new_stack = []
                reverse_idx = []
                batched_select = []
                batched_num_nodes = []
                batched_old_edge_idx = [0]
                idx_offset = 0
                for rev_i, (st, e, rg) in enumerate(zip(stack, engine, root_graph_dict)):
                    for i in st:
                        info = self.tree_processor.get_frag_info(i, e)
                        if len(info['new_to_old']) > 1:
                            new_info_dicts.append(info)
                            reverse_idx.append(rev_i)
                            new_graphs.append(rg['graph'])
                            new_stack.append(i)
                            batched_select.append(info['new_to_old'] + idx_offset)
                            batched_num_nodes.append(len(info['new_to_old']))
                            idx_offset += rg['graph'].number_of_nodes()
                            batched_old_edge_idx.append(batched_old_edge_idx[-1] + rg['graph'].number_of_edges())
                if len(new_info_dicts) == 0:
                    break
                stack = new_stack
                batched_select = torch.from_numpy(np.concatenate(batched_select)).to(device)
                batched_num_nodes = torch.LongTensor(batched_num_nodes).to(device)
                batched_old_edge_idx = torch.LongTensor(batched_old_edge_idx[1:]).to(device)

                # Get batched new DGL graph by extracting subgraph
                frag_batch = dgl.batch(new_graphs).to(device)
                frag_batch = frag_batch.subgraph(batched_select)
                batched_num_edges = torch.bincount(torch.bucketize(frag_batch.edata[dgl.EID], batched_old_edge_idx, right=True))
                frag_batch.set_batch_num_nodes(batched_num_nodes)
                frag_batch.set_batch_num_edges(batched_num_edges)

                frag_forms = [i["form"] for i in new_info_dicts]
                frag_form_vecs = [common.formula_to_dense(i) for i in frag_forms]
                frag_form_vecs = torch.FloatTensor(np.array(frag_form_vecs)).to(device)
                new_frag_hashes = [engine[ri].wl_hash(i) for i, ri in zip(stack, reverse_idx)]

                for st, nfh, ri in zip(stack, new_frag_hashes, reverse_idx):
                    frag_to_hash[ri][st] = nfh

                # if DGL graph has >40000 nodes, split the batch to cap GPU memory usage

                # pred_leaving_list = []
                # for _frag_batch, _new_frag_hashes, _rev_idx, _frag_form_vecs in \
                #         split_batch(frag_batch, new_frag_hashes, reverse_idx, frag_form_vecs):
                frag_batch = frag_batch.to(device)
                self.tree_processor.add_pe_embed(frag_batch)  # add random walk feature. GPU memory intensive!
                inds = torch.tensor(reverse_idx).long().to(device)

                broken_nums_ar = np.array(
                    [frag_hash_to_entry[ri][i]["max_broken"] for i, ri in zip(new_frag_hashes, reverse_idx)]
                )
                broken_nums_tensor = torch.FloatTensor(broken_nums_ar).to(device)

                pred_leaving = self.forward(
                    graphs=frag_batch,
                    root_repr=root_repr,
                    ind_maps=inds,
                    broken=broken_nums_tensor,  # torch.ones_like(inds) * depth,
                    collision_engs=collision_engs,
                    precursor_mzs=precursor_mzs,
                    adducts=adducts if self.embed_adduct else None, 
                    instruments=instruments if self.embed_instrument else None, 
                    root_forms=root_form_vec,
                    frag_forms=frag_form_vecs,
                )
                pred_leaving = pred_leaving.cpu()  # switch to cpu device
                depth += 1

                # Rank order all the atom preds and predictions
                # Continuously add items to the stack as long as they maintain
                # the max node constraint ranked by prob

                # Get all frag probabilities and sort them
                cur_probs = [sorted(
                    [i["prob_gen"] for i in fh2e.values()]
                )[::-1] for fh2e in frag_hash_to_entry]
                if max_nodes is None:
                    min_prob = torch.full(batch_size, threshold)  # force on cpu
                else:
                    cur_prob_len = torch.LongTensor([len(cp) for cp in cur_probs])
                    thresh_prob = torch.FloatTensor([cp[:max_nodes][-1] for cp in cur_probs])
                    min_prob = torch.where(cur_prob_len < max_nodes, torch.full_like(thresh_prob, threshold), thresh_prob)

                new_items = list(
                    zip(stack,
                        new_frag_hashes,
                        pred_leaving,
                        [d['new_to_old'] for d in new_info_dicts],
                        reverse_idx)
                )
                sorted_order = [[] for _ in range(batch_size)]
                for item_ind, item in enumerate(new_items):
                    frag_hash = item[1]
                    pred_vals_f = item[2]
                    rev_idx = item[-1]
                    parent_prob = frag_hash_to_entry[rev_idx][frag_hash]["prob_gen"]
                    for atom_ind, (atom_pred, prob_gen) in enumerate(
                        zip(pred_vals_f, parent_prob * pred_vals_f)
                    ):
                        sorted_order[rev_idx].append(
                            dict(
                                item_ind=item_ind,
                                atom_ind=atom_ind,
                                prob_gen=prob_gen.item(),
                                atom_pred=atom_pred.item(),
                                orig_entry=new_items[item_ind],
                            )
                        )

                sorted_order = [sorted(so, key=lambda x: -x["prob_gen"]) for so in sorted_order]
                new_stack = [[] for _ in range(batch_size)]

                # Process ordered list continuously
                batch_to_process = [{
                    "frag_hash_to_entry": frag_hash_to_entry[rev_idx],
                    "frag_to_hash": frag_to_hash[rev_idx],
                    "form_to_min_score": form_to_min_score[rev_idx],
                    "engine": engine[rev_idx],
                    "min_prob": min_prob[rev_idx],
                    "id_": id_[rev_idx],
                    "sorted_order": sorted_order[rev_idx],
                    "depth": depth,
                    "max_nodes": max_nodes,
                    "threshold": threshold,
                } for rev_idx in range(batch_size)]

                if depth == max_depth and not decode_final_step: # return an unprocessed list
                    return batch_to_process
                else:
                    new_vals = [decoder_wrapper(b) for b in batch_to_process]

                    # Update the global variables
                    for rev_idx in range(batch_size):
                        frag_hash_to_entry[rev_idx] = new_vals[rev_idx]["frag_hash_to_entry"]
                        frag_to_hash[rev_idx] = new_vals[rev_idx]["frag_to_hash"]
                        form_to_min_score[rev_idx] = new_vals[rev_idx]["form_to_min_score"]
                        id_[rev_idx] = new_vals[rev_idx]["id_"]
                        sorted_order[rev_idx] = new_vals[rev_idx]["sorted_order"]
                        new_stack[rev_idx] = new_vals[rev_idx]["new_stack"]

                    stack = new_stack

        # Only get min score for ech formula
        frag_hash_to_entry = [{
            k: v
            for k, v in fh2e.items()
            if f2ms[v["form"]] == v["score"]
        } for fh2e, f2ms in zip(frag_hash_to_entry, form_to_min_score)]

        if max_nodes is not None:
            return_entries = []
            for fh2e in frag_hash_to_entry:
                sorted_keys = sorted(
                    list(fh2e.keys()),
                    key=lambda x: -fh2e[x]["prob_gen"],
                )
                fh2e = {
                    k: fh2e[k] for k in sorted_keys[:max_nodes]
                }
                return_entries.append(fh2e)
            frag_hash_to_entry = return_entries
        if batched_input:
            return frag_hash_to_entry
        else:
            return frag_hash_to_entry[0]

    @staticmethod
    def parallel_consumer_decoder(data):
        param_dic = data["param_dic"]
        max_nodes = param_dic["max_nodes"]

        # The real processing steps
        new_val = decoder_wrapper(param_dic)
        frag_hash_to_entry = new_val["frag_hash_to_entry"]
        form_to_min_score = new_val["form_to_min_score"]

        frag_hash_to_entry = {
            k: v
            for k, v in frag_hash_to_entry.items()
            if form_to_min_score[v["form"]] == v["score"]
        }

        if max_nodes is not None:
            sorted_keys = sorted(
                list(frag_hash_to_entry.keys()),
                key=lambda x: -frag_hash_to_entry[x]["prob_gen"],
            )
            frag_hash_to_entry = {
                k: frag_hash_to_entry[k] for k in sorted_keys[:max_nodes]
            }

        # Formulate the result
        output = {
            "root_canonical_smiles": data["root_canonical_smiles"],
            "name": data["name"],
            "frags": frag_hash_to_entry,
            "collision_energy": data["collision_energy"],
            # "instrument": data["instrument"], # TODO: not sure if this will break things/if ne
        }
        output = json.dumps(output, indent=2)

        return data["out_name"], output


def decoder_wrapper(param_dic):
    return auto_regressive_decode(**param_dic)


def auto_regressive_decode(sorted_order, frag_hash_to_entry, frag_to_hash, form_to_min_score, engine, min_prob, id_,
                           depth, max_nodes, threshold):
    new_stack = []
    for new_item in sorted_order:
        prob_gen = new_item["prob_gen"]
        atom_ind = new_item["atom_ind"]
        atom_pred = new_item["atom_pred"]
        item_ind = new_item["item_ind"]

        # Filter out on minimum prob
        if prob_gen <= min_prob:
            continue

        # Calc stack ind
        orig_entry = new_item["orig_entry"]
        frag_int = orig_entry[0]
        frag_hash = orig_entry[1]
        dgl_new_to_old = orig_entry[3]

        # Get atom ind
        atom = dgl_new_to_old[atom_ind]

        # Calc remove dict
        out_dicts = engine.remove_atom(frag_int, int(atom))

        # Update atoms_pulled for parent
        frag_hash_to_entry[frag_hash]["atoms_pulled"].append(int(atom))
        frag_hash_to_entry[frag_hash]["left_pred"].append(float(atom_pred))
        parent_broken = frag_hash_to_entry[frag_hash]["max_broken"]

        for out_dict in out_dicts:
            out_hash = out_dict["new_hash"]
            out_frag = out_dict["new_frag"]
            rm_bond_t = out_dict["rm_bond_t"]
            frag_to_hash[out_frag] = out_hash
            current_entry = frag_hash_to_entry.get(out_hash)

            max_broken = parent_broken + rm_bond_t

            # Define probability of generating
            if current_entry is None:
                score = engine.score_fragment(int(out_frag))[1]

                new_stack.append(out_frag)
                new_entry = {
                    "frag": int(out_frag),
                    "frag_hash": out_hash,
                    "score": score,
                    "id": id_,
                    "parents": [frag_hash],
                    "atoms_pulled": [],
                    "left_pred": [],
                    "max_broken": max_broken,
                    "tree_depth": depth,
                    "prob_gen": prob_gen,
                }
                id_ += 1
                new_entry.update(
                    engine.atom_pass_stats(out_frag, depth=max_broken)
                )

                # reset to best score
                temp_form = new_entry["form"]
                prev_best_score = form_to_min_score.get(
                    temp_form, float("inf")
                )
                form_to_min_score[temp_form] = min(
                    new_entry["score"], prev_best_score
                )
                frag_hash_to_entry[out_hash] = new_entry

            else:
                current_entry["parents"].append(frag_hash)
                current_entry["prob_gen"] = max(
                    current_entry["prob_gen"], prob_gen
                )

            # Update cur probs for the current batch index
            # This is inefficient and can be made smarter without
            # doing another minimum calculation
            cur_probs = sorted(
                [i["prob_gen"] for i in frag_hash_to_entry.values()]
            )[::-1]
            if max_nodes is None or len(cur_probs) < max_nodes:
                min_prob = threshold
            elif max_nodes is not None and len(cur_probs) >= max_nodes:
                min_prob = cur_probs[max_nodes - 1]
            else:
                raise NotImplementedError()

    return {
        "frag_hash_to_entry": frag_hash_to_entry,
        "frag_to_hash": frag_to_hash,
        "form_to_min_score": form_to_min_score,
        "min_prob": min_prob,
        "id_": id_,
        "sorted_order": sorted_order,
        "new_stack": new_stack,
    }