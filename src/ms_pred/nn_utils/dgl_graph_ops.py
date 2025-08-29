from typing import Tuple, List
import math
import torch
import dgl
from dgl import DGLGraph
import dgl.function as fn
import torch_scatter
from ms_pred import common
import numpy as np


def connected_components(edge_index: torch.LongTensor,
                         num_nodes: int) -> torch.LongTensor:
    r"""
    GPU‐only connected‐components via label‐propagation.

    Starting with each node’s label = its 1‐based index, repeatedly
    lets each node adopt the minimum label of its neighbors until
    convergence.  Finally compresses labels to a 0‐based range.

    Args:
      edge_index (LongTensor[2, E]):
        Undirected GPU edge list, with shape (2, num_edges).
      num_nodes (int):
        Total number of nodes in the graph.

    Returns:
      LongTensor[num_nodes]:
        Component ID for each node, in the range [0 .. C], where
        C = number of connected components, 0 means isolated
        components.
    """
    device = edge_index.device
    src, dst = edge_index

    # 1-based initial labels ensure that 0 can be reserved for “isolated” nodes
    labels = torch.arange(1, num_nodes + 1, device=device)

    # Propagate the minimum neighbor label until stable
    while True:
        lbl_src = labels[src]
        lbl_dst = labels[dst]
        min_to_dst, _ = torch_scatter.scatter_min(
            lbl_src, dst, dim=0, dim_size=num_nodes
        )
        min_to_src, _ = torch_scatter.scatter_min(
            lbl_dst, src, dim=0, dim_size=num_nodes
        )
        best = torch.minimum(min_to_dst, min_to_src)
        new_labels = torch.minimum(labels, best)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels

    # degree to find isolated nodes
    deg = torch.zeros(num_nodes, device=device, dtype=torch.long)
    deg.scatter_add_(0, src, torch.ones_like(src))
    deg.scatter_add_(0, dst, torch.ones_like(dst))
    iso = (deg == 0)

    # compress non-isolated labels to 1..C, isolated -> 0
    non_iso_labels = labels[~iso]
    uniq, inv = torch.unique(non_iso_labels, return_inverse=True)
    out = torch.zeros(num_nodes, device=device, dtype=torch.long)
    out[~iso] = inv + 1  # shift so valid comps are 1..C
    # isolated already 0

    return out


def batch_remove_single_atoms(
    frag_batch: dgl.DGLGraph,
    batch_idx:  torch.LongTensor,   # [K], each in [0..B-1]
    sel_idx:    torch.LongTensor,   # [K], node index within that graph
    map_info:   dict,               # {key: [K]}
):
    """
    For K candidate removals across a batch of B graphs, remove exactly one
    atom per candidate, find the resulting connected components, and
    aggregate “broken‐bond” statistics per fragment.

    Args:
      frag_batch (DGLGraph):
        A batched DGLGraph of B input graphs, produced by dgl.batch.
      batch_idx (LongTensor[K]):
        For each of the K removals, which graph (0..B-1) it applies to.
      sel_idx (LongTensor[K]):
        For each removal, the node index *within* that graph to delete.
      map_info (dict(key: Tensor[K])):
        mapping information

    Returns:
      subg (DGLGraph):
        A batched DGLGraph of all resulting fragments (connected components),
        with internal batch indices set so that dgl.unbatch(subg) splits
        into one graph per fragment.
      broken_bonds (LongTensor[F]):
        Number of cut‐bonds (edges severed) falling into each fragment
        F = total number of fragments.
      ori_batch_id (LongTensor[F]):
        For each fragment, the original graph index (0..B-1) from which it arose.
      mapped_info (dict(key: FloatTensor[F])):
        For each fragment, the original value from map_info.

    Pipeline:
      1. Unbatch the B graphs and select the K target graphs.
      2. Re-batch those K graphs into one “job_batch.”
      3. Build a flat removal mask over all ∑ₖ Nₖ nodes.
      4. Detect all “cut” edges (one end removed, one end kept) and record:
         - removed_node, kept_node (global IDs in job_batch)
         - bond_types via job_batch.edata['e_ind']
      5. Induce the kept‐node subgraph `subg`.
      6. Compute `comp_flat` = connected component ID per node.
      7. Slice out `comp_for_cut` for each cut‐bond (before any reordering).
      8. Remove isolated (component=0) nodes if needed, shift IDs to start at 0.
      9. Permute `subg` so that nodes are grouped by component, then
         set `batch_num_nodes` and `batch_num_edges` so that
         `dgl.unbatch(subg)` works correctly.
     10. Build a flat `cut_info` dictionary (internally) with fields:
         job, batch, removed_node, kept_node, bond_type, comp_id.
     11. Aggregate per-component using `scatter_reduce_` to compute:
         - broken_bonds: total cut‐bonds per fragment
         - ori_batch_id: max original batch_idx per fragment
         - ori_sel_prob: max sel_prob per fragment

    Notes:
      - We use a 1-based label initialization in `connected_components`
        so that the value 0 can be reserved for “unconnected” if you choose to
        filter them out.  We then shift the returned labels down to 0-based.
      - All operations remain on the GPU: DGL subgraph, label‐propagation,
        and torch.scatter‐based reduction.
    """
    device = frag_batch.device
    B, K = frag_batch.batch_size, batch_idx.size(0)

    # 1) Generate a batch of K job graphs
    job_batch, sizes, offsets = slice_batched_graph(frag_batch, batch_idx)

    # 2) Build removal mask
    global_removed = offsets + sel_idx
    N_tot = job_batch.num_nodes()
    rm_flat = torch.zeros(N_tot, dtype=torch.bool, device=device)
    rm_flat[global_removed] = True

    # 3) Detect cut‐edges and record their ends & bond types
    src, dst   = job_batch.edges(order='eid')
    is_cut     = rm_flat[src] ^ rm_flat[dst]
    cut_eids   = torch.nonzero(is_cut, as_tuple=True)[0]
    src_c, dst_c = src[cut_eids], dst[cut_eids]
    rm_src       = rm_flat[src_c]
    removed_nodes = torch.where(rm_src, src_c, dst_c)
    kept_nodes    = torch.where(rm_src, dst_c, src_c)
    bond_types    = job_batch.edata['e_ind'][cut_eids]

    # 4) Induce the kept‐node subgraph
    subg = dgl.node_subgraph(job_batch, ~rm_flat)

    # 5) Capture orig_ids → local mapping before any reordering
    orig_ids = subg.ndata[dgl.NID]
    inv_map  = torch.full((N_tot,), -1, dtype=torch.long, device=device)
    inv_map[orig_ids] = torch.arange(orig_ids.numel(), device=device)
    kept_local = inv_map[kept_nodes]

    # 6) Compute connected‐components
    u_sub, v_sub = subg.edges(order='eid')
    comp_flat = connected_components(torch.stack([u_sub, v_sub], 0), num_nodes=subg.num_nodes())

    # 7) Extract comp_for_cut and filter out comp_flat==0 if desired
    comp_for_cut = comp_flat[kept_local]
    keep_comp_mask = (comp_flat > 0) & (torch.isin(comp_flat, comp_for_cut))  # if not comp_flat in comp_for_cut, is isolated in the input graph
    subg = dgl.node_subgraph(subg, keep_comp_mask)
    rm_flat[~rm_flat] = ~keep_comp_mask
    comp_flat = comp_flat[keep_comp_mask] - 1

    if len(comp_flat) == 0:  # no connected components
        return None, None, None

    # update comp_for_cut and comp_flat to continuous values
    uniq_comp_for_cut_vals, comp_for_cut = torch.unique(comp_for_cut, return_inverse=True)
    if uniq_comp_for_cut_vals.min() > 0:  # if there's no single atoms in this batch
        comp_for_cut += 1
    comp_flat    = torch.unique(comp_flat, return_inverse=True)[1]
    # now comp_flat ∈ [0..F-1]

    # 8) Reorder & set batch sizes so unbatch(subg) works
    perm = torch.argsort(comp_flat)
    subg = dgl.reorder_graph(subg, 'custom', permute_config={'nodes_perm': perm})
    comp_flat = comp_flat[perm]
    batched_num_nodes = torch.bincount(comp_flat)
    comp_per_edge     = comp_flat[subg.edges(order='eid')[0]]
    batched_num_edges = torch.bincount(comp_per_edge)
    subg.set_batch_num_nodes(batched_num_nodes)
    subg.set_batch_num_edges(batched_num_edges)

    # 9) Build flat cut_info for internal aggregation
    job_idx_map = torch.repeat_interleave(torch.arange(K, device=device), sizes)

    # 10) Aggregate per‐fragment via scatter_reduce
    def scatter_aggregate(src, reduce='sum'):
        out = torch.full((comp_flat.max() + 2,), -1, device=device, dtype=src.dtype)
        out.scatter_reduce_(0, comp_for_cut, src, reduce, include_self=False)
        return out[1:]  # ignore zero‐index entries

    broken_bonds = scatter_aggregate(bond_types, reduce='max').float()
    # NOTE: for ring breaking
    #       _____
    #      /     \
    #     A       B   ← Removing B yields broken_bonds = 1
    #      \     /
    #       ‾‾‾‾‾
    # if you want broken_bonds = 2 in this case, use
    # broken_bonds = scatter_aggregate(bond_types, reduce='sum').float() // 2
    mapped_info = {}
    for key, info in map_info.items():
        mapped_info[key] = scatter_aggregate(info[job_idx_map[kept_nodes]], reduce='max')

    return subg, broken_bonds, mapped_info


def slice_batched_graph(
    bg: DGLGraph,
    batch_idx: torch.LongTensor
) -> Tuple[DGLGraph, torch.Tensor, torch.Tensor]:
    """
    Replicate and reorder fragments in a batched graph.

    Args:
        bg: Batched DGLGraph containing M fragments.
        batch_idx: LongTensor of shape (K,) with indices (0 ≤ idx < M) of fragments to replicate.

    Returns:
        new_bg : DGLGraph containing the K requested fragments (with duplicates).
        sizes  : (K,) long tensor giving node‐counts per new fragment.
        offsets: (K,) long tensor of cumulative node‐offsets for concatenation.
    """
    device = batch_idx.device

    # 1) Original per‐fragment node‐ & edge‐counts
    nn_t = bg.batch_num_nodes()  # (M,)
    ne_t = bg.batch_num_edges()  # (M,)

    # 2) Compute old‐fragment node‐offsets for relabeling
    old_node_off = torch.cat([torch.tensor([0], device=device), nn_t[:-1].cumsum(0)], dim=0)  # (M,)
    old_edge_off = torch.cat([torch.tensor([0], device=device), ne_t[:-1].cumsum(0)], dim=0)  # (M,)

    # 3) Decide sizes & offsets of the *new* batch
    sizes     = nn_t[batch_idx]  # (K,)
    offsets   = torch.cat([torch.tensor([0], device=device), sizes[:-1].cumsum(0)], dim=0)  # (K,)
    sizes_e   = ne_t[batch_idx]  # (K,)
    offsets_e = torch.cat([torch.tensor([0], device=device), sizes_e[:-1].cumsum(0)], dim=0)  # (K,)

    # 4) NODE replication: pick out & relabel all h‐features and n_id’s
    old_node_idx = torch.arange(sizes.sum().item(), device=device) - torch.repeat_interleave(offsets, sizes) \
                   + old_node_off[batch_idx].repeat_interleave(nn_t[batch_idx])
    new_h        = bg.ndata['h'][old_node_idx]
    new_n_id     = bg.ndata['n_id'][old_node_idx]

    # 5) EDGE replication: similarly mask and remap endpoints + edge data
    src, dst = bg.edges()
    edge_pos = torch.arange(sizes_e.sum().item(), device=device) - torch.repeat_interleave(offsets_e, sizes_e) \
               + old_edge_off[batch_idx].repeat_interleave(ne_t[batch_idx])
    edge_j = torch.arange(len(batch_idx), device=device).repeat_interleave(ne_t[batch_idx])

    old_src   = src[edge_pos]
    old_dst   = dst[edge_pos]
    old_frag  = batch_idx[edge_j]

    new_src = (old_src - old_node_off[old_frag]) + offsets[edge_j]
    new_dst = (old_dst - old_node_off[old_frag]) + offsets[edge_j]

    new_e     = bg.edata['e'][edge_pos]
    new_e_ind = bg.edata['e_ind'][edge_pos]

    # 6) Build the new batched graph
    new_bg = dgl.graph((new_src, new_dst), num_nodes=new_h.size(0))
    new_bg.ndata['h']    = new_h
    new_bg.ndata['n_id'] = new_n_id
    new_bg.edata['e']    = new_e
    new_bg.edata['e_ind']= new_e_ind

    # 7) Fix up batch metadata
    new_bg.set_batch_num_nodes(sizes.tolist())
    new_bg.set_batch_num_edges(ne_t[batch_idx].tolist())

    return new_bg, sizes, offsets


def batched_mask_subgraph(batched_graphs, mask, nnodes_new):
    ngraphs = batched_graphs.batch_size
    device = batched_graphs.device

    assert torch.sum(mask) == torch.sum(nnodes_new)

    new_graphs = dgl.node_subgraph(batched_graphs, mask)

    # 1) build a “node → fragment‐ID” map of shape [N']
    newgraph_ids = torch.arange(ngraphs, device=device).repeat_interleave(nnodes_new)

    # 2) pull out all edges (directed) in the subgraph
    u, v = new_graphs.edges()

    # 3) find which edges lie wholly inside a single fragment
    f_u = newgraph_ids[u]
    f_v = newgraph_ids[v]
    mask = (f_u == f_v)

    # 4) bin‐count the fragment‐IDs of those edges
    edge_frag_ids = f_u[mask]  # only intra‐frag edges
    num_edges_per_frag = torch.bincount(edge_frag_ids, minlength=ngraphs)

    # 5) set params back on your batched graph
    new_graphs.set_batch_num_nodes(nnodes_new)
    new_graphs.set_batch_num_edges(num_edges_per_frag)

    return new_graphs


def dec2bin(x, bits):
    """
    Convert unsigned integers to a binary representation of length `bits`.
    """
    # Torch branch
    if torch.is_tensor(x):
        mask = 2 ** torch.arange(bits, device=x.device, dtype=x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).bool()

    # NumPy branch
    elif isinstance(x, np.ndarray):
        mask = 2 ** np.arange(bits, dtype=x.dtype)
        # x[..., None] has shape (..., 1); mask has shape (bits,)
        return (x[..., None] & mask) != 0

    else:
        raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or numpy.ndarray")


def bin2dec(x, bits=None, dtype=torch.long):
    """
    Convert binary vectors back to unsigned integers.
    `x` is (..., bits) of booleans or 0/1 ints.
    """
    # Torch branch
    if torch.is_tensor(x):
        if bits is None:
            bits = x.shape[-1]
        powers = 2 ** torch.arange(bits, device=x.device, dtype=dtype)
        return torch.sum(x * powers, dim=-1)

    # NumPy branch
    elif isinstance(x, np.ndarray):
        if bits is None:
            bits = x.shape[-1]
        powers = 2 ** np.arange(bits, dtype=dtype)
        return np.sum(x * powers, axis=-1)

    else:
        raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or numpy.ndarray")


def encode_bin_to_uint8(x, bits=None):
    """
    Pack a binary array (..., bits) into bytes (..., n_bytes).
    """
    # Torch branch
    if torch.is_tensor(x):
        if bits is None:
            bits = x.shape[-1]
        n_bytes = math.ceil(bits / 8)
        parts = [
            bin2dec(x[..., i*8:(i+1)*8], dtype=torch.uint8)
            for i in range(n_bytes)
        ]
        return torch.stack(parts, dim=-1)

    # NumPy branch
    elif isinstance(x, np.ndarray):
        if bits is None:
            bits = x.shape[-1]
        n_bytes = math.ceil(bits / 8)
        parts = [
            bin2dec(x[..., i*8:(i+1)*8], dtype=np.uint8)
            for i in range(n_bytes)
        ]
        return np.stack(parts, axis=-1)

    else:
        raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or numpy.ndarray")


def decode_bin_from_uint8(x, bits):
    """
    Unpack bytes (..., n_bytes) back into a binary array (..., bits).
    """
    # Torch branch
    if torch.is_tensor(x):
        bins = dec2bin(x, 8)  # gives (..., n_bytes, 8)
        bins = bins.reshape(x.shape[:-1] + (-1,))  # collapse the last two dims
        return bins[..., :bits]

    # NumPy branch
    elif isinstance(x, np.ndarray):
        bins = dec2bin(x, 8)  # (..., n_bytes, 8), dtype=bool
        bins = bins.reshape(x.shape[:-1] + (-1,))
        return bins[..., :bits]

    else:
        raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or numpy.ndarray")


def form_vec_to_mass(node_feat):
    device, dtype = node_feat.device, node_feat.dtype
    mass_vec = torch.tensor(common.CHEM_MASSES, device=device, dtype=dtype)
    # if embed_elem_group:
    #     mass_vec = torch.cat((mass_vec, torch.zeros((common.ELEMENT_GROUP_DIM, 1), device=device, dtype=dtype)), 0)
    # mass_vec = torch.cat((mass_vec, torch.arange(common.MAX_H, device=device) * common.ELEMENT_TO_MASS["H"]))
    masses = node_feat[:, :mass_vec.shape[0]] @ mass_vec
    return masses.squeeze(-1)


def frag_to_form_vec(frag_graph, add_hs, embed_elem_group):
    frag_h = frag_graph.ndata['h']
    frag_graph.ndata['__h_heavy'] = frag_h[:, :common.CHEM_ELEMENT_NUM]
    frag_form_vecs = dgl.sum_nodes(frag_graph, '__h_heavy')
    if add_hs:
        start_id, end_id = common.CHEM_ELEMENT_NUM, common.CHEM_ELEMENT_NUM + common.MAX_H
        if embed_elem_group:
            start_id, end_id = start_id + common.ELEMENT_GROUP_DIM, end_id + common.ELEMENT_GROUP_DIM
        node_h = torch.sum(frag_h[:, start_id:end_id] * torch.arange(common.MAX_H, device=frag_graph.device),
                           dim=1, keepdim=True)
        frag_graph.ndata['__h_H'] = node_h  # (total_nodes, 1)
        h_per_graph = dgl.sum_nodes(frag_graph, '__h_H').squeeze(1)  # (num_graphs,)
        frag_form_vecs[:, common.element_to_ind["H"]] = h_per_graph
        del frag_graph.ndata['__h_H']
    del frag_graph.ndata['__h_heavy']
    return frag_form_vecs


def np_like_unique(x, dim=-1):
    """
    torch.unique returns all occurrences in inverse_index, while np.unique returns the first occurrence
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-1072093200
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    rev_idx = inverse.new_full((unique.size(dim),), perm.max()+1).scatter_reduce_(dim, inverse, perm, reduce='min') # always return the first occurrence
    return unique, rev_idx


def msg_passing_frag_graph_hash(graph, feat_dim=32):
    """
    Get hash of a fragmentation graph by message passing
    """
    # change data type to float
    def e_to_float(edges):
        return {'e_ind_float': edges.data['e_ind'].to(dtype=torch.float32)}
    graph.apply_edges(e_to_float)
    # def n_trim(nodes):
    #     return {'h_new': nodes.data['h'][:, :feat_dim]}
    # graph.apply_nodes(n_trim)

    # 1 message passing step
    graph.update_all(fn.u_mul_e('h', 'e_ind_float', 'm'), fn.sum('m', 'h_new'))

    # sum pooling
    hash = dgl.sum_nodes(graph, 'h_new')

    del graph.ndata['h_new']
    del graph.edata['e_ind_float']

    return hash
