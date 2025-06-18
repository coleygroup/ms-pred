""" dgl_modules.

Directly copy dgl modules to patch them

"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn


from dgl.nn import expand_as_pair
import dgl.nn as dgl_nn
import dgl

gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")


class GatedGraphConv(nn.Module):
    r"""Gated Graph Convolution layer from `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__

    .. math::
        h_{i}^{0} &= [ x_i \| \mathbf{0} ]

        a_{i}^{t} &= \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} &= \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`x_i`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(t+1)}`.
    n_steps : int
        Number of recurrent steps; i.e, the :math:`t` in the above formula.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GatedGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = torch.ones(6, 10)
    >>> conv = GatedGraphConv(10, 10, 2, 3)
    >>> etype = torch.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.4652,  0.4458,  0.5169,  0.4126,  0.4847,  0.2303,  0.2757,  0.7721,
            0.0523,  0.0857],
            [ 0.0832,  0.1388, -0.5643,  0.7053, -0.2524, -0.3847,  0.7587,  0.8245,
            0.9315,  0.4063],
            [ 0.6340,  0.4096,  0.7692,  0.2125,  0.2106,  0.4542, -0.0580,  0.3364,
            -0.1376,  0.4948],
            [ 0.5551,  0.7946,  0.6220,  0.8058,  0.5711,  0.3063, -0.5454,  0.2272,
            -0.6931, -0.1607],
            [ 0.2644,  0.2469, -0.6143,  0.6008, -0.1516, -0.3781,  0.5878,  0.7993,
            0.9241,  0.1835],
            [ 0.6393,  0.3447,  0.3893,  0.4279,  0.3342,  0.3809,  0.0406,  0.5030,
            0.1342,  0.0425]], grad_fn=<AddBackward0>)
    """

    def __init__(self, in_feats, out_feats, n_steps, n_etypes, bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = init.calculate_gain("relu")
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, etypes=None):
        """

        Description
        -----------
        Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor, or None
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph. When there's only one edge type,
            this argument can be skipped

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            assert graph.is_homogeneous, (
                "not a homogeneous graph; convert it with to_homogeneous "
                "and pass in the edge type as argument"
            )
            if self._n_etypes != 1:
                assert (
                    etypes.min() >= 0 and etypes.max() < self._n_etypes
                ), "edge type indices out of range [0, {})".format(self._n_etypes)

            zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
            feat = torch.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                if self._n_etypes == 1 and etypes is None:
                    # Fast path when graph has only one edge type
                    graph.ndata["h"] = self.linears[0](feat)
                    graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "a"))
                    a = graph.ndata.pop("a")  # (N, D)
                else:
                    graph.ndata["h"] = feat
                    for i in range(self._n_etypes):
                        eids = (
                            torch.nonzero(etypes == i, as_tuple=False)
                            .contiguous()
                            .view(-1)
                            .type(graph.idtype)
                        )
                        if len(eids) > 0:
                            graph.apply_edges(
                                lambda edges: {
                                    "W_e*h": self.linears[i](edges.src["h"])
                                },
                                eids,
                            )
                    graph.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
                    a = graph.ndata.pop("a")  # (N, D)
                feat = self.gru(a, feat)
            return feat


def aggregate_mean(h):
    """mean aggregation"""
    return torch.mean(h, dim=1)


def aggregate_max(h):
    """max aggregation"""
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    """min aggregation"""
    return torch.min(h, dim=1)[0]


def aggregate_sum(h):
    """sum aggregation"""
    return torch.sum(h, dim=1)


def aggregate_std(h):
    """standard deviation aggregation"""
    return torch.sqrt(aggregate_var(h) + 1e-30)


def aggregate_var(h):
    """variance aggregation"""
    h_mean_squares = torch.mean(h * h, dim=1)
    h_mean = torch.mean(h, dim=1)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def _aggregate_moment(h, n):
    """moment aggregation: for each node (E[(X-E[X])^n])^{1/n}"""
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=1)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + 1e-30, 1.0 / n)
    return rooted_h_n


def aggregate_moment_3(h):
    """moment aggregation with n=3"""
    return _aggregate_moment(h, n=3)


def aggregate_moment_4(h):
    """moment aggregation with n=4"""
    return _aggregate_moment(h, n=4)


def aggregate_moment_5(h):
    """moment aggregation with n=5"""
    return _aggregate_moment(h, n=5)


def scale_identity(h):
    """identity scaling (no scaling operation)"""
    return h


def scale_amplification(h, D, delta):
    """amplification scaling"""
    return h * (np.log(D + 1) / delta)


def scale_attenuation(h, D, delta):
    """attenuation scaling"""
    return h * (delta / np.log(D + 1))


AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": aggregate_moment_3,
    "moment4": aggregate_moment_4,
    "moment5": aggregate_moment_5,
}
SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class PNAConvTower(nn.Module):
    """A single PNA tower in PNA layers"""

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        dropout=0.0,
        edge_feat_size=0,
    ):
        super(PNAConvTower, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.edge_feat_size = edge_feat_size

        self.M = nn.Linear(2 * in_size + edge_feat_size, in_size)
        self.U = nn.Linear((len(aggregators) * len(scalers) + 1) * in_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_size)

    def reduce_func(self, nodes):
        """reduce function for PNA layer:
        tensordot of multiple aggregation and scaling operations"""
        msg = nodes.mailbox["msg"]
        degree = msg.size(1)
        h = torch.cat([AGGREGATORS[agg](msg) for agg in self.aggregators], dim=1)
        h = torch.cat(
            [
                SCALERS[scaler](h, D=degree, delta=self.delta)
                if scaler != "identity"
                else h
                for scaler in self.scalers
            ],
            dim=1,
        )
        return {"h_neigh": h}

    def message(self, edges):
        """message function for PNA layer"""
        if self.edge_feat_size > 0:
            f = torch.cat([edges.src["h"], edges.dst["h"], edges.data["a"]], dim=-1)
        else:
            f = torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
        return {"msg": self.M(f)}

    def forward(self, graph, node_feat, edge_feat=None):
        """compute the forward pass of a single tower in PNA convolution layer"""
        # calculate graph normalization factors
        snorm_n = torch.cat(
            [torch.ones(N, 1).to(node_feat) / N for N in graph.batch_num_nodes()], dim=0
        ).sqrt()
        with graph.local_scope():
            graph.ndata["h"] = node_feat
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat

            graph.update_all(self.message, self.reduce_func)
            h = self.U(torch.cat([node_feat, graph.ndata["h_neigh"]], dim=-1))
            h = h * snorm_n
            return self.dropout(self.batchnorm(h))


class PNAConv(nn.Module):
    r"""Principal Neighbourhood Aggregation Layer from `Principal Neighbourhood Aggregation
    for Graph Nets <https://arxiv.org/abs/2004.05718>`__

    A PNA layer is composed of multiple PNA towers. Each tower takes as input a split of the
    input features, and computes the message passing as below.

    .. math::
        h_i^(l+1) = U(h_i^l, \oplus_{(i,j)\in E}M(h_i^l, e_{i,j}, h_j^l))

    where :math:`h_i` and :math:`e_{i,j}` are node features and edge features, respectively.
    :math:`M` and :math:`U` are MLPs, taking the concatenation of input for computing
    output features. :math:`\oplus` represents the combination of various aggregators
    and scalers. Aggregators aggregate messages from neighbours and scalers scale the
    aggregated messages in different ways. :math:`\oplus` concatenates the output features
    of each combination.

    The output of multiple towers are concatenated and fed into a linear mixing layer for the
    final output.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    aggregators : list of str
        List of aggregation function names(each aggregator specifies a way to aggregate
        messages from neighbours), selected from:

        * ``mean``: the mean of neighbour messages

        * ``max``: the maximum of neighbour messages

        * ``min``: the minimum of neighbour messages

        * ``std``: the standard deviation of neighbour messages

        * ``var``: the variance of neighbour messages

        * ``sum``: the sum of neighbour messages

        * ``moment3``, ``moment4``, ``moment5``: the normalized moments aggregation
        :math:`(E[(X-E[X])^n])^{1/n}`
    scalers: list of str
        List of scaler function names, selected from:

        * ``identity``: no scaling

        * ``amplification``: multiply the aggregated message by :math:`\log(d+1)/\delta`,
        where :math:`d` is the degree of the node.

        * ``attenuation``: multiply the aggregated message by :math:`\delta/\log(d+1)`
    delta: float
        The degree-related normalization factor computed over the training set, used by scalers
        for normalization. :math:`E[\log(d+1)]`, where :math:`d` is the degree for each node
        in the training set.
    dropout: float, optional
        The dropout ratio. Default: 0.0.
    num_towers: int, optional
        The number of towers used. Default: 1. Note that in_size and out_size must be divisible
        by num_towers.
    edge_feat_size: int, optional
        The edge feature size. Default: 0.
    residual : bool, optional
        The bool flag that determines whether to add a residual connection for the
        output. Default: True. If in_size and out_size of the PNA conv layer are not
        the same, this flag will be set as False forcibly.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import PNAConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = torch.ones(6, 10)
    >>> conv = PNAConv(10, 10, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)
    >>> ret = conv(g, feat)
    """

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        dropout=0.0,
        num_towers=1,
        edge_feat_size=0,
        residual=True,
    ):
        super(PNAConv, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        assert in_size % num_towers == 0, "in_size must be divisible by num_towers"
        assert out_size % num_towers == 0, "out_size must be divisible by num_towers"
        self.tower_in_size = in_size // num_towers
        self.tower_out_size = out_size // num_towers
        self.edge_feat_size = edge_feat_size
        self.residual = residual
        if self.in_size != self.out_size:
            self.residual = False

        self.towers = nn.ModuleList(
            [
                PNAConvTower(
                    self.tower_in_size,
                    self.tower_out_size,
                    aggregators,
                    scalers,
                    delta,
                    dropout=dropout,
                    edge_feat_size=edge_feat_size,
                )
                for _ in range(num_towers)
            ]
        )

        self.mixing_layer = nn.Sequential(nn.Linear(out_size, out_size), nn.LeakyReLU())

    def forward(self, graph, node_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute PNA layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            should be the same as out_size.
        """
        h_cat = torch.cat(
            [
                tower(
                    graph,
                    node_feat[
                        :, ti * self.tower_in_size : (ti + 1) * self.tower_in_size
                    ],
                    edge_feat,
                )
                for ti, tower in enumerate(self.towers)
            ],
            dim=1,
        )
        h_out = self.mixing_layer(h_cat)
        # add residual connection
        if self.residual:
            h_out = h_out + node_feat

        return h_out


class GINEConv(nn.Module):
    r"""Graph Isomorphism Network with Edge Features, introduced by
    `Strategies for Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \sum_{j\in\mathcal{N}(i)}\mathrm{ReLU}(h_j^{l} + e_{j,i}^{l})\right)

    where :math:`e_{j,i}^{l}` is the edge feature.

    Parameters
    ----------
    apply_func : callable module or None
        The :math:`f_\Theta` in the formula. If not None, it will be applied to
        the updated node features. The default value is None.
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.

    Examples
    --------

    >>> import dgl
    >>> import torch
    >>> import torch.nn as nn
    >>> from dgl.nn import GINEConv

    >>> g = dgl.graph(([0, 1, 2], [1, 1, 3]))
    >>> in_feats = 10
    >>> out_feats = 20
    >>> nfeat = torch.randn(g.num_nodes(), in_feats)
    >>> efeat = torch.randn(g.num_edges(), in_feats)
    >>> conv = GINEConv(nn.Linear(in_feats, out_feats))
    >>> res = conv(g, nfeat, efeat)
    >>> print(res.shape)
    torch.Size([4, 20])
    """

    def __init__(self, apply_func=None, init_eps=0, learn_eps=False):
        super(GINEConv, self).__init__()
        self.apply_func = apply_func
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def message(self, edges):
        r"""User-defined Message Function"""
        return {"m": F.relu(edges.src["hn"] + edges.data["he"])}

    def forward(self, graph, node_feat, edge_feat):
        r"""Forward computation.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it is the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input feature size requirement of ``apply_func``.
        edge_feat : torch.Tensor
            Edge feature. It is a tensor of shape :math:`(E, D_{in})` where :math:`E`
            is the number of edges.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output feature size of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as :math:`D_{in}`.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata["hn"] = feat_src
            graph.edata["he"] = edge_feat
            graph.update_all(self.message, fn.sum("m", "neigh"))
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata["h"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["h"]
            # return h
            return self.linear(h)

class MultiEdgeGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes):
        super().__init__()
        self.etype_layers = nn.ModuleDict({
            etype: dgl_nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True)
            for etype in etypes
        })

    def forward(self, g, inputs):
        # Input: node features (same for all nodes, since there's a single node type)
        g.ndata['h'] = inputs

        outputs = torch.zeros_like(inputs)

        with g.local_scope():
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                rel_graph.update_all(
                    dgl.function.copy_u('h', 'm'),
                    dgl.function.sum('m', 'h_agg'),
                    etype=etype
                )
                outputs += self.etype_layers[etype](rel_graph, rel_graph.ndata['h_agg'])
        
        return outputs
    
class MultiEdgeGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, etypes, conv_steps):
        super().__init__()
        self.layer1 = MultiEdgeGCNLayer(in_feats, hidden_feats, etypes)
        self.layer2 = MultiEdgeGCNLayer(hidden_feats, out_feats, etypes)
        self.conv_steps = conv_steps

    def forward(self, g, inputs):
        h = inputs
        for _ in range(self.conv_steps-1):
            h = self.layer1(g, h)
            h = F.relu(h)
        h = self.layer2(g, h)
        return h

class HyperGNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_conv, dropout=0):
        super(HyperGNN, self).__init__()
        self.layer = GCNLayer(hidden_size, hidden_size)
        self.layer_out = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout_conv = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

        self.num_conv = num_conv

    def forward(self, g, features):
        for _ in range(self.num_conv):
            features = self.layer(g, features)
            features = self.activation(features)
            features = self.dropout_conv(features)

        result=self.layer_out(features)
        result=self.dropout_output(result)

        return result
