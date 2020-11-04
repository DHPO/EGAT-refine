import torch_geometric as pyg
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from model.node import EGAT_base


class EGAT_wo_att(EGAT_base):
    def __init__(self, vertex_feature, edge_feature, vertex_feature_ratio, vertex_in_feature=None, edge_in_feature=None, heads=1, concat=True,
                 leaky=0.2, dropout=0, **kwargs):
        # parameter
        super(EGAT_wo_att, self).__init__(vertex_feature, edge_feature, heads, concat, leaky, dropout, **kwargs)
        self._vertex_feature_ratio = vertex_feature_ratio
        self._vertex_hidden_feature = int(self._vertex_feature_ratio * self._vertex_out_feature)
        self._edge_hidden_feature = self._vertex_out_feature - self._vertex_hidden_feature
        self._vertex_in_feature = vertex_in_feature if vertex_in_feature is not None else vertex_feature
        self._edge_in_feature = edge_in_feature if edge_in_feature is not None else edge_feature
        self.build_modules()

    def build_custom_modules(self):
        self.v_linear = nn.Linear(self._vertex_in_feature, self._vertex_hidden_feature, bias=False)
        self.e_linear = nn.Linear(self._edge_in_feature, self._edge_hidden_feature, bias=False)

    def reset_parameters(self):
        EGAT_base.reset_parameters(self)
        pyg.nn.inits.glorot(self.v_linear.weight)
        pyg.nn.inits.glorot(self.e_linear.weight)

    def M(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o):
        size = x_j.size(0)
        x_j = x_j.view((size, self._heads, -1))
        e_ij = e_ij.view((size, self._heads, -1))
        return torch.cat([x_j, e_ij], dim=-1), None

    def Matt(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o, Mtmp=None):
        size = x_j.size(0)
        x_i = x_i.view((size, self._heads, -1))
        x_j = x_j.view((size, self._heads, -1))
        e_ij = e_ij.view((size, self._heads, -1))
        return torch.cat([x_i, x_j, e_ij], dim=-1)

    def Vtransform(self, x):
        return self.v_linear(x)

    def Etransform(self, e):
        return self.e_linear(e)

    def custom_update(self, x_i, aggr_out):
        return aggr_out

    @property
    def _vertex_channel(self):
        return self._vertex_hidden_feature // self._heads

    @property
    def _m_att_channel(self):
        return (self._vertex_hidden_feature * 2 + self._edge_hidden_feature) // self._heads

    def message(self, edge_index_i, x_i, x_j, size_i, e, x_o_i, x_o_j, e_o, index, ptr):
        e_size = e.size(0)
        result = self.M(x_i, x_j, e, x_o_i, x_o_j, e_o)
        if self._apart:
            m, m_out, Mtmp = result
        else:
            m, m_out, Mtmp = result[0], result[0], result[1]
        # m_att = self.Matt(x_i, x_j, e, x_o_i, x_o_j, e_o, Mtmp)

        m = m.view(e_size, self._heads, -1)
        # alpha = self.attention(m_att, edge_index_i, size_i, index, ptr)

        if self._apart:
            m, m_out = m, m_out
            return torch.cat([m, m_out], dim=-1)
        else:
            return (m).view(e_size, -1)
