from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Size, OptTensor
from torch_geometric.utils import softmax
class MMGAT(MessagePassing):

    _alpha: OptTensor

    def __init__(
            self, 
            in_channels :int,
            out_channels :int,
            num_node_types: int,
            num_edge_types: int,
            heads: int = 1,
            concat: bool = True,
            negative_slopes: float = 0.2,
            dropout: float = 0.0,
            bias: bool = True,
            **kwargs):  
        
        super(MMGAT, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.concat = concat
        self.negative_slopes = negative_slopes
        self.dropout = dropout
        self.bias = bias

        self.lin_l = Parameter(Tensor(num_node_types, in_channels, heads * out_channels))
        self.lin_r = Parameter(Tensor(num_node_types, in_channels, heads * out_channels))

        self.att = Parameter(Tensor(num_edge_types, 1, heads, out_channels))


        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.lin_l)
        glorot(self.lin_r)
        glorot(self.att)
        zeros(self.bias)

    def forward(
            self, 
            x: Tensor,
            x_type: Tensor,
            edge_index: Tensor,
            edge_type_lengths: Tensor,
            size: Size = None,
            return_attention_weights: bool = False):
        
        H, C = self.heads, self.out_channels
        N = x.size(0)
        type_len = N // 3

        x_l: OptTensor = torch.zeros(N, H * C).to(x.device)
        x_r: OptTensor = torch.zeros(N, H * C).to(x.device)

        for j in range(self.num_node_types):
            x_l[j * type_len: (j + 1) * type_len] = \
                torch.einsum("ij,jk->ik",
                             x[j * type_len: (j + 1) * type_len], 
                             self.lin_l[j])
            
            x_r[j * type_len: (j + 1) * type_len] = \
                torch.einsum("ij,jk->ik",
                             x[j * type_len: (j + 1) * type_len], 
                             self.lin_r[j])
            
        x_l = x_l.view(-1, H, C)
        x_r = x_r.view(-1, H, C)
        assert x_l is not None
        assert x_r is not None

        out = self.propagate(edge_index=edge_index, 
                             x = (x_l, x_r), 
                             edge_type_lengths = edge_type_lengths,
                             size=size)
        
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out
    
    def message(self,
                x_j: Tensor, 
                x_i: Tensor,
                edge_type_lengths: Tensor,
                index: Tensor, 
                ptr: OptTensor,
                size_i: Optional[int]):
        x = x_j + x_i

        x = F.leaky_relu(x, negative_slope=self.negative_slopes)
        alpha = torch.zeros((x.size(0), self.heads)).to(x.device)

        sum_lengths = 0
        for j in range(self.num_edge_types):
            cur_len = edge_type_lengths[j].item()

            alpha_j = (x[sum_lengths: sum_lengths + cur_len] * self.att[j]).sum(-1)
            alpha[sum_lengths: sum_lengths + cur_len] = alpha_j
            
            sum_lengths += cur_len
        
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
        


        






        



