# Implementation was referred to learner.py from
# @misc{MAML_Pytorch,
#   author = {Liangqu Long},
#   title = {MAML-Pytorch Implementation},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/dragen1860/MAML-Pytorch}},
#   commit = {master}
# }

from builtins import NotImplementedError
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class GNNEncoder(nn.Module):
    def __init__(self, config):
        super(GNNEncoder, self).__init__()

        self.config = config
        # containing parameters of GNN Encoder
        self.vars = nn.ParameterList()

        for name, size, bias_ in self.config:
            if name == 'GCN':
                weight = Parameter(torch.ones(*size))
                torch.nn.init.xavier_uniform_(weight)
                self.vars.append(weight)
                if bias_ is True:
                    bias = Parameter(torch.zeros(size[1]))
                    self.vars.append(bias)
                else:
                    bias = None
                    self.vars.append(bias)

    def forward(self, x, vars=None, adj=None):

        if vars is None:
            vars = self.vars
        
        idx = 0
        for name, _, _ in self.config:
            # 2*i th: weight of ith layer
            # 2*i+1 th: bias of ith layer if exists
            if name == 'GCN':
                weight, bias = vars[2*idx], vars[2*idx+1]
                x = torch.mm(x, weight)
                x = torch.sparse.mm(adj, x)
                if bias is not None:
                    x = x + bias
                x = F.relu(x)
                idx += 1
            else:
                continue
        return x

    def parameters(self):

        return self.vars



class LinearClassifier(nn.Module):
    
    def __init__(self, config) -> None:
        super(LinearClassifier, self).__init__()
        self.config = config

        self.vars = nn.ParameterList()
        
        for name, size, bias_ in self.config:
        # 'linear': linear classifier using node embeddings embedded by GNN Encoders
        # size = (in_feat, out_feat) if name == 'linear'
            if name == 'linear':
                weight = Parameter(torch.ones(*size))
                torch.nn.init.xavier_uniform_(weight)
                self.vars.append(weight)
                if bias_ is True:
                    bias = Parameter(torch.zeros(size[1]))
                    self.vars.append(bias)
                else:
                    bias = None
                    self.vars.append(bias)
            else:
                continue
    
    def forward(self, x, vars=None):

        if vars == None:
            vars = self.vars
        
        idx = 0
        for name, _, _ in self.config:
            if name == 'linear':
                weight, bias = vars[2*idx], vars[2*idx+1]
                x = torch.mm(x, weight)
                if bias is not None:
                    x = x + bias
                idx += 1
            else:
                continue
        
        return x


    def parameters(self):

        return self.vars
        



        
class GCN(nn.Module):
    
    def __init__(self, config) -> None:
        super(GCN, self).__init__()
        
        self.config = config
        # containing parameters of GNN Encoder
        self.vars = nn.ParameterList()

        for name, size, bias_ in self.config:
            if name == 'GCN':
                weight = Parameter(torch.ones(*size))
                torch.nn.init.xavier_uniform_(weight)
                self.vars.append(weight)
                if bias_ is True:
                    bias = Parameter(torch.zeros(size[1]))
                    self.vars.append(bias)
                else:
                    bias = None
                    self.vars.append(bias)

            else:
                continue

    
    def forward(self, x, vars=None, adj=None):
        
        if vars is None:
            vars = self.vars

        idx = 0
        
        for name, _, _, in self.config:
            if name == 'GCN':
                weight, bias = vars[2*idx], vars[2*idx+1]
                x = torch.mm(x, weight)
                x = torch.sparse.mm(adj, x)
                if bias is not None:
                    x = x + bias
                idx += 1
            elif name == 'relu':
                x = F.relu(x)
            elif name == 'dropout':
                x = F.dropout(x)
            elif name == 'linear':
                continue

        return x

    def parameters(self):

        return self.vars


