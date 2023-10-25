# Implementation mainly follows utils.py from
# @inproceedings{ding2020graph,
#   title={Graph prototypical networks for few-shot learning on attributed networks},
#   author={Ding, Kaize and Wang, Jianling and Li, Jundong and Shu, Kai and Liu, Chenghao and Liu, Huan},
#   booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
#   pages={295--304},
#   year={2020}
# }

from json import load
import os
import numpy as np
import scipy.sparse as sp
import torch
import random

import torch_geometric
import torch_geometric.transforms as T

import scipy.io as sio
import pickle

from copy import deepcopy




def graph_diffusion(args, Data):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    num_nodes = Data.features.size(0)
    # NaQ-Feat
    if args.type == 'feature':
        features = Data.features
        def cos_sim_matrix(feat1, feat2):
            feat1 = torch.nn.functional.normalize(feat1)
            feat2 = torch.nn.functional.normalize(feat2)
            cos_sim = torch.mm(feat1, feat2.mT)
            return cos_sim
        Diffusion_matrix = cos_sim_matrix(features, features)
    elif args.type == 'diffusion':
        # Calculation of Diffusion matrix
        gdc = T.GDC()
        edge_weight = torch.ones(Data.edge_index.size(1), device=Data.edge_index.device)
        edge_index, edge_weight = gdc.transition_matrix(edge_index=Data.edge_index, edge_weight=edge_weight, num_nodes=num_nodes, normalization='sym')
        Diffusion_matrix = gdc.diffusion_matrix_exact(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes, method='ppr', alpha=args.PPR_alpha)

    # Not to sample node itself in query-generation stage.
    Diffusion_matrix = sparse_fill_diagonal_0_(Diffusion_matrix) if Diffusion_matrix.layout == torch.sparse_coo else Diffusion_matrix.fill_diagonal_(0)
    
    return Diffusion_matrix.to(device)



def normalize(mx):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    '''Symmetrically normalize adjacency matrix.'''
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''Convert a scipy sparse matrix to a torch sparse tensor.'''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_tensor_to_sparse_mx(sparse_tensor):
    '''Convert a torch sparse tensor to a scipy sparse matrix'''
    sparse_tensor = sparse_tensor.cpu().detach().coalesce()
    row, col = sparse_tensor.indices()[0].numpy(), sparse_tensor.indices()[1].numpy()
    value = sparse_tensor.values().numpy()
    shape = (sparse_tensor.size(0), sparse_tensor.size(1))
    return sp.coo_matrix((value, (row, col)), shape=shape)

# Functions for handling (2-D) sparse tensors
def sparse_fill_diagonal_0_(sparse_tensor):
    coalesced = sparse_tensor.coalesce()
    indices = coalesced.indices()
    values = coalesced.values()
    remaining_indices = (~(indices[0,:] == indices[1,:])).nonzero().flatten()
    indices = indices[:,remaining_indices]
    values = values[remaining_indices]
    return torch.sparse_coo_tensor(indices=indices, values=values, size=sparse_tensor.size())

def sparse_tensor_column_zeroing(sparse_tensor, col_indices):
    '''
    This method can be modified to remove or indexing specified 'sub'-matrix
    '''
    coalesced = sparse_tensor.coalesce()
    indices = coalesced.indices()
    values = coalesced.values()
    for ix in col_indices:
        remaining_indices = (indices[1,:]!=ix).nonzero().flatten()
        indices = indices[:,remaining_indices]
        values = values[remaining_indices]
    return torch.sparse_coo_tensor(indices=indices, values=values, size=sparse_tensor.size())

def sparse_topk_dim1(sparse_tensor, row_ix, k):
    '''
    Row-wise Top-k function for sparse matrices
    Please keep len(row_ix) be small to get efficiency.
    '''
    if type(row_ix) != list:
        row_ix = row_ix.tolist()
    coalesced = sparse_tensor.coalesce()
    val_temp, ix_temp = [], []
    for ix in row_ix:
        vals, ixs = sparse_tensor[ix].to_dense().topk(k=k)
        val_temp.append(vals)
        ix_temp.append(ixs)
    values, indices = torch.stack(val_temp), torch.stack(ix_temp)
    return values, indices


def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
