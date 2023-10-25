# Implementation for MAML, UnsupMAML, mainly follows meta.py from
# @misc{MAML_Pytorch,
#   author = {Liangqu Long},
#   title = {MAML-Pytorch Implementation},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/dragen1860/MAML-Pytorch}},
#   commit = {master}
# # }
# For ProtoNet implementation, we referred to https://github.com/kaize0409/GPN_Graph-Few-shot
# @inproceedings{ding2020graph,
#   title={Graph prototypical networks for few-shot learning on attributed networks},
#   author={Ding, Kaize and Wang, Jianling and Li, Jundong and Shu, Kai and Liu, Chenghao and Liu, Huan},
#   booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
#   pages={295--304},
#   year={2020}
# }

import torch
import numpy as np
from torch import nn
from torch import optim

import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model import GNNEncoder, LinearClassifier, GCN
from utils import *
from copy import deepcopy

from sklearn.linear_model import LogisticRegression



class MAML(nn.Module):
    def __init__(self, args, config, Data):
        super(MAML, self).__init__()
        self.config = config
        self.Data = Data
        self.network = GNNEncoder(config)
        self.dim_latent = args.latent

        self.n = args.n_way
        self.k = args.k_shot
        self.q = args.q_query
        self.meta_batch_size = args.meta_batch_size
        self.num_steps_meta = args.num_steps_meta
        self.inner_lr = args.inner_lr
        self.meta_update_lr = args.meta_update_lr

        self.classifier = LinearClassifier(config) # used in Meta-training
        self.l2_penalty = args.l2_penalty # used in Fine-tuning
        
        self.meta_optimizer = optim.Adam(list(self.network.parameters())+list(self.classifier.parameters()), lr=self.meta_update_lr)


    def forward(self, id_spt, y_spt, id_query, y_query):
        # Load Features/Adjacency matrix
        features, adj = self.Data.features, self.Data.adj

        query_size = self.n*self.q
        num_cls_params = len(list(self.classifier.parameters())) # number of set of parameters of linear classifier
        if num_cls_params == 0:
            num_cls_params = -len(list(self.network.parameters()))

        # losses_query[j] = validation loss(loss of query) after jth update in inner-loop (j = 0, ..., self.num_steps_meta)
        losses_query = [0 for _ in range(self.num_steps_meta+1)]
        corrects = [0 for _ in range(self.num_steps_meta+1)]
    
        #---------------- <Meta-Training & Loss recording phase(Inner-loop)> ----------------#
        for i in range(self.meta_batch_size):
            
            # Get Loss & Spt embeddings for ith task for support samples before training
            encodings = self.network(features, vars=None, adj=adj)
            x_spt = encodings[id_spt[i]]
            logits = self.classifier(x_spt, vars=None)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, list(self.network.parameters())+list(self.classifier.parameters()))
            # 1st update of model parameters
            weights_updated = list(map(lambda f: f[1] - self.inner_lr*f[0], zip(grad, list(self.network.parameters())+list(self.classifier.parameters()))))

            # Get Query embeddings & Record loss and accuracy for query samples before the 1st update for the meta-update phase
            with torch.no_grad():
                x_query = encodings[id_query[i]]
                logits_query = self.classifier(x_query, vars=self.classifier.parameters())
                loss_query = F.cross_entropy(logits_query, y_query[i])
                losses_query[0] += loss_query
                
                pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                correct = torch.eq(pred_query, y_query[i]).sum().item()
                corrects[0] = corrects[0] + correct
            
            # Get Query embeddings & Record loss and accuracy after the 1st update
            encodings = self.network(features, vars=weights_updated[:-num_cls_params], adj=adj)
            x_query = encodings[id_query[i]]
            logits_query = self.classifier(x_query, vars=weights_updated[-num_cls_params:])
            loss_query = F.cross_entropy(logits_query, y_query[i])
            losses_query[1] += loss_query

            with torch.no_grad():
                pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                correct = torch.eq(pred_query, y_query[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for j in range(2, self.num_steps_meta+1):
                # (1) Get Spt embeddings & loss for ith task with weights after (j-1)th update (j = 2, ..., self.num_steps_meta)
                x_spt = encodings[id_spt[i]]
                logits = self.classifier(x_spt, vars=weights_updated[-num_cls_params:])
                loss = F.cross_entropy(logits, y_spt[i])
                # (2) Get gradient at current parameter
                grad = torch.autograd.grad(loss, weights_updated)
                # (3) jth update of model parameter
                weights_updated = list(map(lambda f: f[1] - self.inner_lr*f[0], zip(grad, weights_updated)))
                
                # (4) Record loss and accuracy after the jth update for the meta-update phase
                encodings = self.network(features, vars=weights_updated[:-num_cls_params], adj=adj)
                x_query = encodings[id_query[i]]
                logits_query = self.classifier(x_query, vars=weights_updated[-num_cls_params:])
                loss_query = F.cross_entropy(logits_query, y_query[i])
                losses_query[j] += loss_query

                with torch.no_grad():
                    pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_query, y_query[i]).sum().item()
                    corrects[j] = corrects[j] + correct
        #------------------------------------------------------------------------------------#

        #------------------------- <Meta Update Phase(Outer-loop)> --------------------------#
        # Use loss of query samples by using final updated parameter
        final_loss_query = losses_query[-1] / self.meta_batch_size

        # Meta Update
        self.meta_optimizer.zero_grad()
        final_loss_query.backward()
        self.meta_optimizer.step()

        # calculating training accuracy by using final updated parameter
        final_acc = corrects[-1] / (query_size*self.meta_batch_size)
        #------------------------------------------------------------------------------------#

        return final_loss_query, final_acc


    def fine_tuning(self, id_spt, y_spt, id_query, y_query):
        # Load Features/Adjacency matrix
        features, adj= self.Data.features, self.Data.adj

        assert id_spt.shape[0] != self.meta_batch_size

        # Fine-tune the copied model to prevent model learning in test phase
        net_GNN = deepcopy(self.network)
        
        # Get Embeddings
        encodings = net_GNN(features, vars=None, adj=adj).detach().cpu().numpy()
        x_spt, x_query, y_spt, y_query = encodings[id_spt], encodings[id_query], y_spt.detach().cpu().numpy(), y_query.detach().cpu().numpy()

        clf = LogisticRegression(max_iter=1000, C=1/self.l2_penalty).fit(x_spt, y_spt)

        # delete copied net that we used
        del net_GNN

        # Get Test Accuracy
        test_acc = clf.score(x_query, y_query)

        return None, test_acc


class ProtoNet(nn.Module):
    '''
    Prototypical Network that trains GNN encoder
    '''
    def __init__(self, args, config, Data):
        super(ProtoNet, self).__init__()
        self.args = args
        self.config = config
        self.Data = Data
        self.network = GNNEncoder(config)
        self.dim_latent = args.latent
        
        self.n = args.n_way
        self.k = args.k_shot
        self.q = args.q_query
        if self.args.setting == 'unsup':
            self.q_test = args.q_query_test
        
        self.lr = args.lr

        self.l2_penalty = args.l2_penalty

        self.meta_optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'


    def forward(self, id_spt, y_spt, id_query, y_query):
        '''
        Forward Call
        If those are not None, then it is ordinary supervised or unsupervised NaQ setting.
        '''
        features, adj = self.Data.features, self.Data.adj
        
        encodings = self.network(features, adj=adj)
        x_spt, x_query = encodings[id_spt], encodings[id_query]
        
        prototypes = x_spt.view(self.n, self.k, x_spt.size(1)).mean(dim=1)
        dists = self.euclidean_dist(x_query, prototypes)

        output = F.log_softmax(-dists, dim=1)

        # to take care of permutation during task-generation phase
        # 1shot setting or NaQ setting
        if (self.k == 1):
            label_new = torch.LongTensor([y_spt.tolist().index(i) for i in y_query.tolist()]).to(self.device)
        else:
            compressed = y_spt.detach().view(-1, self.k).float().mean(dim=1).long()
            label_new = torch.LongTensor([compressed.tolist().index(i) for i in y_query.tolist()]).to(self.device)
        loss = F.nll_loss(output, label_new)
        
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        train_acc = self.accuracy(output.cpu().detach(), label_new.cpu().detach())

        return loss, train_acc
    

    def fine_tuning(self, id_spt, y_spt, id_query, y_query):
        # Load Features/Adjacency matrix
        features, adj= self.Data.features, self.Data.adj
        
        # Fine-tune the copied model to prevent model learning in test phase
        net_GNN = deepcopy(self.network)
        
        # Get Embeddings
        encodings = net_GNN(features, vars=None, adj=adj).detach().cpu().numpy()
        x_spt, x_query, y_spt, y_query = encodings[id_spt], encodings[id_query], y_spt.detach().cpu().numpy(), y_query.detach().cpu().numpy()

        clf = LogisticRegression(max_iter=1000, C=1/self.l2_penalty).fit(x_spt, y_spt)

        # delete copied net that we used
        del net_GNN

        # Get Test Accuracy
        test_acc = clf.score(x_query, y_query)

        return None, test_acc
   
        
    # for utils
    def euclidean_dist(self, x, y):
        assert x.size(1) == y.size(1)
        n, m, d = x.size(0), y.size(0), x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x-y, 2).sum(dim=2)
    
    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)





class UnsupMAML(nn.Module):
    def __init__(self, args, config, Data):
        super(UnsupMAML, self).__init__()
        self.config = config
        self.Data = Data
        self.network = GNNEncoder(config)
        self.dim_latent = args.latent

        self.n = args.n_way
        self.k = args.k_shot
        self.q = args.q_query
        self.q_test = args.q_query_test
        self.meta_batch_size = args.meta_batch_size
        self.num_steps_meta = args.num_steps_meta
        self.inner_lr = args.inner_lr
        self.meta_update_lr = args.meta_update_lr

        self.classifier = LinearClassifier(config) # used in Meta-training
        self.l2_penalty = args.l2_penalty # used in Fine-tuning
        
        self.meta_optimizer = optim.Adam(list(self.network.parameters())+list(self.classifier.parameters()), lr=self.meta_update_lr)

    def forward(self, id_spt, y_spt, id_query, y_query):
        # Load Features/Adjacency matrix
        features, adj = self.Data.features, self.Data.adj

        query_size = self.n*self.q
        num_cls_params = len(list(self.classifier.parameters())) # number of set of parameters of linear classifier
        if num_cls_params == 0:
            num_cls_params = -len(list(self.network.parameters()))

        # losses_query[j] = validation loss(loss of query) after jth update in inner-loop (j = 0, ..., self.num_steps_meta)
        losses_query = [0 for _ in range(self.num_steps_meta+1)]
        corrects = [0 for _ in range(self.num_steps_meta+1)]
    
        #---------------- <Meta-Training & Loss recording phase(Inner-loop)> ----------------#
        for i in range(self.meta_batch_size):
            
            # Get Loss & Spt embeddings for ith task for support samples before training
            encodings = self.network(features, vars=None, adj=adj)
            x_spt = encodings[id_spt[i]]
            logits = self.classifier(x_spt, vars=None)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, list(self.network.parameters())+list(self.classifier.parameters()))
            # 1st update of model parameters
            weights_updated = list(map(lambda f: f[1] - self.inner_lr*f[0], zip(grad, list(self.network.parameters())+list(self.classifier.parameters()))))

            # Get Query embeddings & Record loss and accuracy for query samples before the 1st update for the meta-update phase
            with torch.no_grad():
                x_query = encodings[id_query[i]]
                logits_query = self.classifier(x_query, vars=self.classifier.parameters())
                loss_query = F.cross_entropy(logits_query, y_query[i])
                losses_query[0] += loss_query
                
                pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                correct = torch.eq(pred_query, y_query[i]).sum().item()
                corrects[0] = corrects[0] + correct
            
            # Get Query embeddings & Record loss and accuracy after the 1st update
            encodings = self.network(features, vars=weights_updated[:-num_cls_params], adj=adj)
            x_query = encodings[id_query[i]]
            logits_query = self.classifier(x_query, vars=weights_updated[-num_cls_params:])
            loss_query = F.cross_entropy(logits_query, y_query[i])
            losses_query[1] += loss_query
            
            with torch.no_grad():
                pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                correct = torch.eq(pred_query, y_query[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for j in range(2, self.num_steps_meta+1):
                # (1) Get Spt embeddings & loss for ith task with weights after (j-1)th update (j = 2, ..., self.num_steps_meta)
                x_spt = encodings[id_spt[i]]
                logits = self.classifier(x_spt, vars=weights_updated[-num_cls_params:])
                loss = F.cross_entropy(logits, y_spt[i])
                # (2) Get gradient at current parameter
                grad = torch.autograd.grad(loss, weights_updated)
                # (3) jth update of model parameter
                weights_updated = list(map(lambda f: f[1] - self.inner_lr*f[0], zip(grad, weights_updated)))
                
                # (4) Record loss and accuracy after the jth update for the meta-update phase
                encodings = self.network(features, vars=weights_updated[:-num_cls_params], adj=adj)
                x_query = encodings[id_query[i]]
                logits_query = self.classifier(x_query, vars=weights_updated[-num_cls_params:])
                loss_query = F.cross_entropy(logits_query, y_query[i])
                losses_query[j] += loss_query

                with torch.no_grad():
                    pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_query, y_query[i]).sum().item()
                    corrects[j] = corrects[j] + correct
        #------------------------------------------------------------------------------------#

        #------------------------- <Meta Update Phase(Outer-loop)> --------------------------#
        # Use loss of query samples by using final updated parameter
        final_loss_query = losses_query[-1] / self.meta_batch_size

        # Meta Update
        self.meta_optimizer.zero_grad()
        final_loss_query.backward()
        self.meta_optimizer.step()

        # calculating training accuracy by using final updated parameter
        final_acc = corrects[-1] / (query_size*self.meta_batch_size)
        #------------------------------------------------------------------------------------#

        return final_loss_query, final_acc


    def fine_tuning(self, id_spt, y_spt, id_query, y_query):
        # Load Features/Adjacency matrix
        features, adj= self.Data.features, self.Data.adj

        assert id_spt.shape[0] != self.meta_batch_size

        # Fine-tune the copied model to prevent model learning in test phase
        net_GNN = deepcopy(self.network)
        
        # Get Embeddings
        encodings = net_GNN(features, vars=None, adj=adj).detach().cpu().numpy()
        x_spt, x_query, y_spt, y_query = encodings[id_spt], encodings[id_query], y_spt.detach().cpu().numpy(), y_query.detach().cpu().numpy()

        clf = LogisticRegression(max_iter=1000, C=1/self.l2_penalty).fit(x_spt, y_spt)

        # delete copied net that we used
        del net_GNN

        # Get Test Accuracy
        test_acc = clf.score(x_query, y_query)

        return None, test_acc
    

