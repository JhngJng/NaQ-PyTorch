import os
import numpy as np
import scipy.sparse as sp
import torch
import random
import pickle

from copy import deepcopy

from utils import *



def downstream_task_generator(args, Data, n_way, k_shot, q_query=8):
    '''
    Downstream Task Generator to Consistent Comparison by saving them.

    Query is set to be 8. This is consistent for all baselines and all settings
    '''   
    # If saved downstream task for current setting exists, load the data and return it.
    if os.path.exists(f'./save/downstream_tasks/{args.dataset}/{n_way}way_{k_shot}shot.pkl'):
        with open(f'./save/downstream_tasks/{args.dataset}/{n_way}way_{k_shot}shot.pkl', 'rb') as tasks:
            valid_pool, test_pool = pickle.load(tasks)
        return valid_pool, test_pool
    
    # If saved downstream task for current setting does not exist, then we generate the tasks and save it.
    os.makedirs(f'./save/downstream_tasks/{args.dataset}/', exist_ok=True)
    
    valid_num, test_num = args.num_valid_tasks, args.num_downstream_tasks

    valid_pool = [supervised_task_generator(Data.id_by_class, Data.class_list_valid, Data.labels, n_way, k_shot, q_query) for _ in range(valid_num)]
    test_pool = [supervised_task_generator(Data.id_by_class, Data.class_list_test, Data.labels, n_way, k_shot, q_query) for _ in range(test_num)]

    tasks = (valid_pool, test_pool)
    with open(f'./save/downstream_tasks/{args.dataset}/{n_way}way_{k_shot}shot.pkl', 'wb') as f:
        pickle.dump(tasks, f)

    return valid_pool, test_pool


# Task-generator for MAML-like algorithms (which require meta-batches)
def meta_batch_generator(args, Data):
    if args.setting == 'sup':
        id_by_class, class_list, labels = Data.id_by_class, Data.class_list_train, Data.labels
        id_spts, y_spts, id_queries, y_queries = [], [], [], []
        for _ in range(args.meta_batch_size):
            id_spt, y_spt, id_query, y_query = supervised_task_generator(id_by_class, class_list, labels, args.n_way, args.k_shot, args.q_query)
            id_spts.append(id_spt)
            y_spts.append(y_spt.view(1, -1))
            id_queries.append(id_query)
            y_queries.append(y_query.view(1, -1))
        
        id_spts, id_queries = np.array(id_spts), np.array(id_queries)
        y_spts, y_queries = torch.cat(y_spts, 0).long(), torch.cat(y_queries, 0).long()
        
        return id_spts, y_spts, id_queries, y_queries
    
    if args.setting == 'unsup':
        id_train = Data.id_train
        diffusion = True
        spt_to_sample = 1 # only 'UMTRA' style initial spt generation is considered.
        qry_to_gen = args.q_query
        
        # When 'neighbors' are utilized as queries
        if args.query_generation == 'NaQ':
            id_spts, y_spts, id_queries, y_queries = [], [], [], []
            
            for _ in range(args.meta_batch_size):
                # initial spt sample sampling
                id_spt, y_spt = spt_sampling(id_train, args.n_way, spt_to_sample)

                # query generation by NaQ
                id_query, y_query = query_generation_NaQ(id_spt, y_spt, qry_to_gen, Data, diffusion)
                
                id_spts.append(id_spt)
                y_spts.append(y_spt.view(1,-1))
                id_queries.append(id_query)
                y_queries.append(y_query.view(1,-1))

            id_spts, y_spts = np.array(id_spts), torch.cat(y_spts, 0).long()
            id_queries, y_queries = np.array(id_queries), torch.cat(y_queries, 0).long()

            return id_spts, y_spts, id_queries, y_queries



# Task-generator for ProtoNet-like algorithms
def proto_task_generator(args, Data):
    if args.setting == 'sup':
        id_by_class, class_list, labels = Data.id_by_class, Data.class_list_train, Data.labels
        id_spt, y_spt, id_query, y_query = supervised_task_generator(id_by_class, class_list, labels, args.n_way, args.k_shot, args.q_query)

        return id_spt, y_spt, id_query, y_query
    
    if args.setting == 'unsup':
        id_train = Data.id_train
        diffusion = True
        spt_to_sample = 1 # only 'UMTRA' style initial spt generation is considered.
        qry_to_gen = args.q_query

        # When NaQ query generation is utilized
        if args.query_generation == 'NaQ':
            id_spt, y_spt = spt_sampling(id_train, args.n_way, spt_to_sample)

            # Query generation by NaQ
            id_query, y_query = query_generation_NaQ(id_spt, y_spt, qry_to_gen, Data, diffusion)
            
            return id_spt, y_spt, id_query, y_query



def spt_sampling(id_train, n_way, k_shot):
    '''
    Unsupervised Task Generator to be used in my algorithm
    Used in Training Phase
    Here, we only return support set indices/pseudo-labels(id_support and y_spt)
    '''
    # 1. Sample just sample indices
    id_support = random.sample(id_train, n_way*k_shot)
    # 2. Give them random labels so that satisfying n_way-k_shot setting
    # 2-1) making random labels
    rand_train_labels = []
    for i in range(n_way):
        labels = [i for _ in range(k_shot)]
        rand_train_labels += labels
    # 2-2) assign pseudo-labels randomly
    random.shuffle(rand_train_labels)
    y_spt = torch.LongTensor(rand_train_labels)

    return np.array(id_support), y_spt


def query_generation_NaQ(id_spt, y_spt, to_gen, Data, diffusion=True):
    '''
    Query generation algorithm based on Diffusion matrix.
    '''
    if diffusion is True:
        # Find top-q similar nodes
        if Data.Diffusion_matrix.layout != torch.sparse_coo:
            _, nbr_ix = Data.Diffusion_matrix[id_spt].topk(k=to_gen, dim=1)
        else:
            _, nbr_ix = sparse_topk_dim1(Data.Diffusion_matrix, id_spt, k=to_gen)
        nbr_ix = nbr_ix.cpu().tolist()
        # Get query samples
        id_query, y_query = [], []
        for i in range(id_spt.shape[0]):
            id_query += nbr_ix[i]
            y_query += [y_spt[i].item() for _ in range(len(nbr_ix[i]))]
        id_query = np.array(id_query)
        y_query = torch.LongTensor(y_query)

    else:
        raise NotImplementedError

    return id_query, y_query


def supervised_task_generator(id_by_class, class_list, labels, n_way, k_shot, q_query):
    '''
    Usual supervised few-shot task generator from GPN code
    Used in Fine-tuning Phase in Unsupervised Setting
    '''
    # sample class indices
    class_selected = random.sample(class_list, n_way)
    # sample n_way-k-shot/q_query samples
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + q_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])
    # New labels for Support/Query samples
    y_spt = torch.LongTensor([class_selected.index(i) for i in labels[np.array(id_support)]])
    y_query = torch.LongTensor([class_selected.index(i) for i in labels[np.array(id_query)]])

    return np.array(id_support), y_spt, np.array(id_query), y_query


