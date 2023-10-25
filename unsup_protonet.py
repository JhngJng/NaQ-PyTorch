import os
import time
import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

import arg

from utils import *
from data import *
from task_generator import proto_task_generator, supervised_task_generator, downstream_task_generator
from meta import ProtoNet


def main(args, config, Data):
    # Unsupervised Meta-learning
    Meta = ProtoNet(args, config, Data).to(device)

    valid_num, test_num = args.num_valid_tasks, args.num_downstream_tasks

    # Sample a pool of episodes for training
    train_pool = [proto_task_generator(args, Data) for _ in range(args.epoch)]

    # Sample a pool of Valid/Test tasks
    valid_pool, test_pool = downstream_task_generator(args, Data, args.n_way, args.k_shot_test)

    # Define 'Empty Cans' to store valid/test accs
    valid_acc = []
    test_acc, test_stdev = [], []

    train_start = time.time()
    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool):
        epoch_start = time.time()

        #------------------------------<Training Phase>------------------------------#
        train_loss, train_acc = Meta(id_spt, y_spt.to(device), id_query, y_query.to(device))

        # Report Training loss and acc
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch}- Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} / time_elapsed: {epoch_time:.4f} seconds')

        # -------------------------<Validation & Test Phase>-------------------------#
        # Report Validation/Test acc by 160 episodes
        if (epoch + 1) % 160 == 0:
            val_temp, test_temp = [], []
            # Validation for 50 validation tasks
            for id_spt, y_spt, id_query, y_query in valid_pool:
                _, val_acc = Meta.fine_tuning(id_spt, y_spt.to(device), id_query, y_query.to(device))
                val_temp.append(val_acc)
            val_temp = np.array(val_temp).mean(axis=0)
            valid_acc.append(val_temp) # Record validation accuracy for selection

            # Testing for 500 downstream tasks
            for id_spt, y_spt, id_query, y_query in test_pool:
                _, tst_acc = Meta.fine_tuning(id_spt, y_spt.to(device), id_query, y_query.to(device))
                test_temp.append(tst_acc)
            test_temp = np.array(test_temp)
            test_temp, test_std = test_temp.mean(axis=0), test_temp.std(axis=0)/np.sqrt(len(test_pool))
            # Record testing accuracy and stdev
            test_acc.append(test_temp)
            test_stdev.append(test_std)

            # Reporting Valid/Test acc by 160 episodes
            print(f'[Valid/Test] epoch {epoch}- Valid Acc: {val_temp:.4f} / Test Acc: {test_temp:.4f}')

    train_time = time.time() - train_start
    print(f'Training ended with total elapsed time {train_time:.4f} seconds')

    # Reporting the best performance based on validation
    valid_acc, test_acc, test_stdev = np.array(valid_acc), np.array(test_acc), np.array(test_stdev)
    Best_ix = np.argmax(valid_acc)
    Best_epoch = Best_ix * 160- 1
    print('<Best Performance (with 95%% confidence interval)>')
    print(f'Valid Acc: {valid_acc[Best_ix]:.4f} / Test Acc: {test_acc[Best_ix]:.4f}±{1.96*test_stdev[Best_ix]:.4f} at epoch {Best_epoch}')

    return valid_acc[Best_ix], test_acc[Best_ix], test_stdev[Best_ix]





if __name__ == '__main__':
    torch.set_num_threads(4)
    seed = 1212
    seed_everything(seed) # Set random seed

    setting = 'unsup'

    # Parse arguments
    args = arg.parse_args_proto(setting)

    #------------------------------<Loading & Preprocessing Data>------------------------------#
    load_started = time.time()
    print(args) # Checking experimental settings

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)

    # Loading Data
    Data = load_data(args)
    if device != 'cpu':
        Data.set_device(device)
    
    # Applying graph Diffusion
    Data.add_diffusion(args)

    num_node_feat = Data.features.shape[1]

    # Model configuration, Here, args.latent value is set to be 64.
    config = [
        ('GCN', (num_node_feat, 2*args.latent), True),
        ('GCN', (2*args.latent, args.latent), True),
        ('linear', (args.latent, args.n_way), True)
    ]

    print(config) # Checking model configuration

    print(f'Data loading and preprocessing have been ended in {time.time()-load_started:4f} seconds')

    # Result recording
    os.makedirs(f'./results/{args.dataset}/ProtoNet', exist_ok=True)
    results = open(f'./results/{args.dataset}/ProtoNet/{args.setting}_{args.query_generation}.txt', 'a')
    results.write('\n====================================================================================================\n')

    # Record hyperparameter settings
    results.write('BaseModel=ProtoNet, ')
    for arg in vars(args).keys():
        results.write(f'{arg}={vars(args)[arg]}, ')
    results.write('\n')

    #------------------------------<Run Experiments>------------------------------#
    Exp_started = time.time()

    Valid_summaries, Test_summaries = [], []
    
    val_acc, test_acc, test_std = main(args, config, Data)

    Total_elapsed = time.time() - Exp_started
    print(f'Total time elapsed in {Total_elapsed:.4f} seconds')

    # Record the summary of the experimental results with 95% C.I.
    results.write(f'Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}±{1.96*test_std:.4f}, Total elapsed: {Total_elapsed:.4f} seconds')

    
    results.write('\n====================================================================================================\n')
    results.close()
