import time
import argparse

def parse_args(setting):
    argparser = argparse.ArgumentParser()

    # Basic Arguments
    argparser.add_argument('--device', type=int, default=0, help='GPU device number to use')

    # Basic Model Setting Arguments
    argparser.add_argument('--setting', type=str, default=setting, help='Supervised(sup)/Unsupervised(unsup) setting we use')
    argparser.add_argument('--dataset', type=str, default='Amazon_clothing', help='Dataset:Amazon_clothing/Cora_full/dblp/Amazon_eletronics')
    argparser.add_argument('--epoch', default=1000, type=int, help='# of total epochs')
    
    # Episode Setting Arguments
    argparser.add_argument('--n_way', type=int, default=5, help='# of classes we use in each episodes, 5/10/20 used')
    if setting == 'sup':
        argparser.add_argument('--k_shot', type=int, default=1, help='# of labeled samples we use in each episodes, 1/5 used')
        argparser.add_argument('--q_query', type=int, default=8, help='# of query samples we use in each episodes 8 used.')
    elif setting == 'unsup':
        argparser.add_argument('--k_shot', type=int, default=1, help='# of labeled samples we use in each "TRAINING" episodes, 1 used.')
        argparser.add_argument('--k_shot_test', type=int, default=1, help='# of labeled samples we use in "DOWNSTREAM" tasks, 1/5 used')
        argparser.add_argument('--q_query', type=int, default=10, help='# of query samples we use in each "TRAINING" episodes')
        argparser.add_argument('--q_query_test', type=int, default=8, help='# of query samples we use in each "DOWNSTREAM" episodes, 8 used')
        
    # Training Phase Arguments
    argparser.add_argument('--meta_batch_size', type=int, default=16, help='meta batch size')
    argparser.add_argument('--num_steps_meta', type=int, default=10, help='# of parameter updates for each tasks during the meta-training phase(inner loop)')
    argparser.add_argument('--inner_lr', type=float, default=0.01, help='learning rate we use for parameter updates for each tasks.')
    argparser.add_argument('--meta_update_lr', type=float, default=3e-3, help='learning rate we use for parameter meta update.')
    
    # Fine-tuning Phase Arguments
    argparser.add_argument('--num_downstream_tasks', type=int, default=500, help='# of downstream tasks to evaluate. 500 used.')
    argparser.add_argument('--num_valid_tasks', type=int, default=50, help='# of validation tasks to evaluate. 50 used.')
    argparser.add_argument('--l2_penalty', type=float, default=1e-4, help='L2 Regularization Hyperparameter used in Fine-tuning phase. Set this to 0 if you do not want l2 regularization')

    # Model Architecture Arguments
    argparser.add_argument('--latent', type=int, default=64, help='dim. of latent space we embed node features')

    # Unsupervised Setting Arguments
    if setting == 'unsup':
        # Query-Generation Strategy Arguments
        argparser.add_argument('--query_generation', type=str, default='NaQ', help='NaQ: using similar nodes as queries')
        argparser.add_argument('--type', type=str, default='feature', help='NaQ type. feature(NaQ-Feat), diffusion(NaQ-Diff) supported')
        argparser.add_argument('--PPR_alpha', type=float, default=0.1, help='hyperparam. alpha value to get Diffusion PPR matrix. Based on Original Paper, value b/w [.05, .2] will be chosen')
    
    # Etc. Arguments
    timestr = time.strftime('%Y%m%d-%H%M')
    argparser.add_argument('--exp_time', type=str, default=timestr)

    args = argparser.parse_args()
    return args



# for ProtoNet
def parse_args_proto(setting):
    argparser = argparse.ArgumentParser()

    # Basic Arguments
    argparser.add_argument('--device', type=int, default=0, help='GPU device number to use')

    # Basic Model Setting Arguments
    argparser.add_argument('--setting', type=str, default=setting, help='Supervised(sup)/Unsupervised(unsup) setting we use')
    argparser.add_argument('--dataset', type=str, default='Amazon_clothing', help='Dataset:Amazon_clothing/Cora_full/dblp/Amazon_eletronics')
    argparser.add_argument('--epoch', default=16000, type=int, help='# of total epochs') # the value set to be (# epoch)*(meta-batch size) in MAML
    
    # Episode Setting Arguments
    argparser.add_argument('--n_way', type=int, default=5, help='# of classes we use in each episodes, 5/10/20 used')
    if setting == 'sup':
        argparser.add_argument('--k_shot', type=int, default=1, help='# of labeled samples we use in each episodes, 1/5 used')
        argparser.add_argument('--q_query', type=int, default=8, help='# of query samples we use in each episodes 8 used.')
    elif setting == 'unsup':
        argparser.add_argument('--k_shot', type=int, default=1, help='# of labeled samples we use in each "TRAINING" episodes, 1 used.')
        argparser.add_argument('--k_shot_test', type=int, default=1, help='# of labeled samples we use in "DOWNSTREAM" tasks, 1/5 used')
        argparser.add_argument('--q_query', type=int, default=10, help='# of query samples we use in each "TRAINING" episodes')
        argparser.add_argument('--q_query_test', type=int, default=8, help='# of query samples we use in each "DOWNSTREAM" episodes, 8 used')
    
    # Training Phase Arguments
    argparser.add_argument('--lr', type=float, default=1e-4, help='ordinary learning rate. Chosen b/w 1e-5/5e-5/1e-4/3e-4/5e-4/1e-3/3e-3')
    
    # Fine-tuning Phase Arguments
    argparser.add_argument('--num_downstream_tasks', type=int, default=500, help='# of downstream tasks to evaluate. 500 used.')
    argparser.add_argument('--num_valid_tasks', type=int, default=50, help='# of validation tasks to evaluate. 50 used.')
    argparser.add_argument('--l2_penalty', type=float, default=1e-4, help='L2 Regularization Hyperparameter used in Fine-tuning phase. Set this to 0 if you do not want l2 regularization')

    # Model Architecture Arguments
    argparser.add_argument('--latent', type=int, default=256, help='dim. of latent space we embed node features')

    # Unsupervised Setting Arguments
    if setting == 'unsup':
        # Query-Generation Strategy Arguments
        argparser.add_argument('--query_generation', type=str, default='NaQ', help='NaQ: using similar nodes as queries')
        argparser.add_argument('--type', type=str, default='feature', help='NaQ type. feature(NaQ-Feat), diffusion(NaQ-Diff) supported')
        argparser.add_argument('--PPR_alpha', type=float, default=0.1, help='hyperparam. alpha value to get Diffusion PPR matrix. Based on Original Paper, value b/w [.05, .2] will be chosen')
    
    # Etc. Arguments
    timestr = time.strftime('%Y%m%d-%H%M')
    argparser.add_argument('--exp_time', type=str, default=timestr)

    args = argparser.parse_args()
    return args

