# Unsupervised Episode Generation for Graph Meta-learning
The source code of the unsupervised episode generation method called **Neighbors as Queries (NaQ)** from the paper "[Unsupervised Episode Generation for Graph Meta-learning](https://arxiv.org/abs/2306.15217)"

## Overview
### 1. NaQ-Feat
NaQ-Feat generates query set by sampling raw-feature level similar nodes for each randomly sampled support set nodes in the entire graph.
<p align="center"><img width="700" src="./images/NaQ-Feat_Figure.png"></p>

### 2. NaQ-Diff
NaQ-Diff generates query set by sampling structurally similar nodes found via [graph Diffusion](https://arxiv.org/abs/1911.05485) for each randomly sampled support set nodes in the entire graph.
<p align="center"><img width="700" src="./images/NaQ-Diff_Figure.png"></p>

## Abstract
We investigate Unsupervised Episode Generation methods to solve Few-Shot Node-Classification (FSNC) task via Meta-learning without labels. Dominant meta-learning methodologies for FSNC were developed under the existence of _abundant_ labeled nodes from _diverse_ base classes for training, which however may not be possible to obtain in the real-world. Although a few studies tried to tackle the label-scarcity problem in graph meta-learning, they still rely on a few labeled nodes, which hinders the full utilization of the information of all nodes in a graph. 
Despite the effectiveness of graph contrastive learning (GCL) methods in the FSNC task without using the label information, they mainly learn generic node embeddings without consideration of the downstream task to be solved, which may limit its performance in the FSNC task.
To this end, we propose a simple yet effective _unsupervised_ episode generation method to benefit from the generalization ability of meta-learning for the FSNC task, while resolving the label-scarcity problem.
Our proposed method, called Neighbors as Queries (NaQ), generates training episodes based on pre-calculated node-node similarity. Moreover, NaQ is model-agnostic; hence it can be used to train any existing supervised graph meta-learning methods in an unsupervised manner, while not sacrificing much of their performance or sometimes even improving them.
Extensive experimental results demonstrate the potential of our unsupervised episode generation methods for graph meta-learning towards the FSNC task.

## How to check implementations?
To check implementations of NaQ, please see task_generator.py method .query_generation_NaQ().

## How to Run?
To run our methods with ProtoNet (e.g. in Amazon-Clothing, 5-way 1-shot)
```bash
# NaQ-Feat
python unsup_protonet.py --dataset Amazon_clothing --n_way 5 --k_shot_test 1 --query_generation NaQ --type feature --lr 1e-4
# NaQ-Diff
python unsup_protonet.py --dataset Amazon_clothing --n_way 5 --k_shot_test 1 --query_generation NaQ --type diffusion --lr 1e-4
```

## Requirements
python=3.8.13  
pytorch=1.11.0  
scikit-learn=1.1.1  
numpy=1.21.5  
scipy=1.5.3  
pyg=2.0.4 (torch-geometric)  
