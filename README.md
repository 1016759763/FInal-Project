# FInal Individual Project: Knowledge Graph Based on Recommender System
This is an individual project by Shaohui Gong for UoB final dissertation.



## Introduction:
This project compare three different Recommender System, Neural Factorization Matrix (NFM), Collaborative Filtering Knowledge Graph (CFKG) and Knowledge Graph Attention Networks(KGAT).


## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.12.0
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.20.0

## Reproducibility & Example to Run the Codes

Here shows the easiest way for you to run the codes. In the file, pretrain datasets files already included, copy these codes and paste in your machine, you will get the results from each model.

* Yelp2018 dataset
```
python Main.py --model_type kgat --alg_type bi --dataset yelp2018 --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 50 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True
```

* Amazon-book dataset
```
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 50 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True
```
