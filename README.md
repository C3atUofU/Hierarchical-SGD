## Hierarchical Federated Learning

This is the code for the paper entitled "Demystifying Why Local Aggregation Helps: Convergence Analysis of Hierarchical SGD", by Jiayi Wang, Shiqiang Wang, Rong-Rong Chen, Mingyue Ji.

### Data and Model
Dataset includes Cifar-10 and MNIST with IID and Non-IID options. It also includes FEMNIST and CelebA obtained from the LEAF framework (https://github.com/TalwalkarLab/leaf/tree/master/data) stored in the `dataset_files` folder. For CelebA, the raw images need to be downloaded separately (see https://github.com/jia-yi-wang/aistats21/blob/main/data_reader/celeba.py#L23 for details). 

Default net is VGG-11 without pretraining.

### Requirement 
Pytorch 1.6.0

Python 3.7

### Run

See the arguments in [options.py](utils/options.py).

The following are examples of running our code.

For two-level case, 
```
python main_fed.py --epochs 1000 --gpu 0 --num_groups 2 --local_period 10 --group_freq 5 --frac 0.2
```

For three-level case,
```
python three_level_main_fed.py --epochs 1000 --gpu 0 --num_groups 2 --local_period 10 --group_freq 5 --frac 0.2 --num_teams 5 --team_epochs 5
```

### Third party library
This code partly reuses code from the following repository:   
https://github.com/shaoxiongji/federated-learning  



