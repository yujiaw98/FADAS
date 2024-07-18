# FADAS

This repository contains the PyTorch implementation of [FADAS: Towards Federated Adaptive Asynchronous Optimization](https://openreview.net/pdf?id=j56JAd29uH) (accepted by ICML 2024).

## Prerequisites

Pytorch 2.0.1
CUDA 12.0

## Running the experiments

To run the experiment for FADAS without delay-adaptive:

```
python3 asyncfl_forward.py --model=resnet --dataset=cifar10 --gpu=0 --iid=0 --optimizer=fedams --local_lr=0.1 --lr=0.0001 --num_users=50 --frac=0.5 --local_bs=50 --local_ep=2 --epochs=500 --scale=1 --update_freq=5 --dir_alpha=0.3 --delay_type=large_delay
```

To run the experiment for  FADAS with delay-adaptive:

```
python3 g_delay_adaptive.py --model=resnet --dataset=cifar10 --gpu=0 --iid=0 --optimizer=fedams --local_lr=0.1 --lr=0.001 --num_users=50 --frac=0.5 --local_bs=50 --local_ep=2 --epochs=500 --scale=1 --update_freq=5 --dir_alpha=0.3 --delay_type=large_delay --tauc=1 
```
## **Hyperparameter details**

The default values for various paramters parsed to the experiment are given in `options.py`.

- `-dataset:` Default: 'cifar10'. Options: 'mnist', 'fmnist', 'cifar100'.
- `-model:` Default: 'cnn'. Options: 'mlp', 'resnet', 'convmixer'.
- `-gpu:` To use cuda, set to a specific GPU ID.
- `-epochs:` Number of rounds of training.
- `-local_ep:` Number of local epochs.
- `-local_lr:` Learning rate for local update.
- `-lr:` Learning rate for global update.
- `-local_bs:` Local update batch size.
- `-iid:` Default set to IID. Set to 0 for non-IID.
- `-num_users:` Number of users. Default is 100.
- `-frac:` Fraction of users to be used for federated updates. Default is 0.1.
- `-optimizer:` Default: 'fedavg'. Options: 'fedadam', 'fedams'.
- `-update_freq:` The buffer size  for the asynchronous update. Default: 5.
- `-scale:` The parameter for Dirichlet sampling in the delay simulation. Default: 1.0.
- `-delay_type:` The type of large or mild delay for the experiments. Set delay_type=large_delay for the large worst case delay
- `-tauc:` The threshold for delay adaptive learning rate

## Citation

Please check our paper for technical details and full results.

```
@inproceedings{
wang2024fadas,
title={{FADAS}: Towards Federated Adaptive Asynchronous Optimization},
author={Yujia Wang and Shiqiang Wang and Songtao Lu and Jinghui Chen},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=j56JAd29uH}
}
```
