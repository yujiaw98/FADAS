#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from dataclasses import replace
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import math
import torch
from torch import nn
from tensorboardX import SummaryWriter
import random
from collections import deque
from queue import Queue
from scipy.stats import halfnorm, poisson

from options import args_parser
from update import LocalUpdate, update_model_inplace, test_inference
from utils import get_model, get_dataset, average_weights, exp_details, average_parameter_delta, set_seed, average_parameter, generate_class_assignments

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)
    
    # define paths
#     out_dir_name = args.model + args.dataset + args.optimizer + '_lr' + str(args.lr) + '_locallr' + str(args.local_lr) + '_localep' + str(args.local_ep) +'_localbs' + str(args.local_bs) + '_eps' + str(args.eps)
    file_name = '/asynfwd_{}_{}_{}_llr[{}]_glr[{}]_eps[{}]_le[{}]_bs[{}]_iid[{}]_mi[{}]_frac[{}]_update[{}]_scale[{}]_dir_alpha[{}]_{}.pkl'.\
                format(args.dataset, args.model, args.optimizer, 
                    args.local_lr, args.lr, args.eps, 
                    args.local_ep, args.local_bs, args.iid, args.max_init, args.frac, args.update_freq, args.scale, args.dir_alpha, args.delay_type)
    logger = SummaryWriter('./logs/'+file_name)
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1) # limit cpu use
    print ('-- pytorch version: ', torch.__version__)
    
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    set_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)

    label_dist_list = []
    # load dataset and user groups
    train_dataset, test_dataset, num_classes, user_groups = get_dataset(args)

    for i in range(len(user_groups)):
        user_list = []
        for user in user_groups[i]:
            user_list.append(train_dataset[user][-1])
        label_dist_list.append(user_list)
    dist_list = [np.array([len([j for j in user_list if j == i]) for i in range(0, 10)], dtype='int64')
                                            / len(user_list) for user_list in label_dist_list]

    dsize = [len(user_groups[i]) for i in range(len(user_groups))]
    time_takens = [0 for _ in range(len(user_groups)) ]
    class_assignments = generate_class_assignments(3, len(user_groups), [3*args.scale, 2*args.scale, args.scale], args.seed)
    print([sum(class_assignments==0),sum(class_assignments==1), sum(class_assignments==2)])

    for i, x in enumerate(class_assignments):
        if x == 0:
            time_takens[i] = 1
        elif x == 1:
            time_takens[i] = 2
        else:
            time_takens[i] = 3

    # Set the model to train and send it to device.
    global_model = get_model(args.model, args.dataset, train_dataset[0][0].shape, num_classes, device)
    global_model.to(device)
    global_model.train()
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    
    # this is the place to initialize states for adam
    ###
    momentum_buffer_list = []
    exp_avgs = []
    exp_avg_sqs = []
    max_exp_avg_sqs = [] 
    for i, p in enumerate(global_model.parameters()):         
        momentum_buffer_list.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        exp_avgs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        exp_avg_sqs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        max_exp_avg_sqs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False)+args.max_init) # 1e-2
        # max_exp_avg_sqs.append(torch.full_like(p.data.detach().clone(), args.eps, dtype=torch.float, requires_grad=False)+args.max_init) # 1e-2


    optimizer_setup = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=args.local_lr)

    lrs = []

    if args.delay_type=='nolrs' or args.delay_type=='large_delay' or args.delay_type == 'new_delay':
        lrs = [args.local_lr for _ in range(args.epochs)]
    # print(lrs)
    
    # Training
    train_loss_sampled, train_loss, train_accuracy = [], [], []
    test_loss, test_accuracy = [], []
    train_acc = []
    taumax, tauavg = [], []
    start_time = time.time()

    client_list = list(range(args.num_users))
    m = max(int(args.frac * args.num_users), 1)

    # init current_list
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    current_list = idxs_users.tolist()
    remaining_elements = [x for x in client_list if x not in current_list]
    print(current_list)
    
    assign_delta = Queue()
    
    completed_tasks_and_time = []
    max_grad = []
    time_stamp = [0]*args.num_users
    for epoch in tqdm(range(args.epochs)):
        ep_time = time.time()
        
        local_weights, local_params, local_losses = [], [], []
        update_weights = []
        update_delta = []
        update_loss = []
        update_acc = []
        local_delta = []
        delay = []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        # this is to store parameters before update
        ###
        par_before = []
        for p in global_model.parameters():  # get trainable parameters
            par_before.append(p.data.detach().clone())
        # this is to store parameters before update
        w0 = global_model.state_dict()  # get all parameters, includeing batch normalization related ones
            
        global_model.train()
        
        # if epoch == 0:
        for i, idx in enumerate(current_list):
            # print(idx)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)
            if epoch==0:
                w, p, loss, acc, max_g = local_model.update_weights_local(
                    model=copy.deepcopy(global_model), global_round=epoch, lrs=lrs)
            else:
                if i == len(current_list) - 1:
                    w, p, loss, acc, max_g = local_model.update_weights_local(
                    model=copy.deepcopy(global_model), global_round=epoch, lrs=lrs)
                else:
                    w, p, loss, acc, max_g = local_model.update_weights_local(
                    model=copy.deepcopy(last_round_global_model), global_round=epoch, lrs=lrs)
            # local_weights.append(copy.deepcopy(w))
            # local_params.append(copy.deepcopy(p))
            # local_losses.append(copy.deepcopy(loss))
            max_grad.append(max_g)
            
            if time_takens[idx] == 1:
                time_taken = random.uniform(1, 2)
            if time_takens[idx] == 2:
                time_taken = random.uniform(3, 5)
            if time_takens[idx] == 3:
                if args.delay_type=='large_delay': 
                    time_taken = random.uniform(50,80)
                else:
                    time_taken = random.uniform(5, 8)

            delta = [p[i] - par_before[i] for i in range(len(par_before))]
            completed_tasks_and_time.append((delta, loss, w, idx, time_taken, acc))
            time_stamp[idx] = epoch
        
        completed_tasks_and_time.sort(key=lambda x: x[4])

        time_list = [sublist[4] for sublist in completed_tasks_and_time]
        # print(time_list)

        for delta, loss, w, client, _, acc in completed_tasks_and_time[:args.update_freq]:
            update_weights.append(w)
            update_loss.append(loss)
            update_acc.append(acc)
            update_delta.append(delta)
            remaining_elements.append(client)
            delay.append((epoch - time_stamp[client]))
        
        print(f'delay:{delay}')
        
        print(max(max_grad))
        
        taumax.append(max(delay))
        tauavg.append(sum(delay)/len(delay))

        last_round_global_model = copy.deepcopy(global_model)
        bn_weights = average_weights(update_weights)
        global_model.load_state_dict(bn_weights)
        global_delta = average_parameter(update_delta)
        update_model_inplace(global_model, par_before, global_delta, args, epoch, 
                momentum_buffer_list, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, 1)
        loss_avg = sum(update_loss) / len(update_loss)
        acc_avg = sum(update_acc) / len (update_acc)
        train_loss.append(loss_avg)
        train_acc.append(acc_avg)


        current_time = completed_tasks_and_time[args.update_freq-1][4]

        if m != args.update_freq:
            completed_tasks_and_time = completed_tasks_and_time[-(m-args.update_freq):]
            completed_tasks_and_time = [(delta, loss, w, idx, time_taken - current_time, acc) for delta, loss, w, idx, time_taken, acc in completed_tasks_and_time]
        elif m == args.update_freq:
            completed_tasks_and_time = []

        client_still_working = [sublist[3] for sublist in completed_tasks_and_time]
        print(f'client_still_working:{client_still_working}')
        remaining_elements = [x for x in client_list if x not in client_still_working]
        current_list = random.sample(remaining_elements, args.update_freq)
        
        print('Epoch Run Time: {0:0.4f} of {1} global rounds'.format(time.time()-ep_time, epoch+1))
        print(f'Training Loss : {train_loss[-1]}')
        print(f'Training Acc : {train_acc[-1]}')
        logger.add_scalar('train loss', train_loss[-1], epoch)
        logger.add_scalar('train acc', train_acc[-1], epoch)
        global_model.eval()
        
        # Test inference after completion of training
        test_acc, test_ls = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        test_loss.append(test_ls)

        # print global training loss after every rounds

        print(f'Test Loss : {test_loss[-1]}')
        print(f'Test Accuracy : {test_accuracy[-1]} \n')

        # print(np.mean(tauavg))
        # print(max(taumax))
        logger.add_scalar('test loss', test_loss[-1], epoch)
        logger.add_scalar('test acc', test_accuracy[-1], epoch)

        if args.save:
            # Saving the objects train_loss and train_accuracy:
            with open(args.outfolder + file_name, 'wb') as f:
                pickle.dump([train_loss, test_loss, test_accuracy, train_acc], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

