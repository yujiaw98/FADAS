#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import compressors
from torch.autograd import Function
from utils import dequantize, quantize, norm_of_param
import random

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
       
        ###### define compressors #######
        self.compressor = compressors.Compressor()
        if args.compressor == 'identical':
            self.compressor.makeTopKCompressor(1) 
        elif args.compressor == 'topk4':
            self.compressor.makeTopKCompressor(1/4) 
        elif args.compressor == 'topk2':
            self.compressor.makeTopKCompressor(1/2) 
        elif args.compressor == 'topk8':
            self.compressor.makeTopKCompressor(1/8) 
        elif args.compressor == 'topk64':
            self.compressor.makeTopKCompressor(1/64)   
        elif args.compressor == 'sign':
            self.compressor.makeSignCompressor() 
        elif args.compressor == 'quantized':
            self.compressor.makeQSGD_FP64()
        else:
            exit('unknown compressor: {}'.format(args.compressor))
        


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=int(1), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                batch_size=int(1), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights_local(self, model, global_round, lrs):
        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_acc = []
        optimizer = torch.optim.SGD(model.parameters(), lr=lrs[global_round], momentum=0, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.local_lr)
        grad_max = []
        for ep in range(self.args.local_ep):
            batch_loss = []
            total = 0
            correct = 0

            # num_batches = len(self.trainloader)
            # random_batch_index = random.randint(0, num_batches - 1)
            # trainloader_iter = iter(self.trainloader)
            # for _ in range(random_batch_index):
            #     next(trainloader_iter)
            # images, labels = next(trainloader_iter)
            # images, labels = images.to(self.device), labels.to(self.device)
            # print(len(self.trainloader))
            # random_batch = random.choice(list(self.trainloader))

            # images, labels = random_batch
            # images, labels = images.to(self.device), labels.to(self.device)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, labels)
                loss.backward()

                _, pred_labels = torch.max(logits, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()

                optimizer.step()
                grads = []
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        grads.append(parameter.grad)
                grad_max.append(norm_of_param(grads))
            # if self.args.verbose and (batch_idx % 10 == 0):
            #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         global_round, iter, batch_idx * len(images),
            #         len(self.trainloader.dataset),
            #         100. * batch_idx / len(self.trainloader), loss.item()))
            
                batch_loss.append(loss.item() * len(labels))
                total += len(labels)

            epoch_loss.append(sum(batch_loss)/total)
            epoch_acc.append(correct/total)

        par_after = []
        for p in model.parameters():
            par_after.append(p.data.detach().clone())

        max_grad = max(grad_max)

        return model.state_dict(), par_after, sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc), max_grad
     
    
    def compressSignal(self, signal, D):
#         transit_bits = 0
        signal_compressed = []
        for p in signal:
            signal_compressed.append(torch.zeros_like(p))

        signal_flatten = torch.zeros(D).to(self.device)

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_flatten[(signal_offset):(signal_offset + offset)] = signal[t].flatten(0)
            signal_offset += offset
            

        signal_flatten = self.compressor.compressVector(signal_flatten)
#         transit_bits += compressors.Compressor.last_need_to_send_advance

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_compressed[t].flatten(0)[:] = signal_flatten[(signal_offset):(signal_offset + offset)]
            signal_offset += offset

        return signal_compressed 

    def compressSignal_layerwise(self, signal, D):
        transit_bits = 0
        signal_compressed = []
        for p in signal:
            signal_compressed.append(torch.zeros_like(p))
      
        signal_flatten = torch.zeros(D).to(self.device)

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_flatten[(signal_offset):(signal_offset + offset)] = self.compressor.compressVector(signal[t].flatten(0), self.iteration)
#             transit_bits += compressors.Compressor.last_need_to_send_advance
            signal_offset += offset

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_compressed[t].flatten(0)[:] = signal_flatten[(signal_offset):(signal_offset + offset)]
            signal_offset += offset

        return signal_compressed 
    
#     def update_weights_ae(self, model, global_round):
#         # Set mode to train model
#         model.train()
#         epoch_loss = []

#         optimizer = torch.optim.SGD(model.parameters(), lr=self.args.local_lr, momentum=0)
   
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             total = 0
#             for batch_idx, (images, labels) in enumerate(self.trainloader):
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 model.zero_grad()
#                 output = model(images)
#                 loss = nn.MSELoss()(output, images)
#                 loss.backward()
#                 optimizer.step()

#                 if self.args.verbose and (batch_idx % 10 == 0):
#                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         global_round, iter, batch_idx * len(images),
#                         len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), loss.item()))
                
#                 batch_loss.append(loss.item() * len(labels))
#                 total += len(labels)
#             epoch_loss.append(sum(batch_loss)/total)

#         par_after = []
#         for p in model.parameters():
#             par_after.append(p.data.detach().clone())
            
        
#         return model.state_dict(), par_after, sum(epoch_loss) / len(epoch_loss)
    
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.local_lr, momentum=0)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
#                 self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item()/len(labels))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        par_after = []
        for p in model.parameters():
            par_after.append(p.data.detach().clone())
        
        return par_after, sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item() * len(labels)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        loss = loss/total
        return accuracy, loss


def update_model_inplace(model, par_before, delta, args, cur_iter, momentum_buffer_list, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, lr_l):
    grads = copy.deepcopy(delta)
    
    # learning rate decay
    iteration = cur_iter + 1  # add 1 is to make sure nonzero denominator in adam calculation
    # if iteration < int(args.epochs/2):
    #     lr_decay = 1.0
    # elif iteration < int(3*args.epochs/4):
    #     lr_decay = 0.1
    # else:
    #     lr_decay = 0.01
    lr_decay=1.0

    for i, param in enumerate(model.parameters()): 
        grad = grads[i]  # recieve the aggregated (averaged) gradient
        
        # SGD calculation
        if args.optimizer == 'fedavg':
            # need to reset the trainable parameter
            # because we have updated the model via state_dict when dealing with batch normalization
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).add_(grad, alpha=args.lr * lr_decay * lr_l)
            # param.data.add_(grad, alpha=args.lr * lr_decay)
        # SGD+momentum calculation
        elif args.optimizer == 'fedavgm':
            buf = momentum_buffer_list[i]
            buf.mul_(args.momentum).add_(grad, alpha=1)
            grad = buf
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).add_(grad, alpha=args.lr * lr_decay)
        # adam calculation
        elif args.optimizer == 'fedadam':
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(args.eps) # without maximum

            step_size = args.lr * lr_decay / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
        elif args.optimizer == 'fedams':
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # torch.maximum(max_exp_avg_sqs[i], torch.tensor(args.eps), out=max_exp_avg_sqs[i])
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(args.eps)
            # denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2))

            step_size = args.lr * lr_l * lr_decay / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
        elif args.optimizer == 'fedamsda':
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            # exp_avg.mul_(args.beta1).add_(lr_l*grad, alpha=1 - args.beta1)
            exp_avg.mul_(1-(1 - args.beta1)*lr_l).add_(lr_l*grad, alpha=1 - args.beta1)
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # torch.maximum(max_exp_avg_sqs[i], torch.tensor(args.eps), out=max_exp_avg_sqs[i])
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(args.eps)
            # denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2))

            step_size = args.lr * lr_decay / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
        elif args.optimizer == 'fedamsd':
            lr_decay=1.0/math.sqrt(iteration)
            
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(args.eps)

            step_size = args.lr * lr_decay / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
        elif args.optimizer == 'fedadagrad':
            exp_avg_sq = exp_avg_sqs[i]
            exp_avg_sq.addcmul_(1, grad, grad)            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(grad, exp_avg_sq.sqrt().add_(args.eps), value=args.lr * lr_decay)
        elif args.optimizer == 'fedrmsprop':
            exp_avg_sq = exp_avg_sqs[i]
            bias_correction2 = 1 - args.beta2 ** iteration
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)  
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(args.eps) # without maximum
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(grad, denom, value=args.lr * lr_decay)
        elif args.optimizer == 'fedyogi':
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            tmp_sq = grad ** 2
            tmp_diff = exp_avg_sq - tmp_sq
            exp_avg_sq.add_( - (1 - args.beta2), torch.sign(tmp_diff) * tmp_sq)
            
            denom = exp_avg_sq.sqrt().add_(args.eps)

            step_size = args.lr * lr_decay * math.sqrt(bias_correction2) / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
            
        else:
            exit('unknown optimizer: {}'.format(args.optimizer))


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item() * len(labels)

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss/total
    return accuracy, loss

class SymmetricUnbiasedQuantization(Function):
    @staticmethod
    def forward(ctx, input, num_bits):
        # qmin = -2**(num_bits - 1)
        # qmax = 2**(num_bits - 1) - 1
        qmin = 0
        qmax = 2**num_bits - 1
        q_x, scale, zero_point = quantize(input, qmin, qmax, num_bits)
        ctx.save_for_backward(q_x, scale, zero_point)
        return dequantize(q_x, scale, zero_point)

    @staticmethod
    def backward(ctx, grad_output):
        q_x, scale, zero_point = ctx.saved_tensors
        grad_input = grad_output.clone() / scale
        return grad_input, None

