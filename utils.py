#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
# from models.randaug import RandAugment
from torch.autograd import Function
from femnist_datasets import FEMNIST, ShakeSpeare

def get_model(model_name, dataset, img_size, nclass, device):
    if model_name == 'vggnet':
        from models import vgg
        model = vgg.VGG('VGG11', num_classes=nclass)
    elif model_name == 'efficientnet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=200)
    elif model_name == 'resnet':
        from models import resnet
        model = resnet.ResNet18(num_classes=nclass)
        if dataset == 'tinyimagenet':
            from torchvision import models
            model = models.resnet18(pretrained=True)
            # model = resnet.ResNet18(num_classes=200)
    elif model_name == 'resnet34':
        from torchvision import models
        model = models.resnet34(pretrained=True)
            # model = resnet.ResNet18(num_classes=200)        
    elif model_name == 'wideresnet':
        from models import wideresnet
        model = wideresnet.WResNet_cifar10(num_classes=nclass, depth=16, multiplier=4)
        
    elif model_name == 'cnnlarge':
        from models import simple
        model = simple.CNNLarge()
        
    elif model_name == 'convmixer':
        from models import convmixer
        model = convmixer.ConvMixer(n_classes=nclass)
    
    elif model_name == 'cnn':
        from models import ModelCNNCifar10
        model = ModelCNNCifar10.ModelCNNCifar10()
        # if dataset == 'mnist':
        #     model = simple.CNNMnist(num_classes=nclass, num_channels=1)
        # elif dataset == 'fmnist':
        #     model = simple.CNNFashion_Mnist(num_classes=nclass)
        # elif dataset == 'cifar10':
        #     model = simple.CNNCifar10(num_classes=nclass)
        # elif dataset == 'cifar100':
        #     model = simple.CNNCifar100(num_classes=nclass)

    elif model_name == 'ae':
        from models import simple
        
        if dataset == 'mnist' or dataset == 'fmnist':
            model = simple.Autoencoder()
         
    elif model_name == 'mlp':
        from models import simple

        len_in = 1
        for x in img_size:
            len_in *= x
            model = simple.MLP(dim_in=len_in,
                               dim_out=nclass)
    
    elif model_name == 'mask':
        from models import masked_layers
        model = masked_layers.Mask4CNN(init='ME_init', activation='relu', device=device)

    elif model_name == 'dense':
        from models import dense_layers
        model = dense_layers.Dense4CNN(device=device)
    
    elif model_name == 'lstm':
        from models import simple
        model = simple.CharLSTM()

    else:
        exit('Error: unrecognized model')

    return model


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    # if args.dataset == 'shakespeare':
    #     train_dataset = ShakeSpeare(train=True)
    #     test_dataset = ShakeSpeare(train=False)
    #     user_groups = train_dataset.get_client_dic()
    #     # user_groups = user_groups[:50]
    #     args.num_users = len(user_groups)
    #     num_classes = 1
    #     if args.iid:
    #         exit('Error: ShakeSpeare dataset is naturally non-iid')
    #     else:
    #         print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
        

    # if args.dataset == 'tinyimagenet':
    #     transform_train = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #         ])
    #     transform_test = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ])

    #     train_dataset = datasets.ImageFolder('../data/tiny-imagenet-200/train', transform=transform_train)
    #     test_dataset = datasets.ImageFolder('../data/tiny-imagenet-200/val', transform=transform_test)
    #     if args.iid:
    #         # Sample IID user data from cifar
    #         user_groups = cifar_iid(train_dataset, args.num_users)
    #     else:
    #         # Sample Non-IID user data from cifar
    #         if args.unequal:
    #             # Chose uneuqal splits for every user
    #             raise NotImplementedError()
    #         else:
    #             # Chose euqal splits for every user
    #             user_groups = cifar_noniid(train_dataset, args.num_users, args.dir_alpha)
    #     # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #     num_classes = 200

    # elif args.dataset == 'femnist':
    #     train_dataset = FEMNIST(train=True)
    #     test_dataset = FEMNIST(train=False)
    #     user_groups = train_dataset.get_client_dic()
    #     args.num_users = len(user_groups)
    #     num_classes = 62
    #     if args.iid:
    #         exit('Error: femnist dataset is naturally non-iid')
    #     else:
    #         print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
            
    if args.dataset == 'cifar10' or 'cifar100':
        
        if args.model == 'cnn':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=14),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandAugment(num_ops=2, magnitude=14),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        # transform_train.transforms.insert(0, RandAugment(2, 14))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == 'cifar10':
            data_dir = '../data/cifar/'
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                           transform=transform_train)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=transform_test)
            num_classes = 10
        elif args.dataset == 'cifar100':    
            data_dir = '../data/cifar100/'
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                           transform=transform_train)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                          transform=transform_test)
            num_classes = 100
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users, args.dir_alpha)
    

#     if args.dataset == 'mnist' or 'fmnist':
#         apply_transform = transforms.Compose([
#             transforms.ToTensor(),
# #             transforms.Normalize((0.1307,), (0.3081,))
#         ])
        
#         if args.dataset == 'mnist':
#             data_dir = '../data/mnist/'
#             train_dataset = datasets.MNIST(data_dir, train=True, download=True,
#                                        transform=apply_transform)

#             test_dataset = datasets.MNIST(data_dir, train=False, download=True,
#                                       transform=apply_transform)
#         else:
#             data_dir = '../data/fmnist/'
#             train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
#                                        transform=apply_transform)

#             test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
#                                       transform=apply_transform)
        

#         train_dataset = datasets.MNIST(data_dir, train=True, download=True,
#                                        transform=apply_transform)

#         test_dataset = datasets.MNIST(data_dir, train=False, download=True,
#                                       transform=apply_transform)
#         num_classes = 10

        
#         # sample training data amongst users
#         if args.iid:
#             # Sample IID user data from Mnist
#             user_groups = mnist_iid(train_dataset, args.num_users)
#         else:
#             # Sample Non-IID user data from Mnist
#             user_groups = cifar_noniid(train_dataset, args.num_users, args.dir_alpha)
    
        

    return train_dataset, test_dataset, num_classes, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_parameter_delta(ws, w0):
    w_avg = copy.deepcopy(ws[0])
    for key in range(len(w_avg)):
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(0, len(ws)):
            w_avg[key] += ws[i][key] - w0[key]
        w_avg[key] = torch.div(w_avg[key], len(ws))
    return w_avg

def average_parameter(ws):
    w_avg = copy.deepcopy(ws[0])
    for key in range(len(w_avg)):
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(0, len(ws)):
            w_avg[key] += ws[i][key]
        w_avg[key] = torch.div(w_avg[key], len(ws))
    return w_avg

def average_parameter_dsize(ws,dsize):
    import numpy as np
    w_avg = copy.deepcopy(ws[0])
    for key in range(len(w_avg)):
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(0, len(ws)):
            w_avg[key] += ws[i][key]*dsize[i]
        w_avg[key] = torch.div(w_avg[key], np.sum(dsize))
    return w_avg

def average_parameter_delay(ws,delay):
    import numpy as np
    w_avg = copy.deepcopy(ws[0])
    for key in range(len(w_avg)):
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(0, len(ws)):
            w_avg[key] += ws[i][key]*delay[i]
        w_avg[key] = torch.div(w_avg[key], np.sum(delay))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def add_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z


def sub_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] - y[i])
    return z


def mult_param(alpha, x):
    z = []
    for i in range(len(x)):
        z.append(alpha*x[i])
    return z


def norm_of_param(x):
    z = 0
    for i in range(len(x)):
        z += torch.norm(x[i].flatten(0))
    return z

def l2_norm(x):
    z = 0
    for i in range(len(x)):
        z += torch.sum(x[i] ** 2)
    return torch.sqrt(z)
    
def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def quantize(x, qmin, qmax, num_bits):
    scale = (x.max() - x.min()) / (qmax - qmin)
    zero_point = -x.min() / scale
    q_x = torch.round(x / scale + zero_point)
    q_x.clamp_(qmin, qmax)
    return q_x, scale, zero_point

def dequantize(q_x, scale, zero_point):
    return scale * (q_x - zero_point)
import numpy as np

def generate_class_assignments(n_classes, n_clients, dir_alphas, seed):
    
    '''
    Assigns a class to each client based on a biased Dirichlet distribution.

    Parameters:
    n_classes (int): The number of classes.
    n_clients (int): The number of clients.
    dir_alphas (list): The alpha parameters for the Dirichlet distribution for each class.

    Returns:
    numpy.ndarray: An array of class assignments for each client.
    '''
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    # Check if dir_alphas length matches n_classes
    if len(dir_alphas) != n_classes:
        raise ValueError("Length of dir_alphas must be equal to n_classes")

    # Generate a Dirichlet distribution for each client
    probabilities = np.random.dirichlet(dir_alphas, n_clients)

    # Assign the class with the highest probability to each client
    class_assignments = np.argmax(probabilities, axis=1)

    return class_assignments


# class SymmetricUnbiasedQuantization(Function):
#     @staticmethod
#     def forward(ctx, input, num_bits):
#         # qmin = -2**(num_bits - 1)
#         # qmax = 2**(num_bits - 1) - 1
#         qmin = 0
#         qmax = 2**num_bits - 1
#         q_x, scale, zero_point = quantize(input, qmin, qmax, num_bits)
#         ctx.save_for_backward(q_x, scale, zero_point)
#         return dequantize(q_x, scale, zero_point)

#     @staticmethod
#     def backward(ctx, grad_output):
#         q_x, scale, zero_point = ctx.saved_tensors
#         grad_input = grad_output.clone() / scale
#         return grad_input, None

class Quantizer:
    def __init__(self, k, bucket_size):
        self.k = k
        self.bucket_size = bucket_size

    def quantize_bucket(self, a):
        raise NotImplementedError

    def quantize(self, a):
        if self.bucket_size == -1:
            return self.quantize_bucket(a)
        quantized = []
        # print("Length %d" % len(a))
        for i in range((len(a) + self.bucket_size - 1) // self.bucket_size):
            # print("quantize %d out of %d" % (i, (len(a) + self.bucket_size - 1) // self.bucket_size))
            quantized += self.quantize_bucket(a[i * self.bucket_size:min((i + 1) * self.bucket_size, len(a))])
        return quantized

class SymmetricUnbiasedQuantization(Quantizer):
    def __init__(self, k, bucket_size):
        super(SymmetricUnbiasedQuantization, self).__init__(k, bucket_size)
        self.k = k
        self.bucket_size = bucket_size

    def quantize_bucket(self, a):
        device = a[0].device
        fmin = torch.min(torch.cat(a))
        fmax = torch.max(torch.cat(a))
        res = []
        for i in range(len(a)):
            unit = (fmax - fmin) / (self.k - 1)
            if fmax - fmin == 0:
                q = fmin
            else:
                v = torch.floor((a[i] - fmin) / unit + torch.rand(a[i].shape, device=device))
                q = fmin + v * unit
            res.append(q)
        return res

# def data_participation_each_node(data_train, n_nodes):
#     dict_users, actual_label_distributions_each_node, num_labels = partition(data_train, n_nodes)
#     participation_prob_multiplier = np.random.dirichlet(participation_dirichlet_alpha * np.ones(num_labels), 1)
#     participation_scaling = participation_prob_mean * num_labels
#     participation_prob_each_node = [np.sum(np.multiply(actual_label_distributions_each_node[n], participation_prob_multiplier)) * participation_scaling for n in range(n_nodes)]
#     participation_prob_each_node = np.maximum(participation_prob_min, participation_prob_each_node)
#     print('participation_prob_each_node:', participation_prob_each_node, flush=True)
#     return dict_users, participation_prob_each_node