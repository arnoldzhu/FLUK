import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import sys
import collections

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference, my_test
from models import MLP, MNIST_CNN, MNIST_2NN, CNN_FMNIST, CNNMnist, CNNFashion_Mnist, CNNCifar, DAVE2
from utils import get_dataset, average_weights, exp_details
from attacks import full_trim_attack, partial_trim_attack

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = torch.device('cuda:0' if args.gpu else 'cpu')

    print(args)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = MNIST_CNN()
            # global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNN_FMNIST(args=args)
            # global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    
    elif args.model == 'dave':
        global_model = DAVE2()

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        
        if args.dataset == 'mnist':
            global_model = MNIST_2NN()
        else:
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    # print(global_weights)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 2

    for epoch in tqdm(range(args.epochs)):
        local_weights_list, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = [i for i in range(args.num_users)]

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            prev_params = global_model.state_dict()
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            grad_dict = collections.OrderedDict()
            for k, v in prev_params.items():
                grad_dict[k] = w[k] - v
            
            local_weights_list.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights_list)

        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = my_test(args, global_model, test_dataset)
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
