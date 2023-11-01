import os
import copy
import time
import datetime
import numpy as np
from tqdm import tqdm
import sys
import collections
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference, my_test
from models import MLP, MNIST_CNN, MNIST_2NN, CNN_FMNIST, CNNMnist, CNNFashion_Mnist, CNNCifar, DAVE2
from utils import get_dataset, average_weights, exp_details, get_malicious_scores
from attacks import full_trim_attack, partial_trim_attack, gaussian_attack, my_full_krum_attack, my_partial_krum_attack, alie_attack

NUM_BYZS = 2
WINDOW_SIZE = 5

def get_directions(prev_params, curr_params):
    prev, curr = None, None
    for k in prev_params.keys():
        prev = prev_params[k].view(-1) if prev is None else torch.cat((prev, prev_params[k].view(-1)), 0)
        curr = curr_params[k].view(-1) if curr is None else torch.cat((curr, curr_params[k].view(-1)), 0)
    
    return torch.sign(prev-curr).reshape(-1, 1)

def make_print_to_file(args):
    path = args.stdout
    if not os.path.isdir(path):
        os.mkdir(path, mode=0o755)
    
    class Logger(object):
        def __init__(self, filename, path):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
            
        def write(self, message):
            self.terminal.write(message)    
            self.log.write(message)
            
        def flush(self):
            pass
        
    filename = datetime.datetime.now().strftime('day' + '%Y_%m_%d_' + args.attack_type + '_' + str(args.nbyz))
    sys.stdout = Logger(filename + '.log', path=path)
    print(filename.center(60, '*'))

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    make_print_to_file(args)
    exp_details(args)

    device = torch.device('cuda:0' if args.gpu else 'cpu')
    print(args)
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
    prev_weights, dir = global_weights, None
    prev_client_weights, corrupted_weights = [], None
    m_b_kl, b_b_kl = [], []
    hor_b_kl, hor_m_kl = [], []
    # m_b_norm, b_b_norm = [], []
    ver_m_kl, ver_b_kl = [], []

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

            attack = False
            if args.attack_type == 'label_flipping' and idx < args.nbyz:
                attack = True
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, attack=attack)

            # print('\n\n\n\ngrad:')
            grad_dict = collections.OrderedDict()
            for k, v in prev_params.items():
                grad_dict[k] = w[k] - v

            # print(grad_dict)

            # print(sys.getsizeof(w))
            # print('wwwwwww', type(w))
            # input()
            local_weights_list.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        clean_weights = []
        len_dict = {}
        for weights in local_weights_list:
            params = None
            for key, param in weights.items():
                params = param.view(-1) if params is None else torch.cat((params, param.view(-1)), 0)
                len_dict[key] = (param.shape, len(param.view(-1)))
            clean_weights.append(params.reshape(-1, 1))

        # ********* perform attacks here *********
        if epoch > 0:
            print('performing attacks...')
            
            prev_model = None
            for k, v in global_model.state_dict().items():
                prev_model = v.view(-1) if prev_model is None else torch.cat((prev_model, v.view(-1)), 0)
            prev_model = prev_model.reshape(-1, 1)
            
            found = None
            if args.attack_type == 'full_trim':
                corrupted_weights = full_trim_attack(copy.deepcopy(clean_weights), args.nbyz, dir)
            elif args.attack_type == 'partial_trim':
                corrupted_weights = partial_trim_attack(copy.deepcopy(clean_weights), args.nbyz)
            elif args.attack_type == 'full_krum':
                found, corrupted_weights = my_full_krum_attack(copy.deepcopy(clean_weights), len(clean_weights), args.nbyz, dir, False, prev_model)
            elif args.attack_type == 'partial_krum':
                _, corrupted_weights = my_partial_krum_attack(copy.deepcopy(clean_weights), len(clean_weights), args.nbyz, False, prev_model)
            elif args.attack_type == 'gaussian':
                corrupted_weights = gaussian_attack(copy.deepcopy(clean_weights), args.nbyz)
            elif args.attack_type == 'label_flipping':
                corrupted_weights = clean_weights
            elif args.attack_type == 'alie_attack':
                corrupted_weights = alie_attack(copy.deepcopy(clean_weights), args.num_users, args.nbyz)
            elif args.attack_type == 'no_attack':
                corrupted_weights = clean_weights
            corrupted_dicts = []    # convert corrupt parameters into dicts
            if found is not None:
                print('lambda found') if found else print('lambda not solved')

            avg, benign, malicious = None, None, None
            for i, vec in enumerate(corrupted_weights):
                sd, ckpt = {}, 0
                for key, value in len_dict.items():
                    shape, length = value
                    sd[key] = vec[ckpt: ckpt+length].reshape(shape)
                    ckpt += length
                corrupted_dicts.append(sd)
                # print(vec.shape)
                vec = vec.reshape(vec.view(-1).shape)
                # print(vec.shape)

                if avg is None:
                    avg = copy.deepcopy(vec)
                else:
                    avg += copy.deepcopy(vec)
                
                if i < args.nbyz:
                    if malicious is None:
                        malicious = copy.deepcopy(vec)
                    else:
                        malicious += copy.deepcopy(vec)
                    print('m', i, malicious)
                else:
                    if benign is None:
                        benign = copy.deepcopy(vec)
                    else:
                        benign += copy.deepcopy(vec)
                    print('b', i, benign)
            
            avg /= len(corrupted_weights)
            malicious /= args.nbyz
            benign /= args.num_users - args.nbyz

            malicious -= avg
            benign -= avg

            ckpt, avg_dict, mali_dict, beni_dict = 0, {}, {}, {}
            for key, value in len_dict.items():
                shape, length = value
                if len(shape) == 1:
                    print(shape[0])
                    shape = torch.Size([shape[0], 1])
                # print('shape', shape, len(shape))
                avg_dict[key] = avg[ckpt: ckpt+length].reshape(shape)
                mali_dict[key] = malicious[ckpt: ckpt+length].reshape(shape)
                beni_dict[key] = benign[ckpt: ckpt+length].reshape(shape)
                ckpt += length
            
            print('attack complete!')

        # ******** perform detection here *********
        if epoch > 0:
            kl_dict, m_b_list, b_b_list = {}, [], []
            # norm_mb_list, norm_bb_list = [], []
            hor_m_list, hor_b_list = [], []
            ver_m_list, ver_b_list = [], []
            hor_kl_record = [[None for _ in range(len(corrupted_weights))] for _ in range(len(corrupted_weights))]
            ver_kl_record = [[] for _ in range(len(corrupted_weights))]
            flag = 0

            # horizontal detection
            for i in range(len(corrupted_weights)):
                for j in range(i+1, len(corrupted_weights)):
                    if i == 0 and j == args.nbyz:
                        flag = 1
                    if i == args.nbyz and j == args.nbyz+1:
                        flag = 2
                    kl_score = F.kl_div(corrupted_weights[i].view(-1).softmax(dim=-1).log(), corrupted_weights[j].view(-1).softmax(dim=-1), reduction='batchmean')
                    kl_dict[str(i)+','+str(j)] = kl_score
                    hor_kl_record[i][j] = kl_score.cpu()
                    if flag == 1:
                        m_b_list.append(torch.tensor(kl_score).view(-1))
                    elif flag == 2:
                        b_b_list.append(torch.tensor(kl_score).view(-1))

            # new horizontal detection
            for i in range(len(corrupted_weights)):
                for j in range(len(corrupted_weights)):
                    if i <= j:
                        kl_score = F.kl_div(corrupted_weights[i].view(-1).softmax(dim=-1).log(), corrupted_weights[j].view(-1).softmax(dim=-1), reduction='batchmean')
                    else:
                        kl_score = F.kl_div(corrupted_weights[j].view(-1).softmax(dim=-1).log(), corrupted_weights[i].view(-1).softmax(dim=-1), reduction='batchmean')
                    if i < args.nbyz:
                        hor_m_list.append(torch.tensor(kl_score).view(-1))
                    else:
                        hor_b_list.append(torch.tensor(kl_score).view(-1))

            # vertical detection
            for i, curr_weights in enumerate(corrupted_weights):  # corrupted shape: [num_cli, tensor(-1, 1)]
                print(torch.sum(curr_weights.view(-1).softmax(dim=-1)))
                # assert (torch.sum(curr_weights.view(-1).softmax(dim=-1)) - 1 < 1e-10)
                kl_score = F.kl_div(curr_weights.view(-1).softmax(dim=-1).log(), prev_client_weights[i].view(-1).softmax(dim=-1), reduction='batchmean')
                if i < args.nbyz:  # malicious
                    ver_m_list.append(torch.tensor(kl_score).view(-1))
                else:            # benign
                    ver_b_list.append(torch.tensor(kl_score).view(-1))
                ver_kl_record[i].append(kl_score.cpu())
                if len(ver_kl_record[i]) > WINDOW_SIZE:
                    del(ver_kl_record[i][0])

            b_b_kl.append(torch.mean(torch.cat(b_b_list)))
            m_b_kl.append(torch.mean(torch.cat(m_b_list)))
            hor_b_kl.append(torch.mean(torch.cat(hor_b_list)))
            hor_m_kl.append(torch.mean(torch.cat(hor_m_list)))
            ver_b_kl.append(torch.mean(torch.cat(ver_b_list)))
            ver_m_kl.append(torch.mean(torch.cat(ver_m_list)))
            print('mali x beni:', torch.mean(torch.cat(m_b_list)))
            print('beni x beni:', torch.mean(torch.cat(b_b_list)))

        # ******** calculate reputation scores here *********
        if epoch > 0:
            kl_mean_2d = []
            for i in range(len(corrupted_weights)):
                # record the horizontal and vertical mean kl scores
                hor_temp_list = []
                for j in range(len(corrupted_weights)):
                    if hor_kl_record[i][j] != None:
                        hor_temp_list.append(hor_kl_record[i][j])
                    elif hor_kl_record[j][i] != None:
                        hor_temp_list.append(hor_kl_record[j][i])
                    # else:
                    #     print('error! error!')
                    #     input()
                kl_mean_2d.append((np.mean(hor_temp_list), np.mean(ver_kl_record[i])))

            malicious_scores = get_malicious_scores(kl_mean_2d, 2)
            # print(kl_mean_2d)
            # print(len(kl_mean_2d))
            # input()

        # ******** record weights for next round detection *********
        prev_client_weights = clean_weights if epoch == 0 else corrupted_weights
                    
        if epoch == 0:
            global_weights = average_weights(local_weights_list)
        else:    
            global_weights = average_weights(corrupted_dicts)
        dir = get_directions(prev_weights, global_weights)
        prev_weights = global_weights

        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
