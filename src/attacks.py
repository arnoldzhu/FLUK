import torch
import torch.nn.functional as F
import numpy as np
from mxnet import nd
from copy import deepcopy
import math
torch.random.manual_seed(42)
ATTACK_RATE = 2

def full_trim_attack(v, f, dirs):
    '''
    Full-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised.
    v: the list of squeezed gradients
    f: the number of compromised worker devices
    '''
    shape = v[0].shape[0]
    # print(shape)

    # calculate update directions of benign nodes
    # grad_dir = torch.sign(torch.normal(0, 1, (shape, 1)))
    # up_dir = torch.maximum(grad_dir, torch.tensor([0]))
    # down_dir = -torch.minimum(grad_dir, torch.tensor([0]))
    up_dir = torch.maximum(dirs.cpu(), torch.tensor([0]))
    down_dir = -torch.minimum(dirs.cpu(), torch.tensor([0]))

    benign_mat = torch.cat(v[f:], dim=1).cpu()

    # calculate min/max signs of benign nodes
    max = torch.max(benign_mat, axis=1).values.reshape(shape, 1)
    max_dir = torch.sign(max)
    min = torch.min(benign_mat, axis=1).values.reshape(shape, 1)
    min_dir = torch.sign(min)

    # "one-hot vectors" for each of the four cases
    oh_first = down_dir * torch.maximum(max_dir, torch.tensor([0]))
    oh_second = down_dir * -torch.minimum(max_dir, torch.tensor([0]))
    oh_third = up_dir * torch.maximum(min_dir, torch.tensor([0]))
    oh_fourth = up_dir * -torch.minimum(min_dir, torch.tensor([0]))

    for i in range(f):
        # generate a new parameter vector for every malicious node
        rand1 = torch.empty(shape, 1).uniform_(1, ATTACK_RATE)
        rand2 = torch.empty(shape, 1).uniform_(1/ATTACK_RATE, 1)

        vec_first = max * rand1 * oh_first
        vec_second = max * rand2 * oh_second
        vec_third = min * rand2 * oh_third
        vec_fourth = min * rand1 * oh_fourth

        v[i] = (vec_first + vec_second + vec_third + vec_fourth).cuda()

    return v       

def partial_trim_attack(v, f):
    '''
    Partial-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised. 
    v: the list of squeezed gradients
    f: the number of compromised worker devices
    '''
    shape = v[0].shape[0]
    malicious_params = torch.cat(v[:f], dim=1).cpu()
    mu = torch.mean(malicious_params, axis=1)
    sigma = torch.sqrt(torch.sum(torch.square(malicious_params - mu.reshape(mu.shape[0], 1)), axis=1) / f)  

    grad_dir = torch.sign(torch.normal(0, 1, (shape, 1)))
    up_dir = torch.maximum(grad_dir, torch.tensor([0]))
    down_dir = -torch.minimum(grad_dir, torch.tensor([0]))

    for i in range(f):
        rand_down = torch.tensor(nd.random.uniform(nd.array((mu+3*sigma).numpy()),
            nd.array((mu+4*sigma).numpy()), (1, 1)).reshape(shape, 1).asnumpy())
        rand_up = torch.tensor(nd.random.uniform(nd.array((mu-4*sigma).numpy()),
            nd.array((mu-3*sigma).numpy()), (1, 1)).reshape(shape, 1).asnumpy())

        v[i] = (rand_down * down_dir + rand_up * up_dir).cuda()

    return v

def euclidean(a, b):
    a = a.view(-1)
    b = b.view(-1)
    # print(a.shape, b.shape)

    dis = torch.sqrt(torch.sum(torch.square(a - b)))
    return dis

def my_multi_krum(updates, v, f, multi=False):
    distances = {}
    num_closest = v - f - 2

    for i in range(v):
        for j in range(i+1, len(updates)):
            distances[str(i)+'_'+str(j)] = euclidean(updates[i], updates[j])

    krum_score = []
    for i in range(v):
        nearest = []
        for j in range(v):
            if i != j:
                if str(i)+'_'+str(j) in distances.keys():
                    nearest.append(distances[str(i)+'_'+str(j)])
                else:
                    nearest.append(distances[str(j)+'_'+str(i)])
        
        nearest.sort()
        squ_nearest = [num*num for num in nearest[:num_closest]]
        krum_score.append(sum(squ_nearest))

    return updates[krum_score.index(min(krum_score))], krum_score.index(min(krum_score))


def my_full_krum_attack(updates, v, f, dirs, multi=False, prev_model=None, eps=0.01):
    d = updates[0].shape[0]
    found_lamb = False

    # calculate the upper bound of lambda
    # first term
    distance_sums = []
    for i in range(f, v):
        # calculate the smallest v-f-2 distances between i and the other benign nodes
        distances = []
        for j in range(f, v):
            if i != j:
                distances.append(euclidean(updates[i], updates[j]) * euclidean(updates[i], updates[j]))
        distances.sort()
        distance_sums.append(sum(distances[:v-f-2]))

    upper_bound = math.sqrt(1/((v-2*f-1)*d) * min(distance_sums))

    # second term
    max_dis = float('-inf')
    for i in range(f, v):
        dis = euclidean(updates[i], prev_model)
        max_dis = dis if dis > max_dis else max_dis
    
    upper_bound += 1/math.sqrt(d) + max_dis
    thr = 1e-5
    lamb = upper_bound

    while lamb > thr:
        candidate = prev_model - lamb * dirs
        expanded_updates = deepcopy(updates)
        expanded_updates.append(candidate)
        update, index = my_multi_krum(expanded_updates, v, f, False)
        if index == len(expanded_updates) - 1:
            # found the correct update, apply epsilon to the solution
            found_lamb = True
            break

        lamb /= 2

    for i in range(f):
        factor = torch.rand(d, 1) * 0.02 + 0.99
        updates[i] = prev_model - lamb * dirs * factor.cuda()

    return found_lamb, updates

def my_partial_krum_attack(updates, v, f, multi=False, prev_model=None):
    d = updates[0].shape[0]
    malicious_updates = updates[:f]
    malicious_mat = torch.cat(malicious_updates, axis=1)
    mean_update = torch.mean(malicious_mat, axis=1)
    dirs = torch.sign(mean_update).reshape(d, 1)
    
    found_lamb, added_updates = False, 1
    
    while found_lamb is False:
        # calculate the upper bound of lambda (treat malicious updates as benign ones)
        # first term
        distance_sums = []
        for i in range(f):
            # calculate the distances between malicious nodes
            distances = []
            for j in range(f):
                if i != j:
                    distances.append(euclidean(malicious_updates[i], malicious_updates[j]) * euclidean(malicious_updates[i], malicious_updates[j]))
            distances.sort()
            distance_sums.append(sum(distances))

        upper_bound = math.sqrt((1 / (f * d))* min(distance_sums))

        # second term
        max_dis = float('-inf')
        for i in range(f):
            dis = euclidean(malicious_updates[i], prev_model)
            max_dis = dis if dis > max_dis else max_dis
        
        upper_bound += 1/math.sqrt(d) + max_dis
        thr = 1e-5
        lamb = upper_bound

        while lamb > thr:
            candidate = prev_model - lamb * dirs
            expanded_updates = deepcopy(malicious_updates)
            for i in range(added_updates):
                expanded_updates.append(candidate)
            update, index = my_multi_krum(expanded_updates, f+added_updates, f, False)
            if index >= f:
                # found the correct update, apply epsilon to the solution
                found_lamb = True
                break

            lamb /= 2
            
        added_updates += 1

    for i in range(f):
        factor = torch.rand(d, 1) * 0.02 - 0.01
        updates[i] = prev_model - lamb * dirs * factor.cuda()

    return lamb, updates

def gaussian_attack(updates, f):
    shape = updates[0].shape[0]
    cat_updates = torch.cat(updates, dim=1).cpu()
    mu = torch.mean(cat_updates, dim=1)
    sigma = torch.sqrt(torch.sum(torch.square(cat_updates - mu.reshape(-1, 1)), axis=1) / f)
    
    for i in range(f):
        updates[i] = torch.tensor(nd.random.normal(nd.array(mu.numpy()),
                nd.array(sigma.numpy()), (1, 1)).reshape(shape, 1).asnumpy()).cuda()
    
    return updates

def alie_attack(updates, n, f):
    s = math.floor(n/2+1) - f 
    z = 0.35
    shape = updates[0].shape[0]
    cat_updates = torch.cat(updates, dim=1).cpu()
    mu = torch.mean(cat_updates, dim=1)
    sigma = torch.sqrt(torch.sum(torch.square(cat_updates - mu.reshape(-1, 1)), axis=1) / f)

    corr_weight = mu + z * sigma

    for i in range(f):
        updates[i] = deepcopy(corr_weight).cuda()

    return updates
 

def JSConvergence(p, q, reduction='batchmean'):
    p_out = p.view(-1).softmax(dim=-1)
    q_out = q.view(-1).softmax(dim=-1)

    log_mean = ((p_out + q_out) / 2).log()

    return (F.kl_div(log_mean, p_out, reduction=reduction) + 
                    F.kl_div(log_mean, q_out, reduction=reduction)) / 2
