import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from mmd_comp import MultipleKernelMaximumMeanDiscrepancy, JointMultipleKernelMaximumMeanDiscrepancy
from kernels import GaussianKernel

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 
def virtual_loss(model,feat,lambd=0.1,T=0.05,k=2,eta=0.05,conf=False):
    w_temp = model.fc.weight
    #feat = F.normalize(feat)
    sum = False
    loss = 0
    for i in range(k):
        model.zero_grad()
        w_temp.requires_grad_()
        out_t1 = torch.mm(feat.detach(), w_temp.t())
        out_t1 = F.softmax(out_t1)
        size_t = out_t1.size(0)
        loss_d = torch.sum(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))/size_t
        loss -= loss_d
        loss_d.backward(retain_graph=True)
        #pdb.set_trace()
        w_delta = -w_temp.grad * eta#-F.normalize(w_temp.grad) * torch.norm(w_temp,dim=1).view(w_temp.size(0),1)*eta
        w_temp_delta = w_delta #+ F.normalize(w_temp)*torch.sum(torch.mm(w_temp, w_delta.t()),1).view(-1,1)
        w_temp = w_temp + w_temp_delta
        w_temp = Variable(w_temp)
        w_temp.requires_grad_()
    out_d = F.softmax(torch.mm(feat, w_temp.t()))
    loss = -lambd * torch.sum(torch.sum(out_d * (torch.log(out_d + 1e-5)), 1))/size_t
    return loss

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduce=False)(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def DAN(features_s, features_t):
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=not False, quadratic_program=False
    )
    return mkmmd_loss(features_s, features_t)

def JAN(features_s, features_t, output_s, output_t):
    thetas = None

    jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
        kernels=(
            [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            (GaussianKernel(sigma=0.92, track_running_stats=False),)
        ),
        linear=False, thetas=thetas
    ).cuda()
    transfer_loss = jmmd_loss(
        (features_s, F.softmax(output_s, dim=1)),
        (features_t, F.softmax(output_t, dim=1))
    )
    return transfer_loss
