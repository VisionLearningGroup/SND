from utils.loss import entropy, pseudo_loss_new
import torch
import torch.nn.functional as F


def train_adapt(method, feat_s, feat_t, out_t, index_t, models, conf, args):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if method == 'NC':
        feat_t = F.normalize(feat_t)
        ### Calculate mini-batch x memory similarity
        feat_mat = models['M'](feat_t, index_t)
        ### We do not use memory features present in mini-batch
        feat_mat[:, index_t] = -1 / conf.model.temp
        ### Calculate mini-batch x mini-batch similarity
        feat_mat2 = torch.matmul(feat_t,
                                 feat_t.t()) / conf.model.temp
        mask = torch.eye(feat_mat2.size(0),
                         feat_mat2.size(0)).bool().cuda()
        feat_mat2.masked_fill_(mask, -1 / conf.model.temp)
        loss_adapt = args.eta * entropy(torch.cat([feat_mat,
                                                   feat_mat2,
                                                   out_t], 1) * args.hp)
        all = loss_adapt

    elif method == 'DANN':
        d_s = models['D'](feat_s)
        d_t = models['D'](feat_t)
        domain_label = torch.cat([torch.zeros(d_s.size(0)),
                                  torch.ones(d_t.size(0))], 0).long().cuda()
        loss_adapt = criterion(torch.cat([d_s, d_t], 0), domain_label)
        gamma = 10
        power = 0.75
        max_iter = 10000.
        eta = 1 - (1 + gamma * min(1.0, step / max_iter)) ** (-power)
        all = loss_adapt * eta * args.hp

    elif method == 'ENT':
        loss_adapt = args.hp * entropy(out_t)
        all = loss_adapt

    elif method == 'PS':
        loss_adapt = 0.05 * pseudo_loss_new(out_t, args.hp)
        all = loss_adapt
    else:
        all = 0
    return all