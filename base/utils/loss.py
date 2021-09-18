import torch
import torch.nn.functional as F

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en

def pseudo_loss(p, thr=0.9):
    p = F.softmax(p)
    p_max = p.max(1)[0]
    positive = p_max > thr
    loss = -torch.log(p + 1e-5)
    loss = loss.min(1)[0]
    tmp = torch.zeros(loss.size(0)).float().cuda()
    if torch.sum(positive.int()) > 0:
        tmp[positive] = loss[positive]
    return torch.mean(tmp)

def pseudo_loss_new(logits, thr=0.9):
    prob = F.softmax(logits)
    max_probs, targets_u = torch.max(prob, dim=-1)
    mask = max_probs.ge(thr).float()
    Lu = (F.cross_entropy(logits, targets_u,
                      reduction='none') * mask).mean()
    return Lu
