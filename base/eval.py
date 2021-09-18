import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils.dev import get_weight, get_dev_risk
from utils.loss import entropy



def neighbor_density(feature, T=0.05):
    feature = F.normalize(feature)
    mat = torch.matmul(feature, feature.t()) / T
    mask = torch.eye(mat.size(0), mat.size(0)).bool()
    mat.masked_fill_(mask, -1 / T)
    result = entropy(mat)
    return result

def test(step, dataset_test, filename, n_share, unk_class, G, C1):
    G.eval()
    C1.eval()
    correct = 0
    correct_close = 0
    size = 0
    class_list = [i for i in range(n_share)]
    class_list.append(unk_class)
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    per_class_correct_cls = np.zeros((n_share + 1)).astype(np.float32)
    all_pred = []
    all_gt = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0], data[1]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            out_t = C1(feat)
            out_t = F.softmax(out_t)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            pred = out_t.data.max(1)[1]
            k = label_t.data.size()[0]
            pred_cls = pred.cpu().numpy()
            pred = pred.cpu().numpy()
            pred_unk = np.where(entr > threshold)
            pred[pred_unk[0]] = n_share
            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == i)
                correct_ind_close = np.where(pred_cls[t_ind[0]] == i)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_correct_cls[i] += float(len(correct_ind_close[0]))
                per_class_num[i] += float(len(t_ind[0]))
                correct += float(len(correct_ind[0]))
                correct_close += float(len(correct_ind_close[0]))
            size += k
    per_class_acc = per_class_correct / per_class_num
    close_p = float(per_class_correct_cls.sum() / per_class_num.sum())
    print(
        '\nTest set :  Accuracy: {}/{} ({:.0f}%)  '
        '({:.4f}%)\n'.format(
            correct, size,
            100. * correct / size, float(per_class_acc.mean())))
    output = [step, list(per_class_acc), 'per class mean acc %s'%float(per_class_acc.mean()),
              float(correct / size), 'closed acc %s'%float(close_p)]
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="%(message)s")
    logger.setLevel(logging.INFO)
    print(output)
    logger.info(output)
    return float(correct / size), float(close_p)


def feat_get(G, dataset_source, dataset_target):
    G.eval()

    for batch_idx, data in enumerate(dataset_source):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_s = data[0]
            label_s = data[1]
            img_s, label_s = Variable(img_s.cuda()), \
                             Variable(label_s.cuda())
            feat_s = G(img_s)
            if batch_idx == 0:
                feat_all_s = feat_s.data.cpu().numpy()
                label_all_s = label_s.data.cpu().numpy()
            else:
                feat_s = feat_s.data.cpu().numpy()
                label_s = label_s.data.cpu().numpy()
                feat_all_s = np.r_[feat_all_s, feat_s]
                label_all_s = np.r_[label_all_s, label_s]
    for batch_idx, data in enumerate(dataset_target):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_t = data[0]
            label_t = data[1]
            img_t, label_t = Variable(img_t.cuda()), \
                             Variable(label_t.cuda())
            feat_t = G(img_t)
            if batch_idx == 0:
                feat_all = feat_t.data.cpu().numpy()
                label_all = label_t.data.cpu().numpy()
            else:
                feat_t = feat_t.data.cpu().numpy()
                label_t = label_t.data.cpu().numpy()
                feat_all = np.r_[feat_all, feat_t]
                label_all = np.r_[label_all, label_t]
    return feat_all_s, feat_all




def test_and_nd(step, dataset_test, name, G, C):
    G.eval()
    C.eval()
    correct = 0
    size = 0
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t  = data[0], data[1]
            img_t, label_t = Variable(img_t.cuda(), volatile=True), \
                             Variable(label_t.cuda(), volatile=True)
            feat = G(img_t)
            out_t = C(feat).cpu()
            pred = out_t.data.max(1)[1]
            correct += pred.eq(label_t.data.cpu()).cpu().sum()
            k = label_t.data.size()[0]
            size += k
            if batch_idx == 0:
                label_all = label_t
                feat_all = feat
                pred_all = out_t
            else:
                pred_all = torch.cat([pred_all, out_t],0)
                feat_all = torch.cat([feat_all, feat],0)
                label_all = torch.cat([label_all, label_t],0)
    print(
        '\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, size,
            100. * correct / size))
    ## Accuracy
    close_p = 100. * float(correct) / float(size)
    #compute_variance(pred_all, label_all)
    #compute_variance(feat_all, label_all)
    ## Entropy
    ent_class = entropy(pred_all)

    ## Neighborhood Density
    pred_soft = F.softmax(pred_all)
    nd_soft = neighbor_density(pred_soft)

    ## Neighborhood Density without softmax
    nd_nosoft = neighbor_density(pred_all)

    output = [step, "closed", "acc %s"%float(close_p),
              "neighborhood density %s"%nd_soft.item(),
              "neighborhood density no soft%s" % nd_nosoft.item(),
              "entropy class %s"%ent_class.item()]

    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    print(output)
    logger.info(output)
    return close_p, nd_soft.item(), nd_nosoft.item(), ent_class.item()

def test_dev(step, source_test_loader, target_test_loader, source_loader, filename, G, C1):
    G.eval()
    C1.eval()
    for batch_idx, data in enumerate(source_test_loader):
        with torch.no_grad():
            img_t, label_t = data[0], data[1]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            out_t = C1(feat)
            out_t = F.softmax(out_t)
            pred = out_t.data.max(1)[1]
            pred = pred.cpu().numpy()
            feat = feat.data.cpu().numpy()
            if batch_idx == 0:
                pred_all = pred
                feat_all = feat
                label_all = label_t.data.cpu().numpy()
            else:
                pred_all = np.r_[pred_all, pred]
                feat_all = np.r_[feat_all, feat]
                label_all = np.r_[label_all, label_t.data.cpu().numpy()]
    error = (pred_all != label_all)
    error = error.reshape((error.shape[0],1))
    #print('source valid error %s'%error.mean())
    #print("compute features")
    source_feats, target_feats = feat_get(G, source_loader, target_test_loader)
    #print("compute proxy risk")
    dev_risk = get_dev_risk(get_weight(source_feats, target_feats, feat_all), error)
    #print('dev risk %s'%dev_risk)
    return error.mean(), dev_risk


def compute_features(G, dataset_source, dataset_target):
    G.eval()
    feat_st_s = {}
    feat_st_sv = {}
    feat_st_t = {}
    feat_st_tv = {}

    for batch_idx, data in enumerate(dataset_source):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_s = data[0]
            label_s = data[1]
            img_s, label_s = Variable(img_s.cuda()), \
                             Variable(label_s.cuda())
            _, feats = G(img_s, return_lower=True)
            for i, feat in enumerate(feats):
                if batch_idx == 0:
                    feat_st_s[i] = 0
                    feat_st_sv[i] = 0
            for i, feat in enumerate(feats):
                #import pdb
                #pdb.set_trace()

                feat_st_s[i] += feat.mean(3).mean(2).mean(1).mean() / len(dataset_source)
                feat_st_sv[i] += feat.mean(3).mean(2).mean(1).var() / len(dataset_source)
    for batch_idx, data in enumerate(dataset_target):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_t = data[0]
            label_t = data[1]
            img_t, label_t = Variable(img_t.cuda()), \
                             Variable(label_t.cuda())
            _, feats = G(img_t, return_lower=True)
            for i, feat in enumerate(feats):
                if batch_idx == 0:
                    feat_st_t[i] = 0
                    feat_st_tv[i] = 0
            for i, feat in enumerate(feats):
                feat_st_t[i] += feat.mean(3).mean(2).mean(1).mean() / len(dataset_target)
                feat_st_tv[i] += feat.mean(3).mean(2).mean(1).var() / len(dataset_target)
    for key in feat_st_t.keys():
        print("layer %s source mean %s var %s target mean %s var %s"%(key, feat_st_s[key],  feat_st_sv[key],
                                                                      feat_st_t[key], feat_st_tv[key]))
    return feat_st_s, feat_st_t
