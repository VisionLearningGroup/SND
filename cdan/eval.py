import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import logging
from dev import get_weight, get_dev_risk

def feat_get(model, dataset_source, dataset_target):
    #G.eval()

    for batch_idx, data in enumerate(dataset_source):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_s = data[0]
            label_s = data[1]
            img_s, label_s = Variable(img_s.cuda()), \
                             Variable(label_s.cuda())
            feat_s, _ = model(img_s)
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
            feat_t, _ = model(img_t)
            if batch_idx == 0:
                feat_all = feat_t.data.cpu().numpy()
                label_all = label_t.data.cpu().numpy()
            else:
                feat_t = feat_t.data.cpu().numpy()
                label_t = label_t.data.cpu().numpy()
                feat_all = np.r_[feat_all, feat_t]
                label_all = np.r_[label_all, label_t]
    return feat_all_s, feat_all


def test_dev(step, loaders, filename, model):
    source_test_loader, target_test_loader, source_loader = loaders["source_test"], loaders["test"], loaders["source"]
    model.train(False)
    for batch_idx, data in enumerate(source_test_loader):
        with torch.no_grad():
            img_t, label_t = data[0], data[1]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat, out_t = model(img_t)
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
    print('source valid error %s'%error.mean())
    print("compute features")
    source_feats, target_feats = feat_get(model, source_loader, target_test_loader)
    print("compute proxy risk")
    dev_risk = get_dev_risk(get_weight(source_feats, target_feats, feat_all), error)
    print('dev risk %s'%dev_risk)
    output = [step, 'source valid error %s'%error.mean(), 'dev risk %s'%dev_risk]
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="%(message)s")
    logger.setLevel(logging.INFO)
    print(output)
    logger.info(output)
    return error.mean(), dev_risk
