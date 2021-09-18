from __future__ import print_function
import yaml
import easydict
import os
import socket
import neptune
import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from apex import amp, optimizers
from data_loader.get_loader import get_dataloaders
from utils.utils import get_models, get_optimizers, log_set
from utils.lr_schedule import inv_lr_scheduler
from models.LinearAverage import LinearAverage
from eval import test, test_dev, test_and_nd
from trainer_funcs import train_adapt

def train(configs):
    print("random seed for data %s "%configs['random_seed'])

    source_loader, target_loader, \
    source_test_loader, test_loader = get_dataloaders(configs)
    filename, logger = log_set(configs)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if args.use_neptune:
        neptune.init(configs['path_neptune'])
        current_name = os.path.basename(__file__)
        PARAMS = {'learning rate': hp,
                  'method': configs['method'],
                  'machine': socket.gethostname(),
                  'gpu': gpu_devices[0],
                  'file': current_name,
                  'net': configs["conf"].model.base_model,
                  'eta': args.eta}
        print(PARAMS)
        neptune.create_experiment(name=filename, params=PARAMS)
        neptune.append_tag("config %s file %s" % (config_file,
                                                  filename))

    models, opt_g, opt_c, param_lr_g, param_lr_c = get_models(configs)
    if configs['method'] == 'NC':
        ## Memory
        ndata = target_loader.dataset.__len__()
        M = LinearAverage(configs['dim'], ndata,
                          configs['temp'], 0.0).cuda()
        models['M'] = M

    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    metric_names = {"neighborhood_density": "nd", "entropy": "ent",
               "source_risk": "s_risk", "dev_risk": "d_risk"}
    ## For s_risk, d_risk, ent, smaller is better.
    ## For neighborhood density, larger is better.
    metrics = {k: 1e5 for k in metric_names.keys()}
    metrics["neighborhood_density"] = 0
    acc_dict = {k: 0 for k in metric_names.keys()}
    iter_dict = {k: 0 for k in metric_names.keys()}
    acc_best = 0
    iter_best = 0
    ## set hyper parameter for this run.

    args.hp = float(hp)

    for step in range(conf.train.min_step + 1):
        models['G'].train()
        models['C'].train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_c, opt_c, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        index_t = data_t[2]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        index_t = Variable(index_t.cuda())
        opt_g.zero_grad()
        opt_c.zero_grad()
        ## Weight normalizztion
        models['C'].module.weight_norm()
        ## Source loss calculation
        feat_s = models['G'](img_s)
        out_s = models['C'](feat_s)
        loss_s = criterion(out_s, label_s)
        feat_t = models['G'](img_t)
        out_t = models['C'](feat_t)
        loss_adapt = train_adapt(configs['method'], feat_s,
                                 feat_t, out_t, index_t,
                                 models, conf, args)
        all_loss = loss_s + loss_adapt

        with amp.scale_loss(all_loss, [opt_g, opt_c]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()
        if configs['method'] == 'NC':
            models['M'].update_weight(feat_t, index_t)
        if step % conf.train.log_interval == 0:
            print('Train [{}/{} ({:.2f}%)]\tLoss Source: {:.6f} '
                  'Loss Adapt: {:.6f}\t'.format(
                step, conf.train.min_step,
                100 * float(step / conf.train.min_step),
                loss_s.item(), loss_adapt.item()))
        if step >= 1000 and step % conf.test.test_interval == 0:
            acc, nd, nd_nosoft, ent = test_and_nd(step,
                                                  test_loader,
                                                  filename,
                                                  models['G'],
                                                  models['C'])
            s_risk, d_risk = test_dev(step,
                                      source_test_loader,
                                      test_loader,
                                      source_loader,
                                      filename,
                                      models['G'],
                                      models['C'])

            output = "Iteration {} : current selected accs".format(step)
            for k, v in metric_names.items():
                value_rep = eval(v)
                if v is not "nd":
                    ## For s_risk, d_risk, ent, smaller is better.
                    if value_rep < metrics[k]:
                        metrics[k] = value_rep
                        acc_dict[k] = acc
                        iter_dict[k] = step
                elif v is "nd":
                    if value_rep > metrics[k]:
                        metrics[k] = value_rep
                        acc_dict[k] = acc
                        iter_dict[k] = step
                output += " {} : {} ".format(k, acc_dict[k])
            if acc > acc_best:
                acc_best = acc
                iter_best = step
            output += "Acc best: {}".format(acc_best)
            print(output)
            logger.info(output)
            if args.use_neptune:
                neptune.log_metric('neighborhood density',
                                   nd)
                neptune.log_metric('neighborhood density w/o softmax',
                                   nd_nosoft)
                neptune.log_metric('entropy',
                                   ent)
                neptune.log_metric('accuracy',
                                   acc)
                neptune.log_metric('weighted risk',
                                   d_risk)
                neptune.log_metric('source risk',
                                   s_risk)
            models['G'].train()
            models['C'].train()
    del models
    return acc_dict, metrics, iter_dict, acc_best, iter_best

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch UDA Validation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str,
                        default='config.yaml', help='/path/to/config/file')
    parser.add_argument('--source_path', type=str,
                        default='./utils/source_list.txt',
                        help='path to source list')
    parser.add_argument('--target_path', type=str,
                        default='./utils/target_list.txt',
                        help='path to target list')
    parser.add_argument('--log-interval', type=int,
                        default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--exp_name', type=str,
                        default='record', help='/path/to/config/file')
    parser.add_argument('--method', type=str, default='NC',
                        choices=['NC', 'PS', 'ENT', 'DANN'],
                        help='method name')
    parser.add_argument("--gpu_devices", type=int, nargs='+',
                        default=None, help="")
    parser.add_argument('--hps', type=float,
                        default=[1.0, 2.0, 1.5, 0.5],
                        nargs="*",
                        help='hyper parameter, specific to each method')
    parser.add_argument('--eta', type=float,
                        default=0.05, help='trade-off for loss')
    parser.add_argument("--random_seed", type=int,
                        default=0,
                        help='random seed to select source validation data')
    parser.add_argument("--use_neptune",
                        default=False, action='store_true')
    parser.add_argument("--path_neptune", type=str,
                        default="keisaito/sandbox",
                        help='path in neptune')

    args = parser.parse_args()
    config_file = args.config
    conf = yaml.load(open(config_file))
    save_config = yaml.load(open(config_file))
    conf = easydict.EasyDict(conf)
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    args.cuda = torch.cuda.is_available()
    use_gpu = torch.cuda.is_available()
    n_share = conf.data.dataset.n_share
    n_source_private = conf.data.dataset.n_source_private
    n_total = conf.data.dataset.n_total
    num_class = n_share + n_source_private
    script_name = os.path.basename(__file__)

    metric_names = {"neighborhood_density": "nd", "entropy": "ent",
                    "source_risk": "s_risk", "dev_risk": "d_risk"}
    all_metrics = {k:{} for k in metric_names.keys()}
    all_iters = {k:{} for k in metric_names.keys()}
    all_acc_dict = {k:{} for k in metric_names.keys()}
    acc_best_all = 0
    iter_best_all = 0

    configs = vars(args)
    configs["conf"] = conf
    configs["script_name"] = script_name
    configs["num_class"] = num_class
    configs["config_file"] = config_file
    configs["hp"] = float(args.hps[0])
    filename, _ = log_set(configs)
    dirname = os.path.dirname(filename)
    summary_path = os.path.join(dirname, "summary_result.txt")

    for hp in args.hps:
        configs["hp"] = float(hp)
        acc_choosen, metrics, iter_choosen, acc_best_gt, iter_best_gt = train(configs)
        if acc_best_gt > acc_best_all:
            acc_best_all = acc_best_gt
            iter_best_all = iter_best_gt
        for name in metric_names.keys():
            all_acc_dict[name][hp], \
            all_metrics[name][hp],\
            all_iters[name][hp] = acc_choosen[name], metrics[name], iter_choosen[name]
    print(all_metrics)
    print(all_acc_dict)
    for met, acc, in zip(all_metrics.keys(), all_acc_dict.keys()):
        dict_met = all_metrics[met]
        if met ==  'neighborhood_density':
            hp_best = max(dict_met, key=dict_met.get)
        else:
            hp_best = min(dict_met, key=dict_met.get)
        output = "metric %s selected hp %s iteration %s, " \
                 "metrics value %s acc choosen %s" % (met, hp_best,
                                                   all_iters[met][hp_best],
                                                   dict_met[hp_best],
                                                   all_acc_dict[met][hp_best])
        print(output)
        import logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename=summary_path, format="%(message)s")
        logger.setLevel(logging.INFO)
        print(output)
        logger.info(output)



