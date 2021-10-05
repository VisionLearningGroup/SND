import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
#import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import torch.nn.functional as F

import data_list
from data_list import ImageList
from tensorboardX import SummaryWriter
from data_list import ImageList, split_train_test
from eval import test_dev

import socket
import neptune
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import random
import pdb
import math

def entropy(p, mean=True):
    p = F.softmax(p)
    if not mean:
        return -torch.sum(p * torch.log(p+1e-5), 1)
    else:
        return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))


def ent(p, mean=True):
    if not mean:
        return -torch.sum(p * torch.log(p+1e-5), 1)
    else:
        return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

def image_classification_test(loader, model, test_10crop=False):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                soft_output = F.softmax(outputs)
                if start_test:
                    all_output = outputs.float().cuda()
                    all_output_soft = soft_output.float().cuda()
                    all_label = labels.float().cuda()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cuda()), 0)
                    all_output_soft = torch.cat((all_output_soft, soft_output.float().cuda()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0).cuda()

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    ent_class = ent(all_output_soft)
    normalized = F.normalize(all_output_soft).cpu()  # .cpu()
    mat = torch.matmul(normalized, normalized.t()) / 0.05
    mask = torch.eye(mat.size(0), mat.size(0)).bool()#.cuda()
    mat.masked_fill_(mask, -1 / 0.05)
    ent_soft = entropy(mat)
    normalized_nosoft = F.normalize(all_output).cpu()  # .cpu()
    mat_nosoft = torch.matmul(normalized_nosoft, normalized_nosoft.t()) / 0.05
    mask = torch.eye(mat.size(0), mat.size(0)).bool()  # .cuda()
    mat_nosoft.masked_fill_(mask, -1 / 0.05)
    ent_nosoft = entropy(mat_nosoft)
    return accuracy, ent_soft, ent_nosoft, ent_class


def train(config):
    ## set pre-process
    print("start MCC training!")
    if config["dataset"] == "visda":
        import preprop_visda as prep
    else:
        import pre_process as prep
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    tensor_writer = SummaryWriter(config["tensorboard_path"])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    dsets["source"], dsets["source_test"] = split_train_test(data_config["source"]["list_path"], prep_dict["source"],
                                                             prep_dict["test"],
                                                             return_id=False, perclass=3, random_seed=config["random_seed"])

    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=2, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=2, drop_last=True)
    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=test_bs, \
                                             shuffle=False, num_workers=2)
    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=2) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=2)

    class_num = config["network"]["params"]["class_num"]

    ## set nc_ps network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
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
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1 and i > config["start_record"]-1:
            base_network.train(False)
            s_risk, d_risk = test_dev(i, dset_loaders, config["result_path"], base_network)

            acc, nd, ent, nd_nosoft = image_classification_test(dset_loaders, \
                                                                base_network, test_10crop=prep_config["test_10crop"])

            output = "Iteration {} : current selected accs".format(i)

            for k, v in metric_names.items():
                value_rep = eval(v)
                if v is not "nd":
                    ## For s_risk, d_risk, ent, smaller is better.
                    if value_rep < metrics[k]:
                        metrics[k] = value_rep
                        acc_dict[k] = acc
                        iter_dict[k] = i
                elif v is "nd":
                    if value_rep > metrics[k]:
                        metrics[k] = value_rep
                        acc_dict[k] = acc
                        iter_dict[k] = i
                output += " {} : {} ".format(k, acc_dict[k])

            temp_model = nn.Sequential(base_network)
            if acc > acc_best:
                acc_best = acc
                iter_best = i
                best_model = temp_model
            output += "Acc best: {}".format(acc_best)
            print(output)

            log_str = "iter: {:05d}, precision: {:.5f} sim ent: {:.5f} " \
                      "ent_class: {:.5f} ".format(i, acc, nd.item(),
                                                  ent.item())

            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            if config["neptune"]:
                neptune.log_metric('sim entropy', nd.item())
                neptune.log_metric('sim entropy no soft', nd_nosoft.item())
                neptune.log_metric('entropy', ent.item())
                neptune.log_metric('accuracy', acc)
                neptune.log_metric('source test error', s_risk)
                neptune.log_metric('weighted risk', d_risk)
            print(log_str)

        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                                                             "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        outputs_target_temp = outputs_target / config['temperature']
        target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
        target_entropy_weight = loss.Entropy(target_softmax_out_temp).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
            target_softmax_out_temp)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = classifier_loss + mcc_loss
        total_loss.backward()
        optimizer.step()

        tensor_writer.add_scalar('total_loss', total_loss, i)
        tensor_writer.add_scalar('classifier_loss', classifier_loss, i)
        tensor_writer.add_scalar('cov_matrix_penalty', mcc_loss, i)

    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return acc_dict, metrics, iter_dict, acc_best, iter_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='5', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office',
                        choices=['office', 'image-clef', 'visda', 'office-home', 'DomainNet'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/dslr_list.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/amazon_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='office_temp2.5',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--random_seed', type=int, default=0, help="seed for selecting images")
    parser.add_argument('--neptune', type=bool, default=False, help="whether use neptune")
    parser.add_argument('--path_neptune', type=str, default="keisaito/sandbox", help="experiment path in neptune. change if you have it")
    parser.add_argument('--hps', type=float,
                        default=[2.5, 3.0, 2.0, 1.5, 3.5],
                        nargs="*",
                        help='trade off parameter')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # train config
    config = {}
    config["gpu"] = args.gpu_id
    config["random_seed"] = args.random_seed
    config["num_iterations"] = 10004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    task_name = args.output_dir + '/' + osp.basename(args.s_dset_path) + '2' + \
                osp.basename(args.t_dset_path)
    config["neptune"] = args.neptune
    config["output_path"] = "snapshot/" + task_name
    config["tensorboard_path"] = "vis/" + task_name
    source = args.s_dset_path.split("/")[-1]
    target = args.t_dset_path.split("/")[-1]
    output_filename = args.dset + '_%s_to_%s' % (source, target)
    config["result_path"] = os.path.join(config["output_path"], output_filename)

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop": False, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    #config["loss"] = {"trade_off": args.trade_off}
    config["loss"] = {"trade_off": 1.}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 28}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": 28}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": 4}}

    if config["dataset"] == "office":
        config["optimizer"]["lr_param"]["lr"] = 0.0003
        config["network"]["params"]["class_num"] = 31
        config["start_record"] = 1000
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.0003
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
        config["start_record"] = 2000
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65
        config["start_record"] = 1000
    elif config["dataset"] == "DomainNet":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 126
        config["start_record"] = 2000
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    setting = args.s_dset_path + "_to_" + args.t_dset_path
    current_name = os.path.basename(__file__)
    config['method'] = "MCC"
    PARAMS = {'weight': config['loss']["trade_off"],
              'machine': socket.gethostname(),
              'file': current_name,
              'net': args.net,
              'setting': setting,
              'dataset': config["dataset"],
              'method': config['method']}


    metric_names = {"neighborhood_density": "nd", "entropy": "ent",
                    "source_risk": "s_risk", "dev_risk": "d_risk"}
    all_metrics = {k: {} for k in metric_names.keys()}
    all_iters = {k: {} for k in metric_names.keys()}
    all_acc_dict = {k: {} for k in metric_names.keys()}
    acc_best_all = 0
    iter_best_all = 0

    for temper in args.hps:
        config["temperature"] = temper
        setting = args.s_dset_path + "_to_" + args.t_dset_path
        print("setting %s" % setting)
        if args.neptune:
            import socket
            import neptune
            neptune.init(args.path_neptune)
            current_name = os.path.basename(__file__)
            PARAMS = {'weight': config['loss']["trade_off"],
                      'lr': args.lr,
                      'machine': socket.gethostname(),
                      'file': current_name,
                      'net': args.net,
                      'setting': setting,
                      'dataset': config["dataset"],
                      'method': config['method']}

            neptune.create_experiment(name=current_name, params=PARAMS)
            neptune.append_tag("setting %s file %s" % (setting, current_name))

        # train(config)
        acc_choosen, metrics, iter_choosen, acc_best_gt, iter_best_gt = train(config)
        if acc_best_gt > acc_best_all:
            acc_best_all = acc_best_gt
            iter_best_all = iter_best_gt
        for name in metric_names.keys():
            all_acc_dict[name][temper], \
            all_metrics[name][temper], \
            all_iters[name][temper] = acc_choosen[name], metrics[name], iter_choosen[name]

    print(all_metrics)
    print(all_acc_dict)
    for met, acc, in zip(all_metrics.keys(), all_acc_dict.keys()):
        dict_met = all_metrics[met]
        if met == 'neighborhood_density':
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
        logging.basicConfig(filename=config["out_file"], format="%(message)s")
        logger.setLevel(logging.INFO)
        print(output)
        logger.info(output)

    #train(config)