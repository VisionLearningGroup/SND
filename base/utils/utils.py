import os
import torch
from torch import nn
import torch.optim as optim
from apex import amp, optimizers
from models.basenet import ResClassifier_MME, ResBase


def get_models(configs, norm=True):
    net = configs['conf'].model.base_model
    num_class = configs['num_class']
    temp = configs['conf'].model.temp
    configs['temp'] = temp
    dim = 2048
    G = ResBase(net)

    if "resnet18" in net:
        dim = 512
    if net == "resnet34":
        dim = 512

    configs['dim'] = dim
    print("selected network %s"%net)
    C = ResClassifier_MME(num_classes=num_class,
                          temp=temp, input_size=dim, norm=norm)

    device = torch.device("cuda")
    if configs['cuda']:
        G.cuda()
        C.cuda()
    G.to(device)
    C.to(device)
    D, M = None, None
    if configs['method'] == 'DANN':
        D = Discriminator(input_size=dim)
        D.to(device)

    models = {'G': G,
              'C': C,
              'D': D,
              'M': M}
    opt_g, opt_c, param_lr_g, param_lr_c = get_optimizers(configs, models)
    return models, opt_g, opt_c, param_lr_g, param_lr_c


def get_optimizers(configs, models):
    params = []
    for key, value in dict(models['G'].named_parameters()).items():
        params += [{'params': [value], 'lr': 0.1,
                    'weight_decay': 0.0005}]
    opt_g = optim.SGD(params, momentum=0.9,
                      weight_decay=0.0005, nesterov=True)
    list_c = list(models['C'].parameters())
    if configs['method'] == 'DANN':
        list_c += list(models['D'].parameters())
    opt_c = optim.SGD(list_c, lr=1.0,
                       momentum=0.9,
                       weight_decay=0.0005,
                       nesterov=True)
    [models['G'], models['C']], [opt_g, opt_c] = amp.initialize([models['G'],
                                                                 models['C']],
                                                                [opt_g, opt_c],
                                                                opt_level="O1")
    models['G'] = nn.DataParallel(models['G'])
    models['C'] = nn.DataParallel(models['C'])
    if configs['method'] == 'DANN':
        models['D'] = nn.DataParallel(models['D'])
    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    return opt_g, opt_c, param_lr_g, param_lr_c


def save_model(model_g, model_c, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c_state_dict': model_c.state_dict(),
    }
    torch.save(save_dic, save_path)


def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c


def log_set(kwargs):
    source_data = os.path.basename(kwargs["source_path"]).replace(".txt", "")
    target_data = os.path.basename(kwargs["target_path"]).replace(".txt", "")
    network = kwargs["conf"].model.base_model
    conf_file = kwargs["config_file"]
    script_name = kwargs["script_name"].replace(".py", "")
    multi = kwargs["hp"]
    method = kwargs["method"]
    target_data = os.path.splitext(os.path.basename(target_data))[0]
    logdir = f"{script_name}_{source_data}_to_{target_data}_{network}_{method}"
    logname = os.path.join("record",
                           os.path.basename(conf_file).replace(".yaml", ""),
                           logdir, f"hp_{multi}")
    if not os.path.exists(os.path.dirname(logname)):
        os.makedirs(os.path.dirname(logname))
    print("record in %s " % logname)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=logname, format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info("{}_2_{}".format(source_data, target_data))
    return logname, logger


