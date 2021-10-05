import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import logging
from evaluate_cityscapes import eval_iter, eval_source
from compute_iou import compute_mIoU
from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet, GTA5DataSet_Test
from dataset.cityscapes_dataset import cityscapesDataSet
import socket
import neptune

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 1
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 30000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000 # 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main(lambda_adv_target1, lambda_adv_target2, output_log):
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    #cudnn.enabled = True
    args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]#args.gpu
    ## set up logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=output_log, format="%(message)s")
    logger.setLevel(logging.INFO)
    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)
    neptune.init('keisaito/sandbox')
    current_name = os.path.basename(__file__)
    PARAMS = {'method': 'segmentation',
              'machine': socket.gethostname(),
              'file': current_name,
              'net': args.model,
              'lambda_adv_target1': lambda_adv_target1,
              'lambda_adv_target2': lambda_adv_target2}
    print(PARAMS)
    filename = "Cityscape"
    save_p = './result/cityscapes_lamb1_%s_lamb2_%s'%(lambda_adv_target1, lambda_adv_target2)

    neptune.create_experiment(name=filename, params=PARAMS)

    model.train()
    model.cuda()

    #cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    model_D1.train()
    model_D1.cuda()

    model_D2.train()
    model_D2.cuda()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_test = data.DataLoader(
        GTA5DataSet_Test(args.data_dir, args.data_list, max_iters=None,
                    crop_size=input_size_target,
                    scale=False, mirror=False, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = iter(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)

    targetloader_eval = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                          crop_size=input_size_target,
                                                          scale=False, mirror=False, mean=IMG_MEAN,
                                                          set=args.set),
                                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True)

    targetloader_iter = iter(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1

    miou_source_best = 0
    siment_best = 0
    entropy_best = -1e5
    miou_best = 0
    miou_siment_selected = 0
    miou_entropy_selected = 0
    miou_source_selected = 0


    for i_iter in range(args.num_steps):
        ## early stopping for efficient experiments.
        if i_iter == 30000:
            break

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            batch = trainloader_iter.next()
            images, labels, _, _ = batch
            images = Variable(images).cuda()

            pred1, pred2 = model(images)
            pred1 = interp(pred1)
            pred2 = interp(pred2)

            loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.item() / args.iter_size
            loss_seg_value2 += loss_seg2.item() / args.iter_size

            # train with target

            batch = targetloader_iter.next()
            images, _, _ = batch
            images = Variable(images).cuda()

            pred_target1, pred_target2 = model(images)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(F.softmax(pred_target2))

            loss_adv_target1 = bce_loss(D_out1,
                                        Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(
                                        ))

            loss_adv_target2 = bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
                                        ))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            D_out1 = model_D1(F.softmax(pred1))
            D_out2 = model_D2(F.softmax(pred2))

            loss_D1 = bce_loss(D_out1,
                               Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(F.softmax(pred_target2))

            loss_D1 = bce_loss(D_out1,
                               Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()  # .data.cpu().numpy()[0]
            loss_D_value2 += loss_D2.item()  # .data.cpu().numpy()[0]

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        # import pdb
        # pdb.set_trace()
        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1,
                loss_adv_target_value2, loss_D_value1, loss_D_value2))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')

            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(),
                       osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(),
                       osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break
        print(i_iter, args.save_pred_every)
        if i_iter % args.save_pred_every == 0 and i_iter != 0 and i_iter >= args.save_pred_every:
            print('taking snapshot ...')
            miou_source = eval_source(model, trainloader_test, 'dataset/cityscapes_list')
            eval_iter(model, save_p=save_p)

            _, miou = compute_mIoU('./data/Cityscapes/gtFine/val', save_p, 'dataset/cityscapes_list')
            _, siment, ent = eval_siment(i_iter, model, targetloader_eval, interp)
            model.train()

            if miou_source > miou_source_best:
                miou_source_best = miou_source
                miou_source_selected = miou

            if siment > siment_best:
                siment_best = siment
                miou_siment_selected = miou

            if -ent > entropy_best:
                entropy_best = -ent
                miou_entropy_selected = miou
            if miou > miou_best:
                miou_best = miou
            #if args.neptune:
            neptune.log_metric('sim entropy', siment)
            neptune.log_metric('entropy', ent)
            neptune.log_metric('miou', miou)
            neptune.log_metric('miou source', miou_source)
            print('miou %s sim ent %s entropy %s' %(miou, siment, ent))
            output = 'miou %s sim ent %s entropy %s \n' %(miou_source, siment, ent)
            output += 'miou source best %s miou entropy best %s miou siment best %s \n' %(miou_source_selected,
                                                                                          miou_entropy_selected,
                                                                                          miou_siment_selected)
            logger.info(output)

            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))
    return {"source miou best": miou_source_best,
            "siment best": siment_best,
            "entropy best": entropy_best,
            "miou source select": miou_source_selected,
            "miou ent select": miou_entropy_selected,
            "miou siment select": miou_siment_selected,
            "miou best": miou_best}


def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en


def eval_siment(step, model, targetloader, interp):
    # interp = nn.Upsample(size=(180, 320), mode='bilinear')
    sum_pred1 = 0
    sum_pred2 = 0
    sum_ent = 0
    model.eval()
    for i, batch in enumerate(targetloader):
        images, _, _ = batch
        images = Variable(images).cuda()
        pred1, pred2 = model(images)
        pred1 = interp(pred1)
        pred2 = interp(pred2)
        pred1 = F.softmax(pred1)
        sum_ent += entropy(pred2).item()
        pred2 = F.softmax(pred2)
        pred1 = pred1.permute(0, 2, 3, 1)
        pred2 = pred2.permute(0, 2, 3, 1)
        pred1 = pred1.view(-1, pred1.size(3))
        pred2 = pred2.view(-1, pred2.size(3))
        pred1_rand = torch.randperm(pred1.size(0))
        pred2_rand = torch.randperm(pred2.size(0))
        pred1 = F.normalize(pred1[pred1_rand[:100]])
        pred2 = F.normalize(pred2[pred2_rand[:100]])
        pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
        pred2_en = entropy(torch.matmul(pred2, pred2.t()) * 20)
        sum_pred1 += pred1_en.item()
        sum_pred2 += pred2_en.item()
    print('step % sim pred1 %s sim pred2 %s ent %s' % (step, sum_pred1 / i, sum_pred2 / i, sum_ent / i))
    return sum_pred1 / i, sum_pred2 / i, sum_ent / i


if __name__ == '__main__':
    #adv_target1 = [0.0002, 0.0001, 0.001, 0.0005, 0.0003]
    adv_target1 = [0.01, 0.1]
    adv_target2 = [0.001]
    output_log = osp.join(args.snapshot_dir, "result.txt")
    all_results = []
    for adv_1 in adv_target1:
        for adv_2 in adv_target2:
            dict_result = main(adv_1, adv_2, output_log)
            all_results.append(dict_result)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=output_log, format="%(message)s")

    logger.setLevel(logging.INFO)
    logging.info(all_results)
    source_iou_best = 0
    siment_best = 0
    entropy_best = 0

    miou_best = 0
    miou_siment_selected = 0
    miou_entropy_selected = 0
    miou_source_selected = 0
    for result in all_results:
        if result['source miou best'] > source_iou_best:
            source_iou_best = result['source miou best']
            miou_source_selected = result['miou source select']
        if result['entropy best'] > entropy_best:
            entropy_best = result['entropy best']
            miou_entropy_selected = result['miou ent select']
        if result['siment best'] > siment_best:
            siment_best = result['siment best']
            miou_siment_selected = result['miou siment select']
    output = "source best %s entropy best %s siment best %s \n" %(source_iou_best, entropy_best, siment_best)
    output += 'miou source best %s miou entropy best %s miou siment best %s \n' % (miou_source_selected,
                                                                                  miou_entropy_selected,
                                                                                   miou_siment_selected)

    print(all_results)
    print(output)
    logging.info(output)








