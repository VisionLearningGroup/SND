import os
import torch
import copy
import numpy as np
from .mydataset import ImageFolder, make_dataset_nolist
from collections import Counter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler



def split_train_test(list, train_trans, test_trans, perclass=3,
                     random_seed=0):
    images, labels = make_dataset_nolist(list)
    label_types = np.unique(labels)
    # images_select = []
    print("random seed is %s" % random_seed)
    np.random.seed(random_seed)
    for i, lb in enumerate(label_types):
        ind_lb = np.where(labels == lb)[0]
        random_perm = np.random.permutation(len(ind_lb))
        ind_select = ind_lb[random_perm[:perclass]]
        ind_noselect = ind_lb[random_perm[perclass:]]
        #print(ind_select)
        if i == 0:
            images_test = images[ind_select]
            labels_test = labels[ind_select]
            images_train = images[ind_noselect]
            labels_train = labels[ind_noselect]
        else:
            images_test = np.r_[images_test, images[ind_select]]
            images_train = np.r_[images_train, images[ind_noselect]]
            labels_test = np.r_[labels_test, labels[ind_select]]
            labels_train = np.r_[labels_train, labels[ind_noselect]]
    train_folder = ImageFolder(image_list=None, imgs=images_train,
                               labels=labels_train, transform=train_trans)
    test_folder = ImageFolder(image_list=None, imgs=images_test,
                              labels=labels_test, transform=test_trans)
    return train_folder, test_folder


def get_dataloaders(configs):
    source_path = configs['source_path']
    target_path = configs['target_path']
    random_seed = configs['random_seed']
    batch_size = configs['conf'].data.dataloader.batch_size
    balanced = configs['conf'].data.dataloader.class_balance
    data_transforms = {
        source_path: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_path: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=data_transforms[target_path],
                                      return_paths=False, return_id=True)

    source_folder, source_test_folder = \
        split_train_test(os.path.join(source_path),
                         data_transforms[source_path],
                         data_transforms["eval"],
                         random_seed=random_seed)
    eval_folder_test = ImageFolder(os.path.join(target_path),
                                   transform=data_transforms["eval"],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=2)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    source_test_loader = torch.utils.data.DataLoader(
        source_test_folder,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

    return source_loader, target_loader, source_test_loader,  test_loader


