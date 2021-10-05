#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
def return_path(dataset,source,target):
    if dataset == 'office':
        p_path = os.path.join('/research/masaito/office/amazon/images')
        class_list = os.listdir(p_path)
        image_set_file_s = os.path.join('/research/masaito/office', 'all_images_' + source + '.txt')
        image_set_file_t = os.path.join('/research/masaito/office',
                                        'split_iccv/labeled_target_images_' + target + '_3.txt')
        image_set_file_t_val = os.path.join('/research/masaito/office',
                                            'split_iccv/validation_target_images_' + target + '_3.txt')
        image_set_file_test = os.path.join('/research/masaito/office',
                                           'split_iccv/unsupervised_target_images_' + target + '_3.txt')
    elif dataset == 'office_home':
        top = '/research/masaito/OfficeHomeDataset_10072016/split_iccv'
        p_path = os.path.join('/research/masaito/OfficeHomeDataset_10072016/Art')
        class_list = os.listdir(p_path)
        image_set_file_s = os.path.join(top, 'labeled_source_images_' + source + '.txt')
        image_set_file_t = os.path.join(top, 'labeled_target_images_' + target + '_%d.txt' % (num))
        image_set_file_t_val = os.path.join(top, 'validation_target_images_' + target + '_%d.txt' % (num))
        image_set_file_test = os.path.join(top, 'unlabeled_target_images_' + target + '_%d.txt' % (num))
    elif dataset == 'multi':
        base_path = '/research/masaito/multisource_data/few_shot_DA_data/split_iccv'
        p_path = os.path.join('/research/masaito/multisource_data/few_shot_DA_data/real')
        class_list = os.listdir(p_path)
        image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + source + '.txt')
        image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + target + '_3.txt')
        image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + target + '_3.txt')
        image_set_file_test = os.path.join(base_path, 'unlabeled_target_images_' + target + '_3.txt')
    return image_set_file_s,image_set_file_test, image_set_file_t,image_set_file_t_val
def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_Values(Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, imgs, labels, transform=None, loader=default_loader, transform_unknown=None):
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.loader = loader
        #import pdb
        #pdb.set_trace()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)



def make_dataset_nolist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def split_train_test(list, train_trans, test_trans, return_id=False, perclass=3, random_seed=0):
    images, labels = make_dataset_nolist(list)
    label_types = np.unique(labels)
    #images_select = []
    print("random seed %s"%(random_seed))
    np.random.seed(random_seed)
    for i, lb in enumerate(label_types):
        ind_lb = np.where(labels==lb)[0]
        random_perm = np.random.permutation(len(ind_lb))
        ind_select = ind_lb[random_perm[:perclass]]
        ind_noselect = ind_lb[random_perm[perclass:]]
        if i == 0:
            images_test = images[ind_select]
            labels_test = labels[ind_select]
            images_train = images[ind_noselect]
            labels_train = labels[ind_noselect]
        else:
            images_test = np.r_[images_test,  images[ind_select]]
            images_train = np.r_[images_train,  images[ind_noselect]]
            labels_test = np.r_[labels_test, labels[ind_select]]
            labels_train = np.r_[labels_train,  labels[ind_noselect]]
    #import pdb
    #pdb.set_trace()
    train_folder = ImageFolder_Values(images_train, labels_train, transform=train_trans)
    test_folder = ImageFolder_Values(images_test, labels_test, transform=test_trans)
    return train_folder, test_folder
