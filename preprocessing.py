import os

import numpy as np
import PIL
from keras.datasets import cifar10

from utils import mirror_padding


def split_indices(nb_indices,train_ratio,val_ratio, seed = 8):
    seed = np.random.seed(seed)

    indices = np.arange(nb_indices)
    indices = np.random.permutation(indices)

    train_index = int(train_ratio*len(indices))
    val_index = int(train_ratio*len(indices))+int(val_ratio*len(indices))

    return indices[:train_index],indices[train_index:val_index], indices[val_index:]


def store_data(src,dst,img_size,img_train,img_val,img_test):
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    if not os.path.exists(dst + '/train'):
        os.mkdir(dst + '/train')

    if not os.path.exists(dst + '/val'):
        os.mkdir(dst + '/val')

    if not os.path.exists(dst + '/test'):
        os.mkdir(dst + '/test')
    
    for img_idx in range(len(img_train)):
        img = PIL.Image.open(src + "/" + img_train[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save(dst + '/train/'+img_train[img_idx].split('.')[0] + ".png")

    for img_idx in range(len(img_val)):
        img = PIL.Image.open(src + "/" + img_val[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save(dst + '/val/'+img_train[img_idx].split('.')[0] + ".png")

    for img_idx in range(len(img_test)):
        img = PIL.Image.open(src + "/" + img_test[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save(dst + '/test/'+img_train[img_idx].split('.')[0] + ".png")

def load_cifar10(dst = "cifar10", nb_images = 2000):
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    if not os.path.exists(dst + "/train"):
        os.mkdir(dst + "/train")
    
    if not os.path.exists(dst + "/val"):
        os.mkdir(dst + "/val")
    
    if not os.path.exists(dst + "/test"):
        os.mkdir(dst + "/test")
    (x_train, _), (x_test, _) = cifar10.load_data()
    cifar = np.concatenate([x_train,x_test])
    cifar = cifar[:min(nb_images,cifar.shape[0])]

    train_index, val_index, test_index = split_indices(cifar.shape[0],0.7,0.2)
    for idx in train_index:
        img = PIL.Image.fromarray(np.uint8(cifar[idx,:,:,:]))
        img.save(dst + "/train/train_{}.png".format(idx))
    
    for idx in val_index:
        img = PIL.Image.fromarray(np.uint8(cifar[idx,:,:,:]))
        img.save(dst + "/val/val_{}.png".format(idx))
    for idx in test_index:
        img = PIL.Image.fromarray(np.uint8(cifar[idx,:,:,:]))
        img.save(dst + "/test/test_{}.png".format(idx))

def load_folder(src,dst,img_size,nb_images):
    img_list = os.listdir(src)
    img_list = img_list[:min(nb_images,len(img_list))]

    train_index, val_index, test_index = split_indices(len(img_list),0.7,0.2)

    train_list = [img_list[i] for i in train_index]
    val_list = [img_list[i] for i in val_index]
    test_list = [img_list[i] for i in test_index]

    store_data(src,dst,img_size,train_list,val_list,test_list)
    
load_cifar10("cifar10", 2000)
#load_folder("data", "test",(32,32))