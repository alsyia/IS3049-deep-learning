import PIL
import os
from utils import mirror_padding
from ModelConfig import *
import numpy as np

folder = "data"

def preprocess_data(src,nb_image,train_ratio,val_ratio, img_size):
    img_list = os.listdir(folder)
    seed = np.random.seed(8)
    indices = np.arange(min(len(img_list),nb_image))
    indices = np.random.permutation(indices)

    train_index = int(train_ratio*len(indices))
    val_index = int(train_ratio*len(indices))+int(val_ratio*len(indices))

    img_train = [img_list[i] for i in indices[:train_index]]
    img_val = [img_list[i] for i in indices[train_index:val_index]]
    img_test = [img_list[i] for i in indices[val_index:]]

    if not os.path.exists('working_data'):
        os.mkdir('working_data')
    
    if not os.path.exists('working_data/train'):
        os.mkdir('working_data/train')

    if not os.path.exists('working_data/val'):
        os.mkdir('working_data/val')

    if not os.path.exists('working_data/test'):
        os.mkdir('working_data/test')
    
    for img_idx in range(len(img_train)):
        img = PIL.Image.open(folder + "/" + img_train[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save('working_data/train/'+img_train[img_idx].split('.')[0] + ".png")

    for img_idx in range(len(img_val)):
        img = PIL.Image.open(folder + "/" + img_val[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save('working_data/val/'+img_train[img_idx].split('.')[0] + ".png")

    for img_idx in range(len(img_test)):
        img = PIL.Image.open(folder + "/" + img_test[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save('working_data/test/'+img_train[img_idx].split('.')[0] + ".png")

def read_image(path):
    img = PIL.Image.open(path)
    img = img.resize(img_input_shape[0:2], PIL.Image.ANTIALIAS)
    img = np.asarray(img)
    return np.reshape(img,(1,*img.shape))

def normalize(img):
    img_mean = np.mean(img,axis = (0,1))
    img_std = np.std(img, axis = (0,1))
    img = (img-img_mean)/img_std
    img = mirror_padding(np.asarray(img),12)
    img = np.reshape(img, (1,*img.shape))
    img_mean = np.reshape(img_mean,(1,3))
    img_std = np.reshape(img_std,(1,3))
    return img

def denormalize(img,mean,std):
    img = (img * std) + mean
    return img
    

folder=  "/media/romain/DATA/Shared_repository/srez/dataset"
preprocess_data(folder,2000,0.7,0.2,(32,32))