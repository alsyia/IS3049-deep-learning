import numpy as np
import keras
from utils import mirror_padding
from ModelConfig import *
import PIL


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, folder,img_list, batch_size=32, dim=(32,32,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.img_list = img_list
        self.folder = folder
        self.len_data = len(img_list)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.mask_idx = 2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.len_data / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_temp = [self.img_list[k] for k in indexes]

        # Generate data
        X_padded, X_mean, X_std, X = self.__data_generation(img_temp)

        mask = np.zeros((self.batch_size,img_input_shape[0]//8,img_input_shape[1]//8,96))

        for i in range(self.mask_idx):
            mask[:,i%mask.shape[1],i//mask.shape[1]%mask.shape[2],:] = 1.0

        return [X_padded, X_mean ,X_std, mask], X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_data)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # compute the dimension after padding
        padded_shape = (self.dim[0]+2*mirror,self.dim[1]+2*mirror,self.dim[2])
        X_padded = np.empty((self.batch_size, *padded_shape ))
        X_mean = np.empty((self.batch_size,3))
        X_std = np.empty((self.batch_size,3))

        # Generate data
        for i in range(len(img_temp)):
            # Store sample
            img = PIL.Image.open(self.folder + "/" + img_temp[i])
            img = img.resize(img_input_shape[0:2], PIL.Image.ANTIALIAS)
            X[i,] = img
            X_mean[i,] = np.mean(X[i,],axis = (0,1))
            X_std[i,] = np.std(X[i,], axis = (0,1))
            X[i,] = (X[i,]-X_mean[i,])/X_std[i,]
            X_padded[i,] = mirror_padding(X[i,],mirror)

        return X_padded, X_mean, X_std, X