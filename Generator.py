import PIL
import keras
import numpy as np

from ModelConfig import *


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, folder, img_list, batch_size=32, dim=(32, 32, 3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.img_list = img_list
        self.folder = folder
        self.len_data = len(img_list)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.len_data / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        img_temp = [self.img_list[k] for k in indexes]

        # Generate data
        X, B = self.__data_generation(img_temp)

        return X, [X, B]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_data)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        B = np.empty((self.batch_size, self.dim[0]/8,self.dim[1]/8, self.dim[2]))
        # Generate data
        for i in range(len(img_temp)):
            # Store sample
            img = PIL.Image.open(self.folder + "/" + img_temp[i])
            img = img.resize(img_input_shape[0:2], PIL.Image.ANTIALIAS)
            img = np.asarray(img)
            X[i,] = img / 255

            # B sert juste à avoir une coherence entre les sorties du réseau et les verites terrains
            # il est rempli de 0
            B[i,] = 0
        return X, B
