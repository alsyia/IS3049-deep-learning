import numpy as np
import keras
from utils import mirror_padding

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img, batch_size=32, dim=(32,32,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.img = img
        self.len_data = self.img.shape[0]
        self.shuffle = shuffle
        self.on_epoch_end()
        self.pad = 12

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.len_data / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_temp = [self.img[k,:,:,:] for k in indexes]

        # Generate data
        X_padded, X = self.__data_generation(img_temp)

        return X_padded, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_data)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        padded_shape = (self.dim[0]+2*self.pad,self.dim[1]+2*self.pad,self.dim[2])
        X_padded = np.empty((self.batch_size, *padded_shape ))

        # Generate data
        for i in range(len(img_temp)):
            # Store sample
            X[i,] = img_temp[i]
            X_padded[i,] = mirror_padding(X[i,],12)
        return X_padded, X