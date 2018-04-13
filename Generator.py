import PIL
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from ModelConfig import *
from utils import extract_patches

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, folder, img_list, vgg_perceptual, vgg_texture, batch_size=32, dim=(32, 32, 3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.img_list = img_list
        self.vgg_perceptual = vgg_perceptual
        self.vgg_texture = vgg_texture
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
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        img_temp = [self.img_list[k] for k in indexes]

        # Generate data
        X, B, F2, F5, F2_patches = self.__data_generation(img_temp)
        # F2_patch = tf.extract_image_patches(F2, ksizes=(1, 8, 8, 1),
        #                                     strides=(1, 8, 8, 1),
        #                                     rates=(1, 1, 1, 1),
        #                                     padding="SAME")
        # with K.get_session():
        #     F2_patch = F2_patch.eval()

        return X, [B, X, F2, F5, F2_patches]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_data)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        B = np.empty((self.batch_size, *self.dim))
        # Generate data
        for i in range(len(img_temp)):
            # Store sample
            img = PIL.Image.open(self.folder + "/" + img_temp[i])
            img = img.resize(img_input_shape[0:2], PIL.Image.ANTIALIAS)
            img = np.asarray(img)
            X[i, ] = img / 255

            # B sert juste à avoir une coherence entre les sorties du réseau et les verites terrains
            # il est rempli de 0
            B[i, ] = 0

        # On génère maintenant les features pour la perceptual_loss
        self.vgg_perceptual._make_predict_function()
        self.vgg_texture._make_predict_function()
        F2, F5 = self.vgg_perceptual.predict(X)

        patch_size = texture_params["patch_size"]
        patches_per_img = (img_input_shape[0]//patch_size[0])*(img_input_shape[1]//patch_size[1])
        # On génère les patchs
        # print("[Generator] Shape of X: " + str(X.shape))
        # print("[Generator] Shape of F2: " + str(F2.shape))
        # Renvoie un tenseur de taille (batch_size*#patches, 8, 8, 3)
        patches = extract_patches(X, (*patch_size, 3))
        # print("[Generator] Patches array shape : " + str(patches.shape))
        # On padde : (batch_size*#patches, 64, 64, 3)
        pad_size_h = (img_input_shape[0] - patch_size[0])//2
        pad_size_v = (img_input_shape[1] - patch_size[1])//2
        padded_patches = np.pad(patches, [[0, 0], [pad_size_h, pad_size_h], [pad_size_v, pad_size_v], [0, 0]], "constant")
        # print("[Generator] Padded patches array shape : " + str(padded_patches.shape))
        # On envoie tout ça comme un gros batch dans le VGG
        # On récupère une liste de 2048 éléments de taille (16, 16, 128)
        textures = self.vgg_texture.predict(padded_patches)
        # print("[Generator] Textures array shape : " + str(textures[0].shape))
        # print("[Generator] Texture list length : " + str(len(textures)))
        # On fait des arrays de taille (64, 16, 16, 128), chaque array correspondant aux textures d'une
        # image
        textures_grouped = []
        for i in range(0, len(textures), patches_per_img):
            textures_grouped.append(np.stack(textures[i:i+patches_per_img], 0))
        # print("[Generator] Grouped textures array shape : " + str(textures_grouped[0].shape))
        # print("[Generator] Grouped textures list length : " + str(len(textures_grouped)))
        # Et on stacke pour avoir un tenseur (32, 64, 16, 16, 128) qui se lit comme suit :
        #   Pour chaque image
        #       Pour chaque patch
        #           Sortie du VGG de taille (16, 16, 128)
        textures_stacked = np.stack(textures_grouped, axis=0)
        # print("[Generator] Final texture shape : " + str(textures_stacked.shape))

        return X, B, F2, F5, textures_stacked
