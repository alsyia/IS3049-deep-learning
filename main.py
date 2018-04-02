import os
import numpy as np
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.datasets import cifar10
from Model import build_model
from ModelConfig import e_input_shape,img_input_shape
from CustomLoss import loss
from utils import mirror_padding
from Generator import DataGenerator
from CustomCallback import IncrementalMask

import keras.backend as K
from tensorflow.python import debug as tf_debug
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)

img_list = os.listdir("data")
train_ratio = 0.7
val_ratio = 0.2

seed = np.random.seed(8)
indices = np.arange(len(img_list))
indices = np.random.permutation(indices)

train_index = int(train_ratio*len(img_list))
val_index = int(train_ratio*len(img_list))+int(val_ratio*len(img_list))



img_train = [img_list[i] for i in indices[:train_index]]
img_val = [img_list[i] for i in indices[train_index:val_index]]
img_test = [img_list[i] for i in indices[val_index:]]

train_generator = DataGenerator("data",img_train,7,img_input_shape)
test_generator = DataGenerator("data",img_test,len(img_test),img_input_shape)


autoencoder = build_model()

print("encoding image into vector with shape {}".format(autoencoder.layers[18].output_shape))

# Plot model graph
plot_model(autoencoder, to_file='autoencoder.png')

# Compile model with adadelta optimizer
# TODO: Code loss !
autoencoder.compile(optimizer='adadelta', loss=loss)


# Train model !
while train_generator.mask_idx < (img_input_shape[0]//8-1)**2:
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto')
    autoencoder.fit_generator(train_generator,
                    epochs=100,
                    validation_data=(test_generator[0]),
                    callbacks = [tensorboard,early_stopping])
    train_generator.mask_idx += 1
    test_generator.mask_idx += 1
