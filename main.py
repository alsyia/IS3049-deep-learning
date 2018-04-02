import os
import numpy as np
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from Model import build_model
from ModelConfig import e_input_shape,img_input_shape
from CustomLoss import loss
from utils import mirror_padding
from Generator import DataGenerator
from CustomCallback import IncrementalMask
# sess = K.get_session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-Wenceslas:6004")
# K.set_session(sess)

# Test with CIFAR10 dataset for now, has images of size (32, 32, 3)


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

train_generator = DataGenerator("data",img_train,32,img_input_shape)
test_generator = DataGenerator("data",img_test,len(img_test),img_input_shape)


autoencoder = build_model()

print("encoding image into vector with shape {}".format(autoencoder.layers[18].output_shape))

# Plot model graph
plot_model(autoencoder, to_file='autoencoder.png')

# Compile model with adadelta optimizer
# TODO: Code loss !
autoencoder.compile(optimizer='adadelta', loss=loss)

incremental = IncrementalMask(96*96, train_generator)
# Train model !
autoencoder.fit_generator(train_generator,
                steps_per_epoch = len(img_train)//32,
                epochs=50,
                validation_data=(test_generator[0]),
                callbacks = [incremental])