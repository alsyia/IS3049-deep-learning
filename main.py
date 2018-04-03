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

# sess = K.get_session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-Wenceslas:6004")
# K.set_session(sess)

# Test with CIFAR10 dataset for now, has images of size (32, 32, 3)

train_list = os.listdir("working_data/train")
val_list = os.listdir("working_data/val")
train_ratio = 0.7
val_ratio = 0.2


train_generator = DataGenerator("working_data/train",train_list,32,img_input_shape)
test_generator = DataGenerator("working_data/val",val_list,32,img_input_shape)

autoencoder = build_model()

print("encoding image into vector with shape {}".format(autoencoder.get_layer('conv_9').get_shape()))

# Plot model graph
plot_model(autoencoder, to_file='autoencoder.png')

# Compile model with adadelta optimizer
# TODO: Code loss !
autoencoder.compile(optimizer='adadelta', loss=loss)

# Train model !
autoencoder.fit_generator(train_generator,
                epochs=50,
                validation_data=test_generator)