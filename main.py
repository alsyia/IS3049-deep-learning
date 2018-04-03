import os
import numpy as np
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.datasets import cifar10
from Model import build_model
from ModelConfig import e_input_shape,img_input_shape
from CustomLoss import loss
from utils import mirror_padding
from Generator import DataGenerator
from keras.optimizers import Adam

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

print("encoding image into vector with shape {}".format(autoencoder.get_layer('e_conv_9').output_shape))

# Plot model graph
plot_model(autoencoder, to_file='autoencoder.png')

# Compile model with adadelta optimizer
# TODO: Code loss !

optimizer = Adam(lr =  1e-4, clipnorm = 1)
autoencoder.compile(optimizer=optimizer, loss=loss)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='auto')
checkpoint = ModelCheckpoint("weights.hdf5", save_best_only=True)

# Train model !
autoencoder.fit_generator(train_generator,
                epochs=50,
                validation_data=test_generator,
                callbacks = [tensorboard,early_stopping,checkpoint])