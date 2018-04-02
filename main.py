import os
import numpy as np
#from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10
from keras.optimizers import Adam
from Model import build_model
from ModelConfig import e_input_shape,img_input_shape
from CustomLoss import loss
from Generator import DataGenerator
import PIL
import preprocessing
""" import keras.backend as K
from tensorflow.python import debug as tf_debug
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)
 """


img_train = os.listdir('working_data/train')
img_val = os.listdir('working_data/val')
img_test = os.listdir('working_data/test')

train_generator = DataGenerator('working_data/train',img_train,32,img_input_shape)
test_generator = DataGenerator('working_data/val',img_val,len(img_val),img_input_shape)


autoencoder = build_model()


print("encoding image into vector with shape {}".format(autoencoder.layers[18].output_shape))

# Plot model graph
#plot_model(autoencoder, to_file='autoencoder.png')

# Compile model with adadelta optimizer
# TODO: Code loss !

# optimizer
optimizer = Adam(lr = 1e-4)

# compilation
autoencoder.compile(optimizer=optimizer, loss=loss)
checkpoint = ModelCheckpoint("weights.hdf5", save_best_only=True)

iteration = 0
# Train model !
while train_generator.mask_idx < (img_input_shape[0]//8-1)**2:
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='auto')
    autoencoder.fit_generator(train_generator,
                    epochs=100,
                    validation_data=(test_generator[0]),
                    callbacks = [tensorboard,early_stopping, checkpoint])
    

    img_list = os.listdir("data")
    img = preprocessing.read_image('data/' + img_list[0])

    imgs = autoencoder.predict([img, np.ones((1,img_input_shape[0]//8,img_input_shape[1]//8,96))])

    img = imgs[0,:,:,:]
    img = PIL.Image.fromarray(np.uint8(img*255))
    img.save("iteration_{}.png".format(iteration))

    train_generator.mask_idx += 1
    test_generator.mask_idx += 1
    iteration += 1