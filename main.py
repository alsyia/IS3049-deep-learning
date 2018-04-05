import os

import PIL.Image
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.losses import mse
from CustomCallbacks import TensorBoardImage, EncoderCheckpoint, HuffmanCallback
from CustomLoss import loss, code, perceptual_2, perceptual_5
from Generator import DataGenerator
from Model import build_model
from ModelConfig import img_input_shape, dataset_path, train_dir, validation_dir, test_dir

# sess = K.get_session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-Wenceslas:6004")
# K.set_session(sess)

# Test with CIFAR10 dataset for now, has images of size (32, 32, 3)

train_list = os.listdir(dataset_path+"/"+train_dir)
val_list = os.listdir(dataset_path+"/"+validation_dir)
test_list = os.listdir(dataset_path+"/"+test_dir)

train_ratio = 0.7
val_ratio = 0.2

autoencoder, vgg = build_model()

train_generator = DataGenerator(dataset_path + "/" + train_dir, train_list, vgg, 32, img_input_shape)
test_generator = DataGenerator(dataset_path + "/" + validation_dir, val_list, vgg, 32, img_input_shape)

# Plot model graph
plot_model(autoencoder, to_file='autoencoder.png')

load_model = False
if load_model:
    weight_path = "weights.hdf5"
    print("loading weights from {}".format(weight_path))
    autoencoder.load_weights(weight_path)

# TODO: Code loss !

# Compile model with adam optimizer
optimizer = Adam(lr=1e-4, clipnorm=1)
# WARNING: Order IS important here ! Please check outputs order in Model.py, should match

autoencoder.compile(optimizer=optimizer, loss={"clipping_layer_1": loss,
                                                "rounding_layer_1": code,
                                                "VGG_block_2": perceptual_2,
                                                "VGG_block_5": perceptual_5})

# Get last log
log_index = None
run_list = os.listdir("./logs")
if len(run_list) == 0:
    log_index = 0
else:
    indexes = [run[-1] for run in run_list]
    log_index = str(int(max(indexes)) + 1)

tensorboard = TensorBoard(log_dir='./logs/run' + str(log_index), histogram_freq=0, batch_size=32)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, verbose=1, mode='auto')
checkpoint = ModelCheckpoint("weights.hdf5", save_best_only=True)
encodercheckpoint = EncoderCheckpoint("encoder.hdf5", save_best_only=True)
tensorboard_image = TensorBoardImage("Reconstruction", test_list=test_list, logs_path='./logs/run' + str(log_index))

huffmancallback = HuffmanCallback(train_generator)


# Train model !
autoencoder.fit_generator(train_generator,
                          epochs=100,
                          validation_data=test_generator,
                          callbacks=[tensorboard_image, tensorboard, early_stopping, checkpoint,encodercheckpoint,huffmancallback],
                          verbose = 1)


img = PIL.Image.open(dataset_path +"/" + validation_dir + "/" +val_list[0])
img_img = img.resize(img_input_shape[0:2], PIL.Image.ANTIALIAS)
img = np.asarray(img_img) / 255
img = img.reshape(1, *img_input_shape)
reconstruction = autoencoder.predict(img)
reconstruction = reconstruction[1]*255
reconstruction = np.clip(reconstruction, 0, 255)
reconstruction = np.uint8(reconstruction)
reconstruction = reconstruction.reshape(*img_input_shape)
reconstruction_img = PIL.Image.fromarray(reconstruction)
img_img.save("input.png")
reconstruction_img.save("output.png")
