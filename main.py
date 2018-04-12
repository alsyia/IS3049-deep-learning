import os
import pickle
import PIL.Image
import numpy as np
from keras.applications import VGG19
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam

from CustomCallbacks import TensorBoardImage, EncoderCheckpoint, HuffmanCallback
from CustomLoss import loss, code, perceptual_2, perceptual_5,texture
from Generator import DataGenerator
from Model import build_model
from ModelConfig import img_input_shape, dataset_path, train_dir, validation_dir, test_dir
from utils import generate_experiment
from predict import predict_from_ae

# sess = K.get_session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-Wenceslas:6004")
# K.set_session(sess)

train_list = os.listdir(dataset_path+"/"+train_dir)
val_list = os.listdir(dataset_path+"/"+validation_dir)
test_list = os.listdir(dataset_path+"/"+test_dir)

seed = np.random.seed(seed = 8)

img = PIL.Image.open(dataset_path + "/" + validation_dir + "/" + val_list[0])
img_img = img.resize(img_input_shape[0:2], PIL.Image.ANTIALIAS)
img = np.asarray(img_img) / 255
img = img.reshape(1, *img_input_shape)

exp_path = generate_experiment()


# VGG for the perceptual loss
base_model = VGG19(weights="imagenet", include_top=False,
                   input_shape=img_input_shape)

perceptual_model = Model(inputs=base_model.input,
                         outputs=[base_model.get_layer("block2_pool").output,
                                  base_model.get_layer("block5_pool").output],
                         name="VGG")
# We don't want to train VGG
perceptual_model.trainable = False
for layer in perceptual_model.layers:
    layer.trainable = False

# Make a prediction to force model instantiation, otherwise we have a really weird race condition issue
perceptual_model.predict(img)
print("Predicted")

texture_model = Model(inputs=base_model.input,
                         outputs=[base_model.get_layer("block2_pool").output],
                         name="VGG")
# We don't want to train VGG
texture_model.trainable = False
for layer in texture_model.layers:
    layer.trainable = False

# Make a prediction to force model instantiation, otherwise we have a really weird race condition issue
texture_model.predict(img)
print("Predicted")


autoencoder, _ = build_model(perceptual_model)

# create data generator
train_generator = DataGenerator(
    dataset_path + "/" + train_dir, train_list, perceptual_model, texture_model, 2, img_input_shape)
test_generator = DataGenerator(
    dataset_path + "/" + validation_dir, val_list, perceptual_model, texture_model, len(val_list), img_input_shape)

# Plot model graph
# plot_model(autoencoder, to_file='autoencoder.png')

load_model = False
if load_model:
    weight_path = "weights.hdf5"
    print("loading weights from {}".format(weight_path))
    autoencoder.load_weights(weight_path)


# Compile model with adam optimizer
optimizer = Adam(lr=1e-4, clipnorm=1)

autoencoder.compile(optimizer=optimizer, loss={"clipping_layer_1": loss,
                                               "rounding_layer_1": code,
                                               "VGG_block_2": perceptual_2,
                                               "VGG_block_5": perceptual_5,
                                               "de_patching_layer_1": texture})

# Get last log
log_index = None
run_list = os.listdir("./logs")
if len(run_list) == 0:
    log_index = 0
else:
    indexes = [run[-1] for run in run_list]
    log_index = str(int(max(indexes)) + 1)

tensorboard = TensorBoard(log_dir='./logs/run' +
                          str(log_index), histogram_freq=0, batch_size=32)
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=1e-5, patience=20, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(exp_path + "/weights.hdf5", save_best_only=True)
encodercheckpoint = EncoderCheckpoint(exp_path + "/encoder.hdf5", save_best_only=True)
tensorboard_image = TensorBoardImage(
    "Reconstruction", test_list=test_list, logs_path='./logs/run' + str(log_index), save_img=True, exp_path=exp_path)



history = autoencoder.fit_generator(train_generator,
                                    epochs=100,
                                    validation_data=test_generator,
                                    callbacks=[tensorboard_image,
                                               tensorboard,
                                               early_stopping,
                                               checkpoint,
                                               encodercheckpoint])

# dumping history into pickle for further use
with open(exp_path + '/history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

predict_from_ae(dataset_path + "/" + validation_dir, autoencoder)
