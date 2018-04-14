import os
import itertools
import shutil
import pickle
import json
import numpy as np
import PIL.Image
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.losses import mse
from keras.applications import VGG19
from keras.utils import plot_model as keras_utils_plot_model

from CustomCallbacks import TensorBoardImage, EncoderCheckpoint, HuffmanCallback, schedule
from CustomLoss import loss, code, perceptual_2, perceptual_5, entropy, texture
from Generator import DataGenerator
from Model import build_model
from ModelConfig import INPUT_SHAPE, DATASET_PATH, TRAIN_DIR, VALIDATION_DIR, TEST_DIR, BATCH_SIZE, EPOCH_NB
from utils import generate_experiment
from predict import predict_from_ae
from main import train

# On importe les données
train_list = os.listdir(DATASET_PATH+"/"+TRAIN_DIR)
val_list = os.listdir(DATASET_PATH+"/"+VALIDATION_DIR)
test_list = os.listdir(DATASET_PATH+"/"+TEST_DIR)

seed = np.random.seed(seed=8)

img = PIL.Image.open(DATASET_PATH + "/" + VALIDATION_DIR + "/" + val_list[0])
img_img = img.resize(INPUT_SHAPE[0:2], PIL.Image.ANTIALIAS)
img = np.asarray(img_img) / 255
img = img.reshape(1, *INPUT_SHAPE)

# On crée le dossier
exp_path = generate_experiment()

# VGG for the perceptual loss
base_model = VGG19(weights="imagenet", include_top=False,
                   input_shape=INPUT_SHAPE)

perceptual_model = Model(inputs=base_model.input,
                         outputs=[base_model.get_layer("block2_pool").output,
                                  base_model.get_layer("block5_pool").output],
                         name="VGG_perceptual")
# We don't want to train VGG
perceptual_model.trainable = False
for layer in perceptual_model.layers:
    layer.trainable = False

# Make a prediction to force model instantiation, otherwise we have a really weird race condition issue
perceptual_model.predict(img)
print("Predicted")

texture_model = Model(inputs=base_model.input,
                      outputs=[base_model.get_layer("block2_pool").output],
                      name="VGG_texture")
# We don't want to train VGG
texture_model.trainable = False
for layer in texture_model.layers:
    layer.trainable = False

# Make a prediction to force model instantiation, otherwise we have a really weird race condition issue
texture_model.predict(img)
print("Predicted")


# Create generator for both train data
train_generator = DataGenerator(
    DATASET_PATH + "/" + TRAIN_DIR, train_list, perceptual_model, texture_model, "model", BATCH_SIZE, INPUT_SHAPE)
val_generator = DataGenerator(
    DATASET_PATH + "/" + VALIDATION_DIR, val_list, perceptual_model, texture_model, "model", BATCH_SIZE, INPUT_SHAPE)


# Different optimizer choice
optimizer_params = {
    1: [Adam, {"lr": 1e-4, "clipnorm": 1}]
}

# Different earlystopping choice
earlystopping_params = {
    1: [EarlyStopping, {"monitor": 'val_loss', "min_delta": 1e-4, "patience": 20, "verbose": 0, "mode": 'auto'}]
}

# loss weights
loss_params = {
    1: [1, 0, 0, 0, 0],
    2: [1, 1, 0, 0, 0],
    3: [1, 1, 1, 1, 0],
    4: [1, 1, 1, 1, 1]
}
experiment = [{"optimizer": optimizer_params[i],
               "earlystopping":earlystopping_params[j],
               "loss_weights":loss_params[k]}
              for (i, j, k) in [x for x in itertools.product(optimizer_params,
                                                             earlystopping_params,
                                                             loss_params)]]

for idx, exp in enumerate(experiment):
    print("starting experiment {} with {}".format(idx, exp))

    # create sub_experiment file
    sub_exp_path = exp_path + '/' + str(idx)
    os.mkdir(sub_exp_path)

    autoencoder, _ = build_model(perceptual_model, texture_model)

    load_model = False
    if load_model:
        weight_path = "weights.hdf5"
        print("loading weights from {}".format(weight_path))
        autoencoder.load_weights(weight_path)

    optimizer = exp["optimizer"][0](**exp["optimizer"][1])
    loss_weights = exp["loss_weights"]

    autoencoder.compile(optimizer=optimizer, loss={"clipping_layer_1": loss,
                                                   "rounding_layer_1": entropy,
                                                   "VGG_block_2": perceptual_2,
                                                   "VGG_block_5": perceptual_5,
                                                   "de_patching_layer_1": texture},
                        loss_weights=loss_weights)

    earlystopping = exp["earlystopping"][0](**exp["earlystopping"][1])
    callbacks = [earlystopping]
    train(autoencoder, EPOCH_NB, sub_exp_path, train_generator,
          val_generator, test_list, BATCH_SIZE, callbacks)

    del autoencoder
