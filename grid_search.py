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
from CustomLoss import loss, code, perceptual_2, perceptual_5, entropy
from Generator import DataGenerator
from Model import build_model
from ModelConfig import img_input_shape, dataset_path, train_dir, validation_dir, test_dir
from utils import generate_experiment
from predict import predict_from_ae
from main import train

# On importe les données
train_list = os.listdir(dataset_path+"/"+train_dir)
val_list = os.listdir(dataset_path+"/"+validation_dir)
test_list = os.listdir(dataset_path+"/"+test_dir)

# On crée le dossier
exp_path = generate_experiment()

# Instanciate the VGG used for texture loss
base_model = VGG19(weights="imagenet", include_top=False,
                   input_shape=img_input_shape)

# Get the relevant layers
perceptual_model = Model(inputs=base_model.input,
                         outputs=[base_model.get_layer("block2_pool").output,
                                  base_model.get_layer("block5_pool").output],
                         name="VGG")

# Freeze this model
perceptual_model.trainable = False
for layer in perceptual_model.layers:
    layer.trainable = False

# Trick to force perceptual_model instanciation
img = PIL.Image.open(dataset_path + "/" + validation_dir + "/" + val_list[0])
img_img = img.resize(img_input_shape[0:2], PIL.Image.ANTIALIAS)
img = np.asarray(img_img) / 255
img = img.reshape(1, *img_input_shape)
perceptual_model.predict(img)
print("Predicted")

# Create generator for both train data
train_generator = DataGenerator(
    dataset_path + "/" + train_dir, train_list, perceptual_model, 32, img_input_shape)
val_generator = DataGenerator(
    dataset_path + "/" + validation_dir, val_list, perceptual_model, len(val_list), img_input_shape)

lr_decay = LearningRateScheduler(schedule)


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
    1: [1, 0, 0, 0],
    2: [1, 1, 0, 0],
    3: [1, 1, 1, 1]
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

    autoencoder, _ = build_model(perceptual_model)

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
                                                          "VGG_block_5": perceptual_5})

    earlystopping = exp["earlystopping"][0](**exp["earlystopping"][1])
    callbacks = [earlystopping, lr_decay]
    train(autoencoder, 1, sub_exp_path, train_generator,
          val_generator, test_list, 32, callbacks)
    
    del autoencoder


