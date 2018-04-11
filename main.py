import os

import numpy as np
import PIL.Image
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.losses import mse
from keras.applications import VGG19
from keras.utils import plot_model as keras_utils_plot_model

from CustomCallbacks import TensorBoardImage, EncoderCheckpoint, HuffmanCallback, schedule
from CustomLoss import loss, code, perceptual_2, perceptual_5
from Generator import DataGenerator
from Model import build_model
from ModelConfig import img_input_shape, dataset_path, train_dir, validation_dir, test_dir
from utils import generate_experiment
from predict import predict_from_ae

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

# Build the model (see Model.py)
autoencoder, _ = build_model(perceptual_model)

# Create generator for both train data
train_generator = DataGenerator(
    dataset_path + "/" + train_dir, train_list, perceptual_model, 32, img_input_shape)
test_generator = DataGenerator(
    dataset_path + "/" + validation_dir, val_list, perceptual_model, len(val_list), img_input_shape)


plot_model = False
if plot_model:
    # Plot model graph
    keras_utils_plot_model(autoencoder, to_file='autoencoder.png')

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
                                               "VGG_block_5": perceptual_5})

# Get last log
log_index = None
run_list = os.listdir("./logs")
if len(run_list) == 0:
    log_index = 0
else:
    indexes = [run[-1] for run in run_list]
    log_index = str(int(max(indexes)) + 1)

# Create callbacks here.
tensorboard = TensorBoard(
    log_dir='./logs/run' + str(log_index),
    histogram_freq=0,
    batch_size=32)
tensorboard_image = TensorBoardImage(
    "Reconstruction",
    test_list=test_list,
    logs_path='./logs/run' + str(log_index),
    save_img=True,
    exp_path=exp_path)
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-5,
    patience=20,
    verbose=1,
    mode='auto')
checkpoint = ModelCheckpoint("weights.hdf5", save_best_only=True)
lr_decay = LearningRateScheduler(schedule)

history = autoencoder.fit_generator(train_generator,
                                    epochs=100,
                                    validation_data=test_generator[0],
                                    callbacks=[tensorboard_image,
                                               tensorboard,
                                               early_stopping,
                                               checkpoint,
                                               lr_decay])

# dumping history into pickle for further use
with open(exp_path + '/history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

predict_from_ae(dataset_path + "/" + validation_dir, autoencoder)
