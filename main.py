import os
import shutil
import pickle
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
from ModelConfig import INPUT_SHAPE, DATASET_PATH, TRAIN_DIR, VALIDATION_DIR, TEST_DIR, BATCH_SIZE
from utils import generate_experiment
from predict import predict_from_ae

def train(autoencoder,
          nb_epochs,
          exp_path,
          train_generator,
          val_generator,
          test_list,
          batch_size,
          extra_callbacks=[]):
    # autoencoder must have been compiled

    # Get last log
    log_index = None
    run_list = os.listdir("./logs")
    if len(run_list) == 0:
        log_index = 0
    else:
        indexes = [run[-1] for run in run_list]
        log_index = str(int(max(indexes)) + 1)

    # Tracking callbacks
    tensorboard = TensorBoard(
        log_dir='./logs/run' + str(log_index),
        histogram_freq=0,
        batch_size=batch_size)
    tensorboard_image = TensorBoardImage(
        "Reconstruction",
        test_list=test_list,
        logs_path='./logs/run' + str(log_index),
        save_img=True,
        exp_path=exp_path)

    checkpoint = ModelCheckpoint(exp_path + "/weights.hdf5", save_best_only=True)
    huffman = HuffmanCallback(val_generator[0][0])
    history = autoencoder.fit_generator(train_generator,
                                        epochs=nb_epochs,
                                        validation_data=val_generator[0],
                                        callbacks=[tensorboard_image,
                                                   tensorboard,
                                                   checkpoint,
                                                   huffman] + extra_callbacks)

    # dumping history into pickle for further use
    with open(exp_path + '/history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    shutil.copytree('./logs/run' + str(log_index), exp_path + '/run' + str(log_index))
    return autoencoder

if __name__ == '__main__':
    # On importe les données
    train_list = os.listdir(DATASET_PATH + "/" + TRAIN_DIR)
    val_list = os.listdir(DATASET_PATH + "/" + VALIDATION_DIR)
    test_list = os.listdir(DATASET_PATH + "/" + TEST_DIR)

    seed = np.random.seed(seed = 8)

    img = PIL.Image.open(DATASET_PATH + "/" + VALIDATION_DIR + "/" + val_list[0])
    img_img = img.resize(INPUT_SHAPE[0:2], PIL.Image.ANTIALIAS)
    img = np.asarray(img_img) / 255
    img = img.reshape(1, *INPUT_SHAPE)

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


    autoencoder, _ = build_model(perceptual_model, texture_model)

    # create data generator
    train_generator = DataGenerator(
        DATASET_PATH + "/" + TRAIN_DIR, train_list, perceptual_model, texture_model,"model", BATCH_SIZE, INPUT_SHAPE)
    val_generator = DataGenerator(
        DATASET_PATH + "/" + VALIDATION_DIR, val_list, perceptual_model, texture_model,"model", BATCH_SIZE, INPUT_SHAPE)


    #train_generator = DataGenerator(
    #    DATASET_PATH + "/" + TRAIN_DIR, train_list, perceptual_model, "celeba64_debug/texture_train","file", BATCH_SIZE, INPUT_SHAPE)
    #test_generator = DataGenerator(
    #    DATASET_PATH + "/" + VALIDATION_DIR, val_list, perceptual_model, "celeba64_debug/texture_val","file", BATCH_SIZE, INPUT_SHAPE)
    # test_generator = DataGenerator(
    #     dataset_path + "/" + validation_dir, val_list, perceptual_model, texture_model, len(val_list), img_input_shape)

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
                                                "rounding_layer_1": entropy,
                                               "VGG_block_2": perceptual_2,
                                               "VGG_block_5": perceptual_5,
                                               "de_patching_layer_1": texture},
                                               loss_weights=[1,1,1,1,1])

    # extra callbacks

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=20,
        verbose=1,
        mode='auto')

    autoencoder = train(autoencoder,
                        30,
                        exp_path,
                        train_generator,
                        val_generator,
                        test_list,
                        BATCH_SIZE,
                        [early_stopping])
    predict_from_ae(DATASET_PATH + "/" + VALIDATION_DIR, autoencoder)
