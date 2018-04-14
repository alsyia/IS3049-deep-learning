import os
import itertools

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import plot_model

from CustomCallbacks import TensorBoardImage, EncoderCheckpoint, HuffmanCallback
from CustomLoss import loss, code
from Generator import DataGenerator
from Model import build_model
from ModelConfig import INPUT_SHAPE, DATASET_PATH, TRAIN_DIR, VALIDATION_DIR, TEST_DIR,load_model


# sess = K.get_session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-Wenceslas:6004")
# K.set_session(sess)

# Test with CIFAR10 dataset for now, has images of size (32, 32, 3)

train_list = os.listdir(DATASET_PATH + "/" + TRAIN_DIR)
val_list = os.listdir(DATASET_PATH + "/" + VALIDATION_DIR)
test_list = os.listdir(DATASET_PATH + "/" + TEST_DIR)

train_ratio = 0.7
val_ratio = 0.2


train_generator = DataGenerator(DATASET_PATH + "/" + TRAIN_DIR, train_list, 32, INPUT_SHAPE)
test_generator = DataGenerator(DATASET_PATH + "/" + VALIDATION_DIR, val_list, 32, INPUT_SHAPE)

# Compile model with adam optimizer
optimizer = {
    1:Adam(lr=1e-4, clipnorm=1),
    2:Adam(lr=1e-3, clipnorm=1)
}

earlystopping = {
    1:EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='auto'),
    2:EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=0, mode='auto')
}

experiment = [{"optimizer":optimizer[i],"earlystop":earlystopping[j]} for (i,j) in [x for x in itertools.product(optimizer,earlystopping)]]

for idx,exp in enumerate(experiment):
    print("starting experiment {} with {}".format(idx,exp))
    autoencoder = build_model()

    # Plot model graph
    plot_model(autoencoder, to_file='autoencoder.png')


    if load_model:
        weight_path = "weights.hdf5"
        print("loading weights from {}".format(weight_path))
        autoencoder.load_weights(weight_path)



    autoencoder.compile(optimizer=exp["optimizer"], loss={'clipping_layer_1':loss,'model_1':code})

    # Get last log
    log_index = None
    run_list = os.listdir("./logs")
    if len(run_list) == 0:
        log_index = 0
    else:
        indexes = [run[-1] for run in run_list]
        log_index = str(int(max(indexes)) + 1)

    tensorboard = TensorBoard(log_dir='./logs/run' + str(log_index), histogram_freq=0, batch_size=32)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint("weights.hdf5", save_best_only=True)
    encodercheckpoint = EncoderCheckpoint("encoder.hdf5", save_best_only=True)
    tensorboard_image = TensorBoardImage("Reconstruction", test_list=test_list, logs_path='./logs/run' + str(log_index))
    huffmancallback = HuffmanCallback(train_generator)


    # Train model !
    autoencoder.fit_generator(train_generator,
                            epochs=100,
                            validation_data=test_generator,
                            callbacks=[tensorboard, exp["earlystop"], checkpoint, tensorboard_image,encodercheckpoint,huffmancallback])