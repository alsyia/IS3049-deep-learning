import os

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import plot_model

from CustomCallbacks import TensorBoardImage
from CustomLoss import loss
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


train_generator = DataGenerator(dataset_path+"/"+train_dir,train_list,32,img_input_shape)
test_generator = DataGenerator(dataset_path+"/"+validation_dir,val_list,32,img_input_shape)

autoencoder = build_model()

print("encoding image into vector with shape {}".format(autoencoder.get_layer('e_conv_9').output_shape))

# Plot model graph
plot_model(autoencoder, to_file='autoencoder.png')

# Compile model with adadelta optimizer
# TODO: Code loss !

optimizer = Adam(lr =  1e-4, clipnorm = 1)
autoencoder.compile(optimizer=optimizer, loss=loss)

# Get last log
log_index = None
run_list = os.listdir("./logs")
if len(run_list) == 0:
    log_index = 0
else:
    indexes = [run[-1] for run in run_list]
    log_index = str(int(max(indexes)) + 1)

tensorboard = TensorBoard(log_dir='./logs/run'+str(log_index), histogram_freq=0, batch_size=32)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='auto')
checkpoint = ModelCheckpoint("weights.hdf5", save_best_only=True)
tensorboard_image = TensorBoardImage("Reconstruction", test_list=test_list, logs_path='./logs/run'+str(log_index))
img = test_generator[0][0]

# Train model !
autoencoder.fit_generator(train_generator,
                epochs=50,
                validation_data=test_generator,
                callbacks = [tensorboard, early_stopping, checkpoint, tensorboard_image])
