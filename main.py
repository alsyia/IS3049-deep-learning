import numpy as np
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from Model import build_model
from ModelConfig import e_input_shape,img_input_shape
from CustomLoss import loss
from utils import mirror_padding
from Generator import DataGenerator

# sess = K.get_session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-Wenceslas:6004")
# K.set_session(sess)

# Test with CIFAR10 dataset for now, has images of size (32, 32, 3)

(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), *img_input_shape))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), *img_input_shape))  # adapt this if using `channels_first` image data format


train_generator = DataGenerator(x_train,32,img_input_shape)
test_generator = DataGenerator(x_test,32,img_input_shape)

autoencoder = build_model()

# Plot model graph
plot_model(autoencoder, to_file='autoencoder.png')

# Compile model with adadelta optimizer
# TODO: Code loss !


autoencoder.compile(optimizer='adadelta', loss=loss)


# Enable tensorboard just in case
tb = TensorBoard(log_dir='./Graph', histogram_freq=1,
                            write_graph=True, write_images=True)

# Train model !
autoencoder.fit_generator(train_generator,
                epochs=50,
                validation_data=test_generator,
                callbacks=[tb])