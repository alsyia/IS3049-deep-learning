import numpy as np
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from Model import build_model
from ModelConfig import e_input_shape

# sess = K.get_session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "PC-Wenceslas:6004")
# K.set_session(sess)

# Test with CIFAR10 dataset for now, has images of size (32, 32, 3)

(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), *e_input_shape))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), *e_input_shape))  # adapt this if using `channels_first` image data format

autoencoder = build_model()

# Plot model graph
plot_model(autoencoder, to_file='autoencoder.png')

# Compile model with adadelta optimizer
# TODO: Code loss !
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Enable tensorboard just in case
tb = TensorBoard(log_dir='./Graph', histogram_freq=0,
                            write_graph=True, write_images=True)

# Train model !
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tb])