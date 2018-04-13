import io
import warnings
import os
import PIL
import PIL.Image
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback

from ModelConfig import *


class PredictCallback(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs={}):
        imgs = self.generator[0][0]
        img = PIL.Image.fromarray(np.uint8(imgs[0]*255))
        img.show()
        imgs = self.model.predict(imgs)
        img = PIL.Image.fromarray(np.uint8(imgs[0]*255))
        img.show()


class HuffmanCallback(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_begin(self, epoch, logs={}):
        codes = self.model.predict(self.generator[0][0])[0]
        values, counts = np.unique(codes, return_counts=True)
        values = values[np.argsort(counts)]
        print("values : {}".format(values))


class EncoderCheckpoint(Callback):
    """ Same as ModelCheckpoint but for encoder
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(EncoderCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EncoderCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def make_image(tensor):
    height, width, channel = tensor.shape
    image = PIL.Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


def image_to_input(path):
    img = PIL.Image.open(path)
    img_img = img.resize(INPUT_SHAPE[0:2], PIL.Image.ANTIALIAS)
    img = np.asarray(img_img) / 255
    img = img.reshape(1, *INPUT_SHAPE)
    return img


def output_to_tf_img(output):
    output = np.uint8(output * 255)
    output = output.reshape(*INPUT_SHAPE)
    output_img = make_image(output)

    return output_img


class TensorBoardImage(Callback):

    def __init__(self, tag, test_list, logs_path, save_img=False, exp_path=None):
        super().__init__()
        self.tag = tag
        self.logs_path = logs_path
        self.test_list = test_list
        self.exp_path = exp_path
        self.save_img = save_img

    def on_epoch_end(self, epoch, logs=None):
        summaries = []
        for idx, img_name in enumerate(self.test_list[:10]):
            path = DATASET_PATH + "/" + TEST_DIR + "/" + img_name

            input = image_to_input(path)
            output = self.model.predict(input)[1]
            output_img = output_to_tf_img(output)
            if self.save_img:
                img = PIL.Image.fromarray(np.uint8(output[0]*255))
                file_name = img_name.split('.')[0] + "_" + str(epoch) + ".png"
                img.save(self.exp_path + "/" + file_name)
            summary = tf.Summary.Value(
                tag=self.tag + "_" + str(img_name), image=output_img)
            summaries.append(summary)
        big_sum = tf.Summary(value=summaries)
        writer = tf.summary.FileWriter(self.logs_path)
        writer.add_summary(big_sum, epoch)
