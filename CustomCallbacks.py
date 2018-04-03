import io

import PIL.Image
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import PIL
import numpy as np
import warnings
from ModelConfig import *


class PredictCallback(Callback):
    def __init__(self,generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs={}):
        imgs = self.generator[0][0]
        img = PIL.Image.fromarray(np.uint8(imgs[0]*255))
        img.show()
        imgs = self.model.predict(imgs)
        img = PIL.Image.fromarray(np.uint8(imgs[0]*255))
        img.show()

class HuffmanCallback(Callback):
    def __init__(self,obj_values, generator):
        self.obj_values = obj_values
        self.generator = generator

    def on_epoch_begin(self, epoch, logs={}):
        codes = self.model.layers[1].predict(self.generator[0][0])[0]
        values, counts = np.unique(codes, return_counts = True)
        self.obj_values.values = values[np.argsort(counts)]
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
                            self.model.layers[1].save_weights(filepath, overwrite=True)
                        else:
                            self.model.layers[1].save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.layers[1].save_weights(filepath, overwrite=True)
                else:
                    self.model.layers[1].save(filepath, overwrite=True)

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
    img_img = img.resize(img_input_shape[0:2], PIL.Image.ANTIALIAS)
    img = np.asarray(img_img) / 255
    img = img.reshape(1, *img_input_shape)
    return img


def output_to_tf_img(output):
    output = np.uint8(output * 255)
    output = output.reshape(*img_input_shape)
    output_img = make_image(output)
    return output_img


class TensorBoardImage(Callback):

    def __init__(self, tag, test_list, logs_path):
        super().__init__()
        self.tag = tag
        self.logs_path = logs_path
        self.test_list = test_list

    def on_epoch_end(self, epoch, logs=None):
        for idx, img_name in enumerate(self.test_list[:2]):
            path = dataset_path + "/" + test_dir + "/" + img_name

            input = image_to_input(path)
            output = self.model.predict(input)[0]
            output_img = output_to_tf_img(output)

            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag + str(idx), image=output_img)])
            writer = tf.summary.FileWriter(self.logs_path)
            writer.add_summary(summary, epoch)
            writer.close()
