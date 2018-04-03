import io

import PIL.Image
import tensorflow as tf
from keras.callbacks import Callback
import PIL
import numpy as np

from ModelConfig import *


class PredictCallback(Callback):
    def __init__(self,generator):
        self.generator = generator

    def on_epoch_end(self, batch, logs={}):
        imgs = self.generator[0][0]
        img = PIL.Image.fromarray(np.uint8(imgs[0]*255))
        img.show()
        imgs = self.model.predict(imgs)
        img = PIL.Image.fromarray(np.uint8(imgs[0]*255))
        img.show()


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
    # return tf.summary.image("Reconstruction", output)

    return output_img


class TensorBoardImage(Callback):

    def __init__(self, tag, test_list, logs_path):
        super().__init__()
        self.tag = tag
        self.logs_path = logs_path
        self.test_list = test_list

    def on_epoch_end(self, epoch, logs=None):
        for idx, img_name in enumerate(self.test_list[:1]):
            path = dataset_path + "/" + test_dir + "/" + img_name

            input = image_to_input(path)
            output = self.model.predict(input)
            output_img = output_to_tf_img(output)
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag + "_" + str(img_name), image=output_img)])
            writer = tf.summary.FileWriter(self.logs_path)
            writer.add_summary(summary, epoch)
            # writer.add_summary(K.eval(output_img), epoch)
            writer.close()
