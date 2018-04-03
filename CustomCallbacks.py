from keras.callbacks import Callback
import PIL
import numpy as np

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