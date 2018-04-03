from keras.callbacks import Callback
import PIL

class PredictCallback(Callback):
    def __init__(self,generator):
        self.generator = generator

    def on_batch_end(self, batch, logs={}):
        imgs = self.model.predict(self.generator[0])
        img = PIL.Image.fromarray(numpy.uint8(imgs[0]))
        img[0].show()