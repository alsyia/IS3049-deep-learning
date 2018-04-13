import os
import argparse
import PIL.Image
import numpy as np

from Generator import DataGenerator
from Model import build_model, Model
from ModelConfig import INPUT_SHAPE
from keras.applications import VGG19


def predict_from_ae(input_path, autoencoder, limit=10):
    if os.path.isfile(input_path):
        img_list = [input_path]
    elif os.path.isdir(input_path):
        img_list = [input_path + '/' + x for x in os.listdir(input_path)]
    else:
        raise Exception("input path does not exist")

    if not os.path.exists("output"):
        os.mkdir('output')

    for img_idx in range(min(limit, len(img_list))):
        img = PIL.Image.open(img_list[img_idx])
        img_img = img.resize(INPUT_SHAPE[0:2], PIL.Image.ANTIALIAS)
        img = np.asarray(img_img) / 255
        img = img.reshape(1, *INPUT_SHAPE)
        reconstruction = autoencoder.predict(img)
        reconstruction = reconstruction[1] * 255
        reconstruction = np.clip(reconstruction, 0, 255)
        reconstruction = np.uint8(reconstruction)
        reconstruction = reconstruction.reshape(*INPUT_SHAPE)
        reconstruction_img = PIL.Image.fromarray(reconstruction)

        filename = os.path.basename(img_list[img_idx]).split('.')[0]
        img_img.save("output/" + filename + "_true.png")
        reconstruction_img.save("output/" + filename + "_pred.png")


def predict_from_weights(input_path, weight_path, limit=10):
    # VGG for the perceptual loss
    base_model = VGG19(weights="imagenet", include_top=False,
                       input_shape=INPUT_SHAPE)

    perceptual_model = Model(inputs=base_model.input,
                             outputs=[base_model.get_layer("block2_pool").output,
                                      base_model.get_layer("block5_pool").output],
                             name="VGG")
    autoencoder, _ = build_model(perceptual_model)

    if os.path.isfile(weight_path):
        print("loading weights from {}".format(weight_path))
        autoencoder.load_weights(weight_path)
    else:
        raise Exception("weight path does not exist")

    predict_from_ae(input_path, autoencoder, limit)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='predict images')

    argparser.add_argument(
        '-w',
        '--weight',
        help='path to weight file')

    argparser.add_argument(
        '-i',
        '--input',
        help='path to input file or folder')

    argparser.add_argument(
        '-l',
        '--limit',
        help='maximum number of prediction',
        default=10)

    args = argparser.parse_args()

    limit = int(args.limit)
    input_path = args.input
    weight_path = args.weight

    predict_from_weights(input_path, weight_path, limit)
