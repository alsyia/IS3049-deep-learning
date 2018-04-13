import PIL
import os
from utils import extract_patches
import numpy as np
from keras.datasets import cifar10
from keras.applications import VGG19
from keras.models import Model
from ModelConfig import *

def split_indices(nb_indices,train_ratio,val_ratio, seed = 8):
    seed = np.random.seed(seed)

    indices = np.arange(nb_indices)
    indices = np.random.permutation(indices)

    train_index = int(train_ratio*len(indices))
    val_index = int(train_ratio*len(indices))+int(val_ratio*len(indices))

    return indices[:train_index],indices[train_index:val_index], indices[val_index:]


def store_data(src,dst,img_size,img_train,img_val,img_test):
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    if not os.path.exists(dst + '/train'):
        os.mkdir(dst + '/train')

    if not os.path.exists(dst + '/val'):
        os.mkdir(dst + '/val')

    if not os.path.exists(dst + '/test'):
        os.mkdir(dst + '/test')
    
    for img_idx in range(len(img_train)):
        img = PIL.Image.open(src + "/" + img_train[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save(dst + '/train/'+img_train[img_idx].split('.')[0] + ".png")

    for img_idx in range(len(img_val)):
        img = PIL.Image.open(src + "/" + img_val[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save(dst + '/val/'+img_train[img_idx].split('.')[0] + ".png")

    for img_idx in range(len(img_test)):
        img = PIL.Image.open(src + "/" + img_test[img_idx])
        img = img.resize(img_size, PIL.Image.ANTIALIAS)
        img.save(dst + '/test/'+img_train[img_idx].split('.')[0] + ".png")

def load_cifar10(dst = "cifar10", nb_images = 2000):
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    if not os.path.exists(dst + "/train"):
        os.mkdir(dst + "/train")
    
    if not os.path.exists(dst + "/val"):
        os.mkdir(dst + "/val")
    
    if not os.path.exists(dst + "/test"):
        os.mkdir(dst + "/test")
    (x_train, _), (x_test, _) = cifar10.load_data()
    cifar = np.concatenate([x_train,x_test])
    cifar = cifar[:min(nb_images,cifar.shape[0])]

    train_index, val_index, test_index = split_indices(cifar.shape[0],0.7,0.2)
    for idx in train_index:
        img = PIL.Image.fromarray(np.uint8(cifar[idx,:,:,:]))
        img.save(dst + "/train/train_{}.png".format(idx))
    
    for idx in val_index:
        img = PIL.Image.fromarray(np.uint8(cifar[idx,:,:,:]))
        img.save(dst + "/val/val_{}.png".format(idx))
    for idx in test_index:
        img = PIL.Image.fromarray(np.uint8(cifar[idx,:,:,:]))
        img.save(dst + "/test/test_{}.png".format(idx))

def load_folder(src,dst,img_size,nb_images,train_ratio = 0.7, val_ratio = 0.2):
    img_list = os.listdir(src)
    img_list = img_list[:min(nb_images,len(img_list))]

    train_index, val_index, test_index = split_indices(len(img_list),train_ratio,val_ratio)

    train_list = [img_list[i] for i in train_index]
    val_list = [img_list[i] for i in val_index]
    test_list = [img_list[i] for i in test_index]

    store_data(src,dst,img_size,train_list,val_list,test_list)

def preprocess_patch(src,dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    src_list = os.listdir(src)

    # VGG for the perceptual loss
    base_model = VGG19(weights="imagenet", include_top=False,
                    input_shape=INPUT_SHAPE)

    texture_model = Model(inputs=base_model.input,
                            outputs=[base_model.get_layer("block2_pool").output],
                            name="VGG_texture")
    
    for idx in range(len(src_list)):
            img = PIL.Image.open(src + "/" + src_list[idx])
            img = img.resize(INPUT_SHAPE[0:2], PIL.Image.ANTIALIAS)
            img = np.asarray(img)
            img = img / 255
            X = np.reshape(img,(1,*img.shape))

            patch_size = TEXTURE_PARAMS["patch_size"]
            patches_per_img = (INPUT_SHAPE[0] // patch_size[0]) * (INPUT_SHAPE[1] // patch_size[1])
            # On génère les patchs
            # print("[Generator] Shape of X: " + str(X.shape))
            # print("[Generator] Shape of F2: " + str(F2.shape))
            # Renvoie un tenseur de taille (batch_size*#patches, 8, 8, 3)
            patches = extract_patches(X, (*patch_size, 3))
            # print("[Generator] Patches array shape : " + str(patches.shape))
            # On padde : (batch_size*#patches, 64, 64, 3)
            pad_size_h = (INPUT_SHAPE[0] - patch_size[0]) // 2
            pad_size_v = (INPUT_SHAPE[1] - patch_size[1]) // 2
            padded_patches = np.pad(patches, [[0, 0], [pad_size_h, pad_size_h], [pad_size_v, pad_size_v], [0, 0]], "constant")
            # print("[Generator] Padded patches array shape : " + str(padded_patches.shape))
            # On envoie tout ça comme un gros batch dans le VGG
            # On récupère une liste de 2048 éléments de taille (16, 16, 128)
            textures = texture_model.predict(padded_patches)
            # print("[Generator] Textures array shape : " + str(textures[0].shape))
            # print("[Generator] Texture list length : " + str(len(textures)))
            # On fait des arrays de taille (64, 16, 16, 128), chaque array correspondant aux textures d'une
            # image
            textures_grouped = []
            for i in range(0, len(textures), patches_per_img):
                textures_grouped.append(np.stack(textures[i:i+patches_per_img], 0))
            # print("[Generator] Grouped textures array shape : " + str(textures_grouped[0].shape))
            # print("[Generator] Grouped textures list length : " + str(len(textures_grouped)))
            # Et on stacke pour avoir un tenseur (32, 64, 16, 16, 128) qui se lit comme suit :
            #   Pour chaque image
            #       Pour chaque patch
            #           Sortie du VGG de taille (16, 16, 128)
            textures_stacked = np.stack(textures_grouped, axis=0)
            # print("[Generator] Final texture shape : " + str(textures_stacked.shape))

            src_name = dst + '/' + os.path.basename(src_list[idx]).split(".")[0] + "_patches.npy"
            np.save(src_name,textures_stacked[0,:,:,:,:])

#load_cifar10("cifar10", 2000)
#load_folder("test", "test2",(64,64),1)
#store_data('test','test2',(64,64),['1.jpg','2.png'],[],[])
preprocess_patch("celeba64_debug/val","texture_val")