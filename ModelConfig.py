# Dataset
DATASET_PATH = "./celeba64_debug100"
TEST_DIR = "test"
TRAIN_DIR = "train"
VALIDATION_DIR = "val"

BATCH_SIZE = 16
EPOCH_NB = 30

# Encoder
ALPHA = 0.3  # Leaky ReLu alpha value
MIRROR = 12  # number of zeros on one side added by mirror padding (see Generator)
INPUT_SHAPE = (64, 64, 3)  # Encoder input shape (32, 32, 3) for CIFAR
E_INPUT_SHAPE = INPUT_SHAPE
E_RES_BLOCKS_CONV_PARAMS = {  # Parameters of residual blocks
    "filters": 128,
    "kernel_size": (3, 3),
    "padding": "same",
}

# Decoder
D_RES_BLOCKS_CONV_PARAMS = {  # Parameters of residual blocks
    "filters": 128,
    "kernel_size": (3, 3),
    "padding": "same",
}

# Texture loss
TEXTURE_PARAMS = {
    "patch_size": (8, 8)
}


# Loss
LOSS_PARAMS = {
    "mse": 1,
    "bit": 0,
    "entropy": 0.01,
    "perceptual_2": 0.001,
    "perceptual_5": 0.1
}
