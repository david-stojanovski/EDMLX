from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET = edict()
__C.DATASET.DATA_DIR = r'/home/ds17/Documents/phd/edm/segmentation_data/CAMUS'
__C.DATASET.DATA_TYPE = '.png'
__C.DATASET.RESULTS_DIR = r'/home/ds17/Documents/phd/edm/shit'
__C.DATASET.IMAGE_SIZE = 256

__C.NETWORK = edict()
__C.NETWORK.ALPHA_PARAM = 3.75
__C.NETWORK.BETA_PARAM = 10.8
__C.NETWORK.NUM_LAYERS = 3  # number of layers in autoencoder: 2 for EDM-L128 and 3 for EDM-L64
__C.NETWORK.IN_CHANNELS = 1
__C.NETWORK.OUT_CHANNELS = 1
__C.NETWORK.LATENT_CHANNELS = 1
__C.NETWORK.NUM_RES_BLOCKS = 2
__C.NETWORK.NUM_NORM_GROUPS = 32
__C.NETWORK.USE_FLASH_ATTENTION = True
__C.NETWORK.COMPILE = True
__C.NETWORK.LOAD_PATH = None

__C.CONST = edict()
__C.CONST.BATCH_SIZE = 6
__C.CONST.NUM_WORKERS = 10
__C.CONST.PIN_MEMORY = True
__C.CONST.AMP = True
__C.CONST.GPU_ID = 'cuda:0'

# Only used for the training in train_classification.py
__C.TRAIN = edict()
__C.TRAIN.NUM_EPOCHS = 250
__C.TRAIN.VALIDATION_INTERVAL = 1
__C.TRAIN.LR = 1e-4  # NOTE: multiplied by ddp world_size
__C.TRAIN.MIN_LR = 1e-4  # lr will decay to this value over training cycle using cosine annealing lr schedule
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.NESTEROV = True
__C.TRAIN.SHUFFLE = True
__C.TRAIN.SEED = None

__C.LOSS = edict()
__C.LOSS.PERCEPTUAL_WEIGHT = 1e-3
__C.LOSS.KL_WEIGHT = 1e-3
__C.LOSS.MSE_WEIGHT = 10
__C.LOSS.MSE_LATENT_WEIGHT = 1e-2
