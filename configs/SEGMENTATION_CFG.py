from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET = edict()
__C.DATASET.DATA_DIR = r'/home/ds17/Documents/phd/edm/segmentation_data/CAMUS'
__C.DATASET.DATA_TYPE = '.png'
__C.DATASET.RESULTS_DIR = r'/home/ds17/Documents/phd/edm/shit'
__C.DATASET.IMAGE_SIZE = 256

__C.NETWORK = edict()
__C.NETWORK.NUM_CLASSES = 4
__C.NETWORK.DROPOUT = 0.1
__C.NETWORK.IN_CHANNELS = 1
__C.NETWORK.RES_BLOCK = True
__C.NETWORK.COMPILE = False
__C.NETWORK.LOAD_PATH = None  # must include this for testing

__C.CONST = edict()
__C.CONST.BATCH_SIZE = 128
__C.CONST.NUM_WORKERS = 10
__C.CONST.PIN_MEMORY = True
__C.CONST.GPU_ID = 'cuda:0'

__C.TRAIN = edict()
__C.TRAIN.NUM_EPOCHS = 250
__C.TRAIN.LR = 1e-4
__C.TRAIN.SHUFFLE = True
__C.TRAIN.SEED = None
