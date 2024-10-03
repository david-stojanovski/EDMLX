from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET = edict()
__C.DATASET.DATA_DIR = r'/home/ds17/Documents/phd/edm/segmentation_data/CAMUS'
__C.DATASET.DATA_TYPE = '.png'
__C.DATASET.RESULTS_DIR = r'/home/ds17/Documents/phd/edm/shit'
__C.DATASET.IMAGE_SIZE = 256

__C.NETWORK = edict()
__C.NETWORK.NUM_CLASSES = 2
__C.NETWORK.IN_CHANNELS = 1
__C.NETWORK.COMPILE = False
__C.NETWORK.LOAD_PATH = r'/home/ds17/Documents/phd/edm/segmentation/results_classification/camus_real/model.pth'

__C.CONST = edict()
__C.CONST.BATCH_SIZE = 32
__C.CONST.NUM_WORKERS = 10
__C.CONST.PIN_MEMORY = True
__C.CONST.GPU_ID = 'cuda:0'

# Only used for the training in train_classification.py
__C.TRAIN = edict()
__C.TRAIN.NUM_EPOCHS = 250
__C.TRAIN.LR = 1e-4
__C.TRAIN.SHUFFLE = True
__C.TRAIN.SEED = None

# Only used for the testing in test_classification.py
__C.TEST = edict()
__C.TEST.NUM_REPEATS = 1000
__C.TEST.SUBSET_FRAC = 0.8
