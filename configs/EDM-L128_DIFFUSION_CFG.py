from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET = edict()
__C.DATASET.DATA_DIR = r'/home/ds17/Documents/phd/edm/data/all_views_combo'
__C.DATASET.RESULTS_DIR = r'/home/ds17/Documents/phd/edm/shit'
__C.DATASET.VAE_CFG_PTH = r'/home/ds17/Documents/phd/EDMLX/configs/VAE_128_CFG.py'
__C.DATASET.LOAD_VAE_MODEL_PTH = r'/home/ds17/Documents/phd/edm/final_results/gamma_128/autoencoder_128_final.pth'
__C.DATASET.LOAD_PKL_PTH = None
__C.DATASET.LOAD_MODEL_PTH = None

__C.NETWORK = edict()
__C.NETWORK.DIFFUSION_RES = 128
__C.NETWORK.OUPUT_RES = 256
__C.NETWORK.COND = False
__C.NETWORK.SEMANTIC_COND_N_CLASSES = 10
__C.NETWORK.ARCH = 'adm'
__C.NETWORK.PRECOND = 'edm'
__C.NETWORK.CBASE = None
__C.NETWORK.CRES = None

__C.CONST = edict()
__C.CONST.BATCH_SIZE = 2
__C.CONST.BATCH_GPU = None
__C.CONST.PIN_MEMORY = True
__C.CONST.AMP = True
__C.CONST.BENCH = True
__C.CONST.CACHE = False  # had OOM issues on our cluster
__C.CONST.NUM_WORKERS = 1
__C.CONST.SEED = None

# Only used for the training in train_classification.py
__C.TRAIN = edict()
__C.TRAIN.DURATION = 0.5  # in units of Mega images
__C.TRAIN.VAL_FREQ = 1e8  # in units of Kilo images
__C.TRAIN.LR = 1e-4  # NOTE: multiplied by ddp world_size
__C.TRAIN.EMA = 0.0
__C.TRAIN.DROPOUT = 0.1
__C.TRAIN.LS = 1
__C.TRAIN.AUGMENT = 0.1  # with USE_AUGMENT_V2 = True this acts just as a bool to do augmentations instead of the probability
__C.TRAIN.USE_AUGMENT_V2 = True  # uses the simple augmentations from EDMLX (must set AUGMENT to >0)

__C.IO = edict()
__C.IO.NOSUBDIR = True
__C.IO.TICK = 1
__C.IO.SNAP = 1e8  # don't want any
__C.IO.DUMP = 1e8  # don't want any
__C.IO.DRY_RUN = False
__C.IO.DESC = False
