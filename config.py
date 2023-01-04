# =============== Dataset/Dataloader Configs ============
# DATA_DIR = './data'
DATA_DIR = '/scratch/frankyu/NeuralRendering/DECA/processed/frank_main'
DATASET_NAME = "imagenet" 

TRAIN = 'ear-adjust-train'
VALIDATION = 'ear-adjust-validation'
TEST = 'ear-adjust-test'

# Use augmentation
USE_AUG = 1 
RANDOM_FLIP = 0
FLOAT_IMAGE = 0
T_DIFF = 2

# =============== Logging Configurations ===========
USE_TENSORBOARD = 0 # if 0, use wandb

CHECKPOINT_DIR = 'logs'
LOG_DIR = 'logs'
JOB_ID = 111

# =============== Training Configurations ===========
EPOCH = 150
BATCH_SIZE = 4 
CROP_W = 256
CROP_H = 256
LEARNING_RATE = 1e-3
BETAS = '0.9, 0.999'
L2_WEIGHT_DECAY = '0.001, 0.0001, 0.00001, 0'
EPS = 1e-8
LOAD = None
LOAD_STEP = 0
EPOCH_PER_CHECKPOINT = 50

NT_COEF = 1.
UNET_DECAY = 0.0
USE_LR_DECAY = 1

# =============== Warping Model Configs ============
TEXTURE_W = 1024
TEXTURE_H = 1024
TEXTURE_DIM = 16
USE_PYRAMID = True
VIEW_DIRECTION = 1

WARP_SAMPLE = 0
CONCAT_UV_DIFF = 1 
CACHE_FEATURES = 1
SH_SKIPS = 1 

USE_POSE = 1

SH_POSE = 1
USE_MLP = 1 

USE_MASK = 1
USE_UPCONV = 1
USE_WARP = 1
USE_INSTANCE_NORM = 0
OUR_MODEL = 1

NETWORK_SIZE = 2 
PREDICT_MASK = 1

CACHE_LAYER = 3
C_SELECTION = 345

SKIP_CACHE_DIM = 8
USE_EXP = 1

# =============== Lowpass Filter Conifg ============
USE_LOWPASS = 1 
LOWPASS_SIGMA = 2

# =============== Baseline Model Configs ============
# DNR++ Baseline Config
SINGLE_TEMPORAL = 0 # SHOULD BE 0










