import platform
deviceName = platform.node()


if deviceName == 'KChoi.MBPR.2013.home' or deviceName == 'KChoiMBPR2013.local':
	isLaptop = True
	isServer = False
	isExeter = False
elif deviceName == 'exeter.eecs.qmul.ac.uk':
	isLaptop = False	
	isServer = True
	isExeter = True
else:
    print "WHERE ARE YOU???"


if isLaptop:
	PHD_PATH = '/Users/gnu/Gnubox/Study_PhD/'
	SRC_PATH = '/Users/gnu/Gnubox/Srcs/music/'
	GNU_SPEC_PATH =  '/Users/gnu/Gnubox/datasets/gtzan_stft/'
	GTZAN_WAV_PATH = '/Users/gnu/Gnubox/datasets/gtzan/'


else:
	PHD_PATH = '/homes/kc306/'
	SRC_PATH = PHD_PATH + 'srcs/'

if isServer:
	SID_SPEC_PATH = '/import/c4dm-02/people/siddharths/Chords/features/11025_4096_2048.h5'
	GNU_SPEC_PATH = '/import/c4dm-02/people/keunwoo/gtzan_stft/'
	GTZAN_WAV_PATH = '/import/c4dm-datasets/gtzan/'

PROJ_FOLDER = 'nvr/'
DATA_FOLDER = 'data/'
TXT_FOLDER  = 'txt/'
LYRICS_FOLDER= 'lyrics/'


PROJ_PATH = PHD_PATH + PROJ_FOLDER
DATA_PATH = PHD_PATH + PROJ_FOLDER + DATA_FOLDER
TXT_PATH  = PHD_PATH + PROJ_FOLDER + TXT_FOLDER
LYRICS_PATH = PHD_PATH + PROJ_FOLDER + LYRICS_FOLDER

TRAIN_LIST_FILE = 'gtzan_train1.txt'
VALID_LIST_FILE = 'gtzan_valid1.txt'
TEST_LIST_FILE  = 'gtzan_test1.txt'

GTZAN_h5FILE_BASENAME = 'dict'
GTZAN_DATA = 'gtzan_data.p'
GTZAN_TRAINING_TEST = 'gtzan_training_test.p'

MODEL_FILE = 'model.p'



