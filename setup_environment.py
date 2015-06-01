import platform
deviceName = platform.node()


if deviceName == 'KChoi.MBPR.2013.home':
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
else:
	PHD_PATH = '/homes/kc306/'
	SRC_PATH = PHD_PATH + 'srcs/'

if isServer:
	SID_SPEC_PATH = '/import/c4dm-02/people/siddharths/Chords/features/11025_4096_2048.h5'


PROJ_FOLDER = 'nvr/'
DATA_FOLDER = 'data/'
TXT_FOLDER  = 'txt/'

PROJ_PATH = PHD_PATH + PROJ_FOLDER
DATA_PATH = PHD_PATH + PROJ_FOLDER + DATA_FOLDER
TXT_PATH  = PHD_PATH + PROJ_FOLDER + TXT_FOLDER

TRAIN_FiLE = 'train_list.txt'
VALID_FiLE = 'valid_list.txt'
TEST_FiLE  = 'test_list.txt'
