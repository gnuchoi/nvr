import platform
deviceName = platform.node()


if deviceName == 'KChoi.MBPR.2013.home':
	isLaptop = True
	isServer = False
	isExeter = False
elif deviceName == 'eceter.eecs.qmul.ac.uk':
	isLaptop = False
	isServer = True
	isExeter = True


if isLaptop:
	PHD_PATH = '/Users/gnu/Gnubox/Study_PhD/'

	SRC_PATH = '/Users/gnu/Gnubox/Srcs/music/'
else:
	PHD_PATH = '~/'
	SRC_PATH = '~/srcs/'



PROJ_FOLDER = 'nvr/'
DATA_FOLDER = 'data/'

PROJ_PATH = PHD_PATH + PROJ_FOLDER
DATA_PATH = PHD_PATH + PROJ_FOLDER + DATA_FOLDER