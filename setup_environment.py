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
        import sys
        sys.path.append("modules/librosa")

    

PROJ_FOLDER = 'nvr/'
DATA_FOLDER = 'data/'

PROJ_PATH = PHD_PATH + PROJ_FOLDER
DATA_PATH = PHD_PATH + PROJ_FOLDER + DATA_FOLDER
