import platform
deviceName = platform.node().lower()


if deviceName == 'exeter.eecs.qmul.ac.uk':
	isLaptop = False	
	isServer = True
	isExeter = True
elif deviceName.rstrip('eecs.qmul.ac.uk') == 'frank' or deviceName.rstrip('eecs.qmul.ac.uk') == 'jackal' or deviceName.rstrip('eecs.qmul.ac.uk') == 'jazz':
	print '='*30
	print 'Do not run it on frank'
	print 'Move to Exeter server!'
	print '='*30

else:
	print '='*30
	print 'Local machine. right?'
	print '='*30
	isLaptop = True
	isServer = False
	isExeter = False




if isLaptop:
	PHD_PATH = '/Users/gnu/GoogleDrive/phdCodes/'
	SRC_PATH = '/Users/gnu/Gnubox/Srcs/music/'
	GNU_SPEC_PATH =  '/Users/gnu/Gnubox/datasets/gtzan_stft/'
	GTZAN_WAV_PATH = '/Users/gnu/Gnubox/datasets/gtzan/'

if isServer:
	PHD_PATH = '/homes/kc306/'
	SRC_PATH = PHD_PATH + 'srcs/'
	SID_SPEC_PATH = '/import/c4dm-02/people/siddharths/Chords/features/11025_4096_2048.h5'
	GNU_SPEC_PATH = '/import/c4dm-02/people/keunwoo/gtzan_stft/'
	GTZAN_WAV_PATH = '/import/c4dm-datasets/gtzan/'

PROJ_FOLDER = 'nvr/' # under the PHD PATH
PROJ_PATH = PHD_PATH + PROJ_FOLDER

DATA_FOLDER = 'data/' # under PROJ Folder
TXT_FOLDER  = 'txt/'
LYRICS_FOLDER= 'lyrics/'

DATA_PATH = PROJ_PATH + DATA_FOLDER
TXT_PATH  = PROJ_PATH + TXT_FOLDER
LYRICS_PATH = PROJ_PATH + LYRICS_FOLDER

TRAIN_LIST_FILE = 'gtzan_train1.txt'
VALID_LIST_FILE = 'gtzan_valid1.txt'
TEST_LIST_FILE  = 'gtzan_test1.txt'

GTZAN_h5FILE_BASENAME = 'dict'
GTZAN_DATA = 'gtzan_data.p'
GTZAN_TRAINING_TEST = 'gtzan_training_test.p'

MODEL_FILE = 'model.p'

GTZAN_LIST_FILE = 'gtzan_filelist.txt'

