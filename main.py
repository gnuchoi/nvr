
# 29 May 2015, Keunwoo Choi, keunwoo.choi@qmul.ac.uk

from setup_environment import *

import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
import h5py # to read/write data
import os
import sys

import cPickle 
import pdb

from constants import *
import numpy as np

import keras
from keras.optimizers import RMSprop
from keras.utils import np_utils

def prepareTexts():

	listpath = TXT_PATH + GTZAN_LIST_FILE
	if os.path.exists(listpath):
		return
	print '--- generate text file of gtzan list ---'

	listoftextfiles = []
	for folder in os.listdir(GTZAN_WAV_PATH):
		if not os.path.isfile(folder):
			for filename in os.listdir(GTZAN_WAV_PATH + folder):
				listoftextfiles.append(folder + '/' + filename + '\n')
				#f_txt.write(folder + '/' + filename + '\n')
	
	f_txt = open( listpath, 'w')
	for names in listoftextfiles:
		f_txt.write(names)
	f_txt.close()
	
	numFiles = len(listoftextfiles)
	
	#traning/valid/test set
	trainingSetIndex = np.random.choice(100, 60, replace=False)
	theothers = [num for num in range(100) if not num in trainingSetIndex]
	validSetIndex = theothers[0:20]
	testSetIndex = theothers[20:40]

	f_training_txt = open( TXT_PATH + TRAIN_LIST_FILE, 'w') # SHOULD BE MOVED IN constants.py
	f_valid_txt = open(TXT_PATH + VALID_LIST_FILE, 'w')
	f_test_txt = open( TXT_PATH + TEST_LIST_FILE, 'w')
	
	list_training = []
	list_valid = []
	list_test = []

	for folder in os.listdir(GTZAN_WAV_PATH):
		if not os.path.isfile(folder):
			for ind, filename in enumerate(os.listdir(GTZAN_WAV_PATH + folder)):
				entry = folder + '/' + filename + '\n'
				if ind in trainingSetIndex:
					f_training_txt.write(entry)
					#list_training.append(folder + '/' + filename + '\n')
				elif ind in validSetIndex:
					f_valid_txt.write(entry)
					#list_valid.append(folder + '/' + filename + '\n')
				else:
					f_test_txt.write(entry)
					#list_test.append(folder + '/' + filename + '\n')
	f_training_txt.close()
	f_valid_txt.close()
	f_test_txt.close()


def prepareGtzan():
	'''
	input: path of gtzan dataset
	what it does: load the files, get stft, save it into the disk, and update h5 dictionary
	'''
	h5filepath = GNU_SPEC_PATH + GTZAN_h5FILE

	if os.path.exists(h5filepath):
		print "let's just load the Gtzan STFTs"
		return
	
	f_h5 = h5py.File(h5filepath)

	for folder in os.listdir(GTZAN_WAV_PATH):
		if not os.path.isfile(folder):
			genre = folder
			for filename in os.listdir(GTZAN_WAV_PATH + folder):
				#check if it already in the h5 file..->no, just do it all.
				# if not filename in f_h5.keys():
				src, sr = librosa.load(GTZAN_WAV_PATH + folder + '/' + filename)
				#1. _stft: spectrogram
				specgram = np.absolute(librosa.stft(src, n_fft=N_FFT, hop_length=N_HOP, win_length=N_WIN, window=TYPE_WIN)) # STFT of signal
				dset = f_h5.create_dataset(filename + '_stft', data=specgram, chunks=True)
				dset.attrs['genre'] = genre
				#2. _mfcc; mfcc
				mfcc = librosa.feature.mfcc(src, sr, n_mfcc=N_MFCC)
				dset = f_h5.create_dataset(filename + '_mfcc', data=mfcc, chunks=True)
				dset.attrs['genre'] = genre
				print filename + ' is STFTed/MFCCed and stored.'
	f_h5.close()

def genreToClass():
	retDict = {'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4,
				'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9}
	return retDict


def buildModel(n_input):
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation
	from keras.optimizers import RMSprop

	model = Sequential()
	model.add(Dense(n_input , 256, init='lecun_uniform')) #500: arbitrary number
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	'''
	model.add(Dense(128, 128, init='lecun_uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	'''
	model.add(Dense(256, 128, init='lecun_uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(128, 10, init='lecun_uniform'))
	model.add(Activation('softmax'))

	#sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='categorical_crossentropy', optimizer=sgd)
	rms = RMSprop()
	model.compile(loss='categorical_crossentropy', optimizer=rms)
	return model


def die_with_usage():
	""" HELP MENU """
	print '-'*20 + ' usage ' + '-'*20
	print 'keras-based Deep NN for GTZAN prediction.'
	print '$ python main.py minNumFr nb_epoch'
	print '     gtzan, * minNumFr < 1290 (when N_FFT = 1024)'
	print '            * nb_epoch : 1~10~200~, any integer.'
	print '-'*46
	

	sys.exit(0)

if __name__ == '__main__':

	if len(sys.argv) != 3:
		die_with_usage()
	if int(sys.argv[1]) > 1290:
		print 'argv[2], minNumFr must be <= 1290 and positive integer.'
		die_with_usage()

	modelname_suffix = '_' + sys.argv[1] + '_' + sys.argv[2]

	GTZAN_h5FILE = GTZAN_h5FILE_BASENAME + '_' + str(N_FFT) + '.h5' #should be done before prepareGtzan()

	#get file list in the source folder	
	# prepare() #old one for the beginning.
	prepareTexts()
	prepareGtzan()
	genreToClassDict = genreToClass()
	'''
	minNumFr = 99999
	for i in range(len(f_h5)):
		minNumFr = min(minNumFr, f_h5[f_h5.keys()[i]].shape[1])
	#Now I Know, it's 1290 for GTZAN
	'''
	#about optimisation
	batch_size = 64
	nb_classes = 10
	nb_epoch = 9
	nb_epoch = int(sys.argv[2])
	
	#spectrogram constants
	minNumFr = 600
	minNumFr = 5 #to reduce the datapoints, for temporary.
	minNumFr = int(sys.argv[1])

	lenFreq = N_FFT/2+1 #length on frequency axis
	
	#about training data loading
	numGenre = 10
	numSongPerGenre = 100
	portionTraining = 0.8
	portionTesting = 0.2

	numIteration = 1
 
	#for iter_i in range(numIteration):
	
	print '--- prepare data --- '

	f_training_txt = open( TXT_PATH + TRAIN_LIST_FILE, 'r') 
	f_valid_txt = open(TXT_PATH + VALID_LIST_FILE, 'r')
	f_test_txt = open( TXT_PATH + TEST_LIST_FILE, 'r')

	train_files = f_training_txt.readlines()
	valid_files = f_valid_txt.readlines()
	test_files  = f_test_txt.readlines()
	
	h5filepath = GNU_SPEC_PATH + GTZAN_h5FILE
	f_h5 = h5py.File(h5filepath, 'r')
	
	numDataPoints = int(portionTraining * numSongPerGenre) * numGenre * minNumFr
	training_x = np.zeros((numDataPoints, lenFreq + N_MFCC))
	training_y = np.zeros((numDataPoints,1))	
	print '          p.s. numDataPoints: ' + str(numDataPoints)
	for ind, train_file in enumerate(train_files):
		filenameHere = train_file.split('/')[1].rstrip('\n')
		genre = f_h5[filenameHere + '_stft'].attrs['genre']
		specgram = f_h5[filenameHere + '_stft'][:, 0:minNumFr]
		mfcc = f_h5[filenameHere + '_mfcc'][:, 0:minNumFr]

		genreToClassDict[genre]
		training_x[ind*minNumFr: (ind+1)*minNumFr, :] = np.hstack((np.transpose(specgram), np.transpose(mfcc)))
		training_y[ind*minNumFr: (ind+1)*minNumFr, :] = np.ones((specgram.shape[1], 1))* genreToClassDict[genre] # int, 0-9


	'''
	for genre_i in range(numGenre):
		for song_i in range(int(portionTraining * numSongPerGenre)): # 0:80	
			ind = genre_i* numSongPerGenre + song_i # 100
			indToWrite = genre_i * int(portionTraining * numSongPerGenre) + song_i # 80
			genre = f_h5[f_h5.keys()[ind]].attrs['genre']
			specgram = f_h5[f_h5.keys()[ind]][:,0:minNumFr] # 513x1290
			
			training_x[indToWrite*minNumFr:(indToWrite+1)*minNumFr, : ] = np.transpose(specgram)
			training_y[indToWrite*minNumFr:(indToWrite+1)*minNumFr, : ] = np.ones((specgram.shape[1], 1)) * genreToClassDict[genre] # int, 0-9
	'''
	print '--- training data loaded ---'
	#after loading from all genre, let's make it appropriate for the model
	print '--- model fitting! ---'
	training_y = np_utils.to_categorical(training_y, nb_classes)
	
	training_x = training_x.astype("float32")

	
	#about model
	model = buildModel(lenFreq + N_MFCC)
	model.fit(training_x, training_y, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2)		
	#cPickle.dump(model, open(DATA_PATH + MODEL_FILE + modelname_suffix, "wb"))

	print '--- prepare test data  ---'
	numDataPoints = int(portionTraining * numSongPerGenre) * numGenre * minNumFr
	test_x = np.zeros((numDataPoints, lenFreq + N_MFCC))
	test_y = np.zeros((numDataPoints,1))	

	for ind, test_file in enumerate(train_files):
		filenameHere = train_file.split('/')[1].rstrip('\n')
		genre = f_h5[filenameHere + '_stft'].attrs['genre']
		specgram = f_h5[filenameHere + '_stft'][:, 0:minNumFr]
		mfcc = f_h5[filenameHere + '_mfcc'][:, 0:minNumFr] 

		test_x[ind*minNumFr: (ind+1)*minNumFr, :] = np.hstack((np.transpose(specgram), np.transpose(mfcc)))
		test_y[ind*minNumFr: (ind+1)*minNumFr, :] = np.ones((specgram.shape[1], 1))* genreToClassDict[genre] # int, 0-9

	print '--- test data loaded ---'
	print '--- prediction! for ---'
	test_x = test_x.astype("float32")
	test_y = np_utils.to_categorical(test_y, nb_classes)
	score = model.evaluate(test_x, test_y, show_accuracy=True, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	pdb.set_trace()

	
