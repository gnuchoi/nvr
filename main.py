
# 29 May 2015, Keunwoo Choi, keunwoo.choi@qmul.ac.uk

from setup_environment import *

import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
import h5py # to read/write data
import os

import cPickle 
import pdb

from constants import *
import numpy as np

import keras
from keras.optimizers import RMSprop
from keras.utils import np_utils

def prepare():
	def getUnprocessedWaveList():
		def getWaveList():
			''' return: wav file name list, 
			 e.g. ['a.wav', 'b.wav'] '''
			from os import listdir
			from os.path import isfile, join
			filelist = [f for f in listdir(SRC_PATH) if isfile(join(SRC_PATH, f))]
			return [f for f in filelist if f.lower().endswith('.wav')]

		filelist_wav = getWaveList()

		result = []
		for wavfile in filelist_wav:
			if not os.path.exists( DATA_PATH + getSTFTFilename(wavfile) ):
				result.append(wavfile)
				#pdb.set_trace()

		return result

	def getSTFT(wavfilepath):
		''' 
		input:  path of wave file. e.g. blah/blahblah/test.wav
		will cut middle of the file, for 30-s
		output: STFT (numpy array)
		'''
		if os.path.exists(wavfilepath):
			mix,sr = librosa.load(wavfilepath)
			lenSec    = len(mix) / float(sr)
			if lenSec > 30:
				fromSample = len(mix)/2 - 15*sr
				toSample   = fromSample + 30*sr
				mix = mix[fromSample:toSample]

			return librosa.stft(mix, n_fft=N_FFT, hop_length=N_HOP, win_length=N_WIN, window=TYPE_WIN) # STFT of mixture signal
		else:
			print 'NO SUCH FILE - to STFT.'


	def getSTFTFilename(wavfilename):
		'''
		Just a simple file extension converter
		input: filename e.g. text.wav
		output: filename e.g. text.p 
		'''
		return os.path.splitext(wavfilename)[0] + '.p'
	#get spectrogram and save it in cPickle
	wavListToSTFT = getUnprocessedWaveList();

	for wavfile in wavListToSTFT:
		SRC = getSTFT(SRC_PATH + wavfile)
		print 'STFT finished:' + wavfile
		cPickle.dump(SRC, open(DATA_PATH + getSTFTFilename(wavfile), 'wb' )) #write spectrogram

def loadSpectrogramList():
	'''
	load file list in the folder
	'''

def loadSpectrogramFile(filename):
	'''
	load the spectrogram in the file
	'''

def buildModel():
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation
	from keras.optimizers import RMSprop

	model = Sequential()
	model.add(Dense(513, 128, init='uniform')) #500: arbitrary number
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))

	model.add(Dense(128, 32, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))

	model.add(Dense(32, 10, init='uniform'))
	model.add(Activation('softmax'))

	rms = RMSprop()
	model.compile(loss='categorical_crossentropy', optimizer=rms)
	return model

def prepareGtzan():
	'''
	input: path of gtzan dataset
	what it does: load the files, get stft, save it into the disk, and update h5 dictionary
	'''
	h5filepath = GNU_SPEC_PATH + GTZAN_h5FILE

	if os.path.exists(h5filepath):
		print "let's just load the Gtzan STFTs"
		return;
	else:
		f_h5 = h5py.File(h5filepath, 'w')

	for folder in os.listdir(GTZAN_WAV_PATH):
		if not os.path.isfile(folder):
			genre = folder
			for filename in os.listdir(GTZAN_WAV_PATH + folder):
				#check if it already in the h5 file
				if not filename in f_h5.keys():
					src, sr = librosa.load(GTZAN_WAV_PATH + folder + '/' + filename)
					specgram = np.absolute(librosa.stft(src, n_fft=N_FFT, hop_length=N_HOP, win_length=N_WIN, window=TYPE_WIN)) # STFT of signal
					dset = f_h5.create_dataset(filename, data=specgram)
					dset.attrs['genre'] = genre
					print filename + ' is STFTed and stored.'


	f_h5.close()

def genreToClass():
	retDict = {'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4,
				'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9}
	return retDict

def saveData(train_x, train_y, test_x, test_y):
	cPickle.dump([train_x, train_y, test_x, test_y], (open(GNU_SPEC_PATH + GTZAN_TRAINING_TEST, 'wb')))

def loadData():
	data = cPickle.load(open(GNU_SPEC_PATH + GTZAN_TRAINING_TEST, 'rb'))
	return data[0], data[1], data[2], data[3]



if __name__ == '__main__':
	#get file list in the source folder	
	# prepare() #old one for the beginning.
	prepareGtzan()
	genreToClassDict = genreToClass()
	# load to train
	h5filepath = GNU_SPEC_PATH + GTZAN_h5FILE
	f_h5 = h5py.File(h5filepath, 'r')
	print 'STFTs are loaded'
	'''
	minNumFr = 99999
	for i in range(len(f_h5)):
		minNumFr = min(minNumFr, f_h5[f_h5.keys()[i]].shape[1])
	#Now I Know, it's 1290 for GTZAN
	'''
	#about model
	model = buildModel()
	#about optimisation
	batch_size = 64
	nb_classes = 10
	nb_epoch = 10
	#about training data loading
	minNumFr = 1290
	minNumFr = 10 #to reduce the datapoints, for temporary.
	lenFreq = 513 #length on frequency axis
	numGenre = 10
	numSongPerGenre = 100
	portionTraining = 0.8

	numIteration = 1
 
	#for iter_i in range(numIteration):
	
	numDataPoints = int(portionTraining * numSongPerGenre) * numGenre * minNumFr
	training_x = np.zeros((numDataPoints, 513))
	training_y = np.zeros((numDataPoints,1))	
	print '--- prepare data --- p.s. numDataPoints: ' + str(numDataPoints)
	for genre_i in range(numGenre):
		for song_i in range(int(portionTraining * numSongPerGenre)): # 0:80	
			ind = genre_i* numSongPerGenre + song_i # 100
			indToWrite = genre_i * int(portionTraining * numSongPerGenre) + song_i # 80
			genre = f_h5[f_h5.keys()[ind]].attrs['genre']
			specgram = f_h5[f_h5.keys()[ind]][:,0:minNumFr] # 513x1290
			#print 'genre_i:' + str(genre_i) + ', song_i:' + str(song_i) + ', so, ' +str(ind*minNumFr) + ' to ' + str((ind+1)*minNumFr) + ', out of ' + str(numDataPoints)
			training_x[indToWrite*minNumFr:(indToWrite+1)*minNumFr, : ] = np.transpose(specgram)
			training_y[indToWrite*minNumFr:(indToWrite+1)*minNumFr, : ] = np.ones((specgram.shape[1], 1)) * genreToClassDict[genre] # int, 0-9
	print '--- training data loaded ---'
	#after loading from all genre, let's make it appropriate for the model
	print '--- model fitting! ---'
	training_y = np_utils.to_categorical(training_y, nb_classes)
	model.fit(training_x, training_y, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2)		

	print '--- prepare test data  ---'
	numDataPoints = int((1-portionTraining) * numSongPerGenre) * numGenre * minNumFr
	test_x = np.zeros((numDataPoints, 513))
	test_y = np.zeros((numDataPoints,1))
	for genre_i in range(numGenre):		
		for song_i in range(int(portionTraining*numSongPerGenre), numSongPerGenre):
			ind = genre_i * 100 + song_i
			indToWrite = genre_i * int((1-portionTraining) * numSongPerGenre) + song_i # 20
			genre = f_h5[f_h5.keys()[ind]].attrs['genre']
			specgram = f_h5[f_h5.keys()[ind]][:,0:minNumFr] # 513x1290
			# specVector = np.reshape(specgram, (1, lenFreq*minNumFr))
			test_x[indToWrite*minNumFr:(indToWrite+1)*minNumFr, : ] = np.transpose(specgram)
			test_y[indToWrite*minNumFr:(indToWrite+1)*minNumFr, : ] = np.ones((specgram.shape[1], 1)) * genreToClassDict[genre] # int, 0-9
	print '--- test data loaded ---'
	print '--- prediction! for ' + genre + ' ---'
	test_y = np_utils.to_categorical(test_y, nb_classes)
	score = model.evaluate(test_x, test_y, show_accuracy=True, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])









