
# 29 May 2015, Keunwoo Choi, keunwoo.choi@qmul.ac.uk

from setup_environment import *

import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
#import h5py as hdf # to read/write data
import os

import cPickle 
import pdb

from constants import *
import numpy as np

import keras

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
	from keras.optimizers import SGD
	model = Sequential()
	model.add(Dense(500, 128, init='uniform')) #500: arbitrary number
	model.add(Activation('Tanh'))
	model.add(Dropout(0.5))

	model.add(Dense(128, 32, init='uniform'))
	model.add(Activation('Tanh'))
	model.add(Dropout(0.5))

	model.add(Dense(32, 2, init='uniform'))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss = 'mean_squared_error', optimizer=sgd)

def prepareGtzan(gtzan_path):
	'''
	input: path of gtzan dataset
	what it does: load the files, get stft, save it into the disk, and update h5 dictionary
	'''
	h5filepath = GNU_SPEC_PATH + GTZAN_h5FILE

	if os.path.exists(h5filepath):
		f_h5 = h5py.File(h5filepath, 'r+')
	else:
		f_h5 = h5py.File(h5filepath, 'w')

	for folder in os.listdir(GTZAN_WAV_PATH):
		if not os.path.isfile(folder):
			genre = folder
			for filename in os.listdir(GTZAN_WAV_PATH + folder):
				#check if it already in the h5 file
				if not filename in f_h5.keys()
					src, sr = librosa.load(GTZAN_WAV_PATH + folder + '/' + filename)
					specgram = librosa.stft(src, n_fft=N_FFT, hop_length=N_HOP, win_length=N_WIN, window=TYPE_WIN) # STFT of signal
					dset = f_h5.create_dataset(filename, specgram.shape, dtype='f')
					dset = specgram


	f_h5.close()









if __name__ == '__main__':
	#get file list in the source folder	
	# prepare() #old one for the beginning.
	prepareGtzan()
	pdg.set_trace()
	# load to train
	filenameList = loadSpectrogramList() # load file names 
	for filename in filenameList:
		specgramHere = loadSpectrogramFile(filename)
		# do something

	h5 = h5py.File(SID_SPEC_PATH) # read h5 dict
	f_text = open(TXT_PATH + TRAIN_FILE, "r")
	for line in f_text.readlines():
		line = line.split('\n')[0] #remove newline 
		specHere = h5[line]['x'] # 'x' is the key for stft 

	model = buildModel()

	#load data

	#fit model









