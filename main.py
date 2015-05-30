
# 29 May 2015, Keunwoo Choi, keunwoo.choi@qmul.ac.uk

from setup_environment import *

import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
#import h5py as hdf # to read/write data
import cPickle

from constants import *
import numpy as np

def prepare():
	import os
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


if __name__ == '__main__':
	#get file list in the source folder	
	prepare()
	# load 



