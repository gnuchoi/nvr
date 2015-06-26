# 04 June 2015, Keunwoo Choi
# main_lyrics.py; to analyse my lyric datasets.
import nltk
import os
import pdb

import numpy as np


from setup_environment import *

punctuationSet =  (',', "'", '"', ",", ';', ':', '.', '?', '!', '(', ')',
                   '{', '}', '/', '\\', '_', '|', '-', '@', '#', '*', '`')
import string
englishAlphabetSet = set(list(string.ascii_lowercase))
englishAlphabetSet.add(' ')



def makeSingleLine(lines):
	''' get multiple lines, remove new lines and  '''
	lyrics_flat = ''
	for eachline in lines:
		lyrics_flat = lyrics_flat + ' ' + eachline.lower().rstrip("\r\n")
	return lyrics_flat

def replaceWords(lyrics_flat):
	''' lyric: a single line string
	Assuming English lyric input,'''
	lyrics_flat = lyrics_flat.replace("'m ", " am ")
	lyrics_flat = lyrics_flat.replace("'re ", " are ")
	lyrics_flat = lyrics_flat.replace("'ve ", " have ")
	lyrics_flat = lyrics_flat.replace("'d ", " would ")
	lyrics_flat = lyrics_flat.replace("'ll ", " will ")
	lyrics_flat = lyrics_flat.replace(" he's ", " he is ")
	lyrics_flat = lyrics_flat.replace(" she's ", " she is ")
	lyrics_flat = lyrics_flat.replace(" it's ", " it is ")
	lyrics_flat = lyrics_flat.replace(" ain't ", " is not ")
	lyrics_flat = lyrics_flat.replace("n't ", " not ")
	lyrics_flat = lyrics_flat.replace("'s ", " ")
	return lyrics_flat

def removePunctuation(lyrics_flat):
    #for p in punctuationSet:
    #for p in englishAlphabetSet: #strict!
    characterSet = set(lyrics_flat)
    for ch in characterSet:
    	if ch not in englishAlphabetSet:
        	lyrics_flat = lyrics_flat.replace(ch, '')

    return lyrics_flat

def stemWords(lyrics_flat):
	ret = []
	from nltk.stem.snowball import SnowballStemmer
	stemmer = SnowballStemmer("english")
	for w in lyrics_flat:
		ret.append(stemmer.stem(w).encode('utf8', 'ignore')) # stemmer returns a unicode string, e.g. u'string'. Hence we make it back to ascii 

	return ret

class termDocMatrix(object):
	def __init__(self):
		'''
		initiate a term-document matrix.
		a column is a class! i.e. the matrix is looks like this:

				      \ 1 1 1 0 0 0 1 0 0 1 1 0 1 ...
				word1 | 3 0 0 0 2 0 0 1
				word2 | 0 0 2 0 0 2 0 1
				word3 | 0 2 0 0 0 0 0 1 ...

		Generally #word > #doc, so #row > #column

		'''

		self.wordlist = [] # ['hey', 'hear', 'sound' ...]
		self.wordindex = {} # {'hey':0, 'hear':1, 'sound':2, ...} word-to-index
		self.classlist = []# [1,1,1,1,1,1,0,0,0,0,1,0,...] 
		self.tdMtx = np.zeros((0,0))

	def update(self, newWordList, newClassLabel):
		'''
		update matrix data given new word list (which should be pre-processed first).
		 - first, add new column.
		 - second, fill the column.

		newWordList: a list, new words
		newClassLabel: a number, 0,1,2, ... indicating the class of document that newWordList comes from.

		'''
		#first, add a new blank column for new document
		self.tdMtx = np.concatenate(( self.tdMtx, np.zeros((self.tdMtx.shape[0] ,1))), 1) # add new column
		self.classlist.append(newClassLabel)
		#second, make a dictionary, {word: #occurence} given newWordList
		dictHere = {}
		for w in newWordList:
			if not w in dictHere:
				dictHere[w] = 1.0;
			else:
				dictHere[w] = dictHere[w] + 1.0;
		#third, post-processing of the dict. perhaps log(occurence) to compress.
		#this part can be modified by how much I want to compress (or not) it.
		for w in dictHere:
			dictHere[w] = 1.0 + np.log10(dictHere[w])

		#fourth, add it up!
		for w in dictHere:
			# add new word if necessary.
			if not w in self.wordlist:  
				self.wordlist.append(w) # add new word
				self.wordindex[w] = len(self.wordlist) - 1
				self.tdMtx = np.concatenate((self.tdMtx, np.zeros((1, self.tdMtx.shape[1]))), 0) # add new row
			# then word_frequency++
			self.tdMtx[self.wordindex[w], -1] = self.tdMtx[self.wordindex[w], -1] + dictHere[w]
		#that's it! good boy.
			
	def bulkUpdateFromFolder(self, folderpath, classLabel):
		'''
		update for many lyrics, but they should be the same class label!

		'''
		print 'Bulk update from folder: ' + folderpath + ', for class label: ' + str(classLabel)

		theFiles = os.listdir(folderpath)

		for fileHere in theFiles:
			
			if not '.txt' == os.path.splitext(fileHere)[1]:
				continue
			f_txt = open(folderpath + fileHere, 'r')
			lyricsHere = f_txt.readlines()
			lyricsHere = makeSingleLine(lyricsHere) # one long line
			lyricsHere = replaceWords(lyricsHere) # refine words
			lyricsHere = removePunctuation(lyricsHere) # remove punctuation
			tokenHere = nltk.word_tokenize(lyricsHere) # e.g. 200 words
			wordsHere = [w for w in tokenHere if not w in nltk.corpus.stopwords.words('english')] # e.g. 100 words
			wordsHere = stemWords(wordsHere) # e.g. stemming --> stem, things --> thing

			self.update(wordsHere, classLabel)
	def getLSA(self, numComponent=10):
		'''
		Do LSA and save it. 
		'''
		if len(self.wordlist) <= numComponent:
			print '#word(' + str(len(self.wordlist)) + ') <= #componrnt(' + str(numComponent) + '), so cannot do LSA.'
			return
		print "Let's LSA by PCA."
		from sklearn.decomposition import PCA
		pca = PCA(n_components = numComponent, whiten=True)

		self.LSA_tdMtx = np.transpose(pca.fit_transform(np.transpose(self.tdMtx)))







if __name__ == '__main__':

	springPath = LYRICS_PATH + 'spring/'
	top100Path = LYRICS_PATH + 'top100+sw/'
	freqMtx     = termDocMatrix() # init a word-frequency matrix object.
	freqMtx.bulkUpdateFromFolder(top100Path, 0) # non-spring: 0
	freqMtx.bulkUpdateFromFolder(springPath, 1) # spring: 1
	# SVD (Latent semantic analysis) to reduce dimension
	freqMtx.getLSA(numComponent = 10)
	pdb.set_trace()

	# prediction





