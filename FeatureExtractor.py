"""
This file implements 4 basic feature extraction functions as defined in 
"Twitter Sentiment Classification using Distant Supervision [2017]"
These are:
1. Unigram Features
2. Bigram Features
3. Unigram + Bigram Features
4. Unigram + POS Tag Features
"""

#All imports
import re
import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#####################################################
################# Utility Functions #################
#####################################################

def replaceUserName(tweet):
	tweet = re.sub('@[^\s]+','USERNAME',tweet)
	return tweet

def replaceLink(tweet):
	tweet = re.sub('((www\.[^\s]+)|((https|http)?://[^\s]+))','URL',tweet)
	return tweet

def replaceWithTwoChar(tweet):
    pattern = re.compile(r"(.)\1{1,}") 
    tweet = pattern.sub(r"\1\1", tweet)
    return tweet

def cleanTweet(tweet):
	tweet = replaceUserName(tweet)
	tweet = replaceLink(tweet)
	tweet = replaceWithTwoChar(tweet)
	return tweet

wordDict = dict()
wordList = []
trainTweets = []
stopWords = list(set(stopwords.words("english")))
stopWords = [str(word) for word in stopWords]

def computeWordDict():
	trainingTweets = csv.reader(open("training_dataset.csv", 'rt'))
	global wordList
	global wordDict
	global trainTweets
	global stopWords

	porterStemmer = PorterStemmer()

	print "Computing Word Dict and Word List!"
	for tweet in trainingTweets:
		tweet[1].lower()
		#Doing all optimizations as mentioned in the paper
		processedTweet = cleanTweet(tweet[1])
		wordsInTweet = processedTweet.split()
		
		#Removing StopWords to reduce no. of features
		text = ' '.join([word for word in wordsInTweet if word not in stopWords])
		wordsInTweet =  text.split()
		
		#Stemming the words to remove no. of features
		wordsInTweet1 = []
		for word in wordsInTweet:
			if word.isalpha():
				wordsInTweet1.append(str(porterStemmer.stem(word)))
			else:
				wordsInTweet1.append(word)
		wordsInTweet = wordsInTweet1

		trainTweets.append((wordsInTweet,tweet[0]))

		for each in wordsInTweet:
			wordDict[each] = wordDict.get(each,0) + 1
	print "Computing Word Dict and Word List Done!"
	wordList = list(wordDict.keys())

def extractUnigramFeatures(tweet):
	features = dict()
	wordsInTweet = set(tweet)

	for eachWord in wordList:
		if eachWord in wordsInTweet:
			features[eachWord] = 1
		else:
			features[eachWord] = 0
	
	return features
#computeWordDict()