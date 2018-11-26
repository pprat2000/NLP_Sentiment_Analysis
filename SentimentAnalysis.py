"""
This file implements 3 Machine Learning  as defined in 
"Twitter Sentiment Classification using Distant Supervision [2017]"
These are:
1. Naive Bayes
2. Maximum Entropy
3. SVM
"""

#All imports
import csv
from nltk import classify
from nltk import NaiveBayesClassifier
import FeatureExtractor as fm

if __name__ == '__main__':
	
	fm.computeWordDict()

	print len(fm.wordList)

	print "Extracting Unigram Features!"
	trainingData = classify.apply_features(fm.extractUnigramFeatures, fm.trainTweets)
	print "Extracting Unigram Features Done!"

	print "Training Started!"
	naiveBayes = NaiveBayesClassifier.train(trainingData)
	print "Training Done!"

	testTweet = csv.reader(open("sampleTweets.csv","rt"))

	i = 0
	c = 0
	for tweet in testTweet:
		
		tweet[0] = tweet[0][1:-1]
		tweet[1] = tweet[1][1:-1]
		tweet[1] = fm.cleanTweet(tweet[1])
		
		output = naiveBayes.classify(fm.extractUnigramFeatures(tweet[1].split()))
		
		if output == tweet[0]:
			i += 1
		c += 1
	print float(i*1.0/c)
	print naiveBayes.show_most_informative_features(5)