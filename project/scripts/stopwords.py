"""Code for submiting"""

import numpy as np
import json
import re
import matplotlib.pyplot as plt
from pyspark.sql import *
from pyspark import SparkContext, SQLContext
from pyspark.ml.feature import *
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import functions as F
import pickle
import string


sc = SparkContext()
sqlContext = SQLContext(sc)
data = sc.textFile("/datasets/tweets-leon")

"""Select tweet by filtering"""

def selection_tweet(tweet):
    array = tweet.split("\t")
    if (len(array)==5):
    	if (array[0] == 'en' and array[2][-4:]=='2014'): 
        	return True
    return False

"""Remove punctuation, hashtags and urls"""
table = string.maketrans("","")
def punctuation(s):
    s = re.sub(r"@\S+", "", s)     #mention
    s = re.sub(r"http\S+", "", s)  #urls
    s = re.sub(r"#\S+", "", s)     #hashtag
    return s.translate(table, string.punctuation)

"encode tweet by mapping"
def encode_tweet(tweet):
    "Encode UTF-8"
    encoded = [t.encode("utf-8") for t in tweet.split("\t")]
    "Return ID and processed text"
    return Row(id=encoded[1], sentence=punctuation(encoded[4]).split(' '))


"Take a small dataset"
terms = data.filter(selection_tweet) \
            .map(encode_tweet)

"Transform to DF"
df_terms = sqlContext.createDataFrame(terms)  

"""Remove Stop-words"""
stopwords_rdd = sc.textFile('/user/khau/data/stopword.txt')
stopwords = stopwords_rdd.collect()
stopwords = stopwords[0].split(',')

remover = StopWordsRemover(inputCol="sentence", outputCol="filtered", stopWords=stopwords)
removed_stopwords = remover.transform(df_terms)

removed_stopwords.save('removed_stopwords_2014', mode='overwrite')