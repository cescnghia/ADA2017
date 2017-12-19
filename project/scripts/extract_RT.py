import json
import re
from pyspark.sql import *
from pyspark import SparkContext, SQLContext

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

"""Extract the retweeted user from the tweet content"""
def extract_RT(tweet):
    encoded = [t.encode("utf-8") for t in tweet.split("\t")]
    return [(part[1:],1) for part in encoded[4].split() if (part.startswith('@') and part.endswith(":") and "RT" in encoded[4])]
data_2014 = data.filter(selection_tweet)

hashtags = data_2014.flatMap(extract_RT) \
               		.reduceByKey(lambda a,b : a+b) \
               		.sortBy(lambda wc: -wc[1])
        
hashtag_counts = sqlContext.createDataFrame(hashtags.map(lambda wc: Row(hashtag=wc[0], count=wc[1])))

hashtag_counts.write.json("hdfs:///user/djambazo/extract_ReT.txt")


