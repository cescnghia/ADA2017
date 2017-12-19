
from pyspark.sql import *
from pyspark import SparkContext, SQLContext
from pyspark.ml.feature import *
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import functions as F
import pickle
import string

sc = SparkContext()
sqlContext = SQLContext(sc)

tfidf = sqlContext.read.load('tfidf_2014')

nbTopics= 20
maxIterations = 10
optimizer = 'online'

corpus = small_tfidf.select(F.col('id').cast("long"), 'tfidf').rdd.map(lambda x: [x[0], x[1]])
ldaModel = LDA.train(rdd=corpus, k=nbTopics, maxIterations=maxIterations, optimizer=optimizer)

df_topics = sqlContext.createDataFrame(ldaModel.describeTopics(10), ['terms','scores'])
df_topics.save('topics', mode='overwrite')