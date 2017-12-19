import json
from pyspark.sql import *
from pyspark import SparkContext, SQLContext
from pyspark.ml.feature import *
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import functions as F
import json

sc = SparkContext()
sqlContext = SQLContext(sc)

removed_stopwords = sqlContext.read.load('removed_stopwords_2014')
#removed_stopwords = sqlContext.read.load('lemmatization_2014')
try:
    removed_stopwords = removed_stopwords.drop('sentence')
except:
    pass

"TF"
"Parameter minDF: this term have to be appear in a specific nb of docs (1000 here)"
cv = CountVectorizer(inputCol="filtered", outputCol="vectors", minDF=1000.0)
count_vectorizer_model = cv.fit(removed_stopwords)
tf = count_vectorizer_model.transform(removed_stopwords)

try:
    tf = tf.drop('filtered')
except:
    pass

print('################################# FINISH TF STAGE #############################################################################')
#voca = count_vectorizer_model.vocabulary
#vocabulary = sc.parallelize(voca)
#vocabulary_df = sqlContext.createDataFrame(vocabulary.map(Row))
#vocabulary_df.write.json("vocabulary_2014.txt")

"IDF"
new_tf = tf.randomSplit([1.0, 17999.0], seed=2014)[0]

idf = IDF(inputCol="vectors", outputCol="tfidf")
idfModel = idf.fit(new_tf)
tfidf = idfModel.transform(new_tf)

try:
    tfidf = tfidf.drop('vectors')
except:
    pass

tfidf.show()

print('################################# FINISH TF-IDF STAGE #############################################################################')

nbTopics= 20
maxIterations = 10
optimizer = 'online'

corpus = tfidf.select(F.col('id').cast("long"), 'tfidf').rdd.map(lambda x: [x[0], x[1]])


ldaModel = LDA.train(rdd=corpus, k=nbTopics, maxIterations=maxIterations, optimizer=optimizer)

print('################################# FINISH LDA STAGE #############################################################################')

df_topics = sqlContext.createDataFrame(ldaModel.describeTopics(10), ['terms','scores'])
df_topics.save('topics_99k', mode='overwrite')