import json
from pyspark.sql import *
from pyspark import SparkContext, SQLContext
from pyspark.ml.feature import *
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import functions as F

sc = SparkContext()
sqlContext = SQLContext(sc)

"Read stopword"
#removed_stopwords = sqlContext.read.load('removed_stopwords_2014')
removed_stopwords = sqlContext.read.load('lemmatization_2014')

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

"Save the vocabulary (terms)"

voca = count_vectorizer_model.vocabulary
vocabulary = sc.parallelize(voca)
vocabulary_df = sqlContext.createDataFrame(vocabulary.map(Row))
vocabulary_df.write.json("vocabulary_2014.txt")

"IDF"

idf = IDF(inputCol="vectors", outputCol="tfidf")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)

try:
    tfidf = tfidf.drop('vectors')
except:
    pass

tfidf.save('tfidf_2014', mode='overwrite')