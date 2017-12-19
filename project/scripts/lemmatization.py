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
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet


sc = SparkContext()
sqlContext = SQLContext(sc)

"A function map all type of Noun to  wordnet.NOUN"
"E.x: Map ['NN', 'NNS', 'NNP', 'NNPS'] to wordnet.NOUN"
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

"A function lemmatizer a word with its POS"

lemmatiser = WordNetLemmatizer()
def lemmatization(row):
    sentence = row.filtered
    id_ = row.id
    if (len(sentence) == 0): # Nothing to lemmatiser
        return Row(id=id_, lemmatized=[])
    else :
        tokens_pos = pos_tag(sentence)
        tokens_pos = [(w, get_wordnet_pos(p)) for (w,p) in tokens_pos]
        return Row(id=id_, lemmatized=[lemmatiser.lemmatize(w, pos=p) for (w,p) in tokens_pos if p != None and all(ord(char) < 128 for char in w)])


"Read stopwords dataframe"
removed_stopwords = sqlContext.read.load('removed_stopwords_2014')

"Drop useless column"
try:
    removed_stopwords = removed_stopwords.drop('sentence')
except:
    pass

"Transform in RDD for mapping"
rdd_stopwords = removed_stopwords.rdd

"Do lemmatization"
rdd_lemma = rdd_stopwords.map(lemmatization)

df_lemma = sqlContext.createDataFrame(rdd_lemma)


"Re-transform into DF for storing"			 

df_lemma.save('lemmatization_2014', mode='overwrite')
