import numpy as np
import pandas as pd

dataset = pd.read_csv("C:\\Users\\LAPTOPS HUB\\Desktop\\Segment analysis\\a2_RestaurantReviews_FreshDump.tsv", delimiter = "\t", quoting = 3)
print(dataset.head())

import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus=[]

for i in range(0, 100):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
import pickle
cvFile='c1_BoW_Sentiment_Model.pkl'
cv = pickle.load(open(cvFile, "rb"))

X_fresh = cv.transform(corpus).toarray()
X_fresh.shape

import joblib
classifier = joblib.load('c2_Classifier_Sentiment_Model')

y_pred = classifier.predict(X_fresh)
print(y_pred)

dataset['predicted_label'] = y_pred.tolist()
dataset.head()

dataset.to_csv("c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)