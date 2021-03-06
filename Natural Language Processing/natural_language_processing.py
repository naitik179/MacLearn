# -*- coding: utf-8 -*-
"""Natural Language Processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YZhIbABg953V-LdcQO28_IdZT9rk3dln

## Importing the Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Quoting  == 3 ignores the double quotes"""

dataset = pd.read_csv('Restaurant_Reviews.tsv' , delimiter='\t' , quoting= 3)
dataset

"""## Cleaning the Text


re function used here will eliminate all the unnecessary words and will allow only the small and upper case Alphabets and then adding space between the words .

nltk contains the list of words called stopwords which are not uselful in classification of the words as positive or negative so we can eliminate those words directly
"""

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] # for storing all the cleaned reviews of our dataset
for i in range (0 ,1000):

  review = re.sub('[^a-zA-Z]' , ' ' , dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer() # creating object to apply stemming to each of the word in the review
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)


corpus

"""## Creating the Bag of words Model"""

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[: , 1].values

y
X

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)