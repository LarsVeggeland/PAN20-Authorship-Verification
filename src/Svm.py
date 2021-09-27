#---------- Imported Libraries ----------

from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC

from collections import Counter

from nltk.tokenize import word_tokenize

import numpy as np

import time
    


#---------- Prepare Data ----------

def prep_data(X, Y, train_size : float = 0.9):
    #Splits the text pairs and truth values into seperate into training and test data
    return tts(X, Y, train_size=train_size)
    


#---------- Train Model ----------

def svm(X, Y, vec_space):
    X_train, X_test, Y_train, Y_test = prep_data(X, Y)
    
    X_train, X_test = np.array(X_train), np.array(X_test)

    svm = SVC(kernel='linear')
    svm.fit(X_train, Y_train)

    print("Training completed, evaluating model...")

    Y_pred = svm.predict((X_test))
    c = 0
    for i in range(len(Y_test)):
        c += Y_pred[i] == Y_test[i]

    print(f"Accuracy: {round(c/len(Y_test) * 100, 2)} %")
