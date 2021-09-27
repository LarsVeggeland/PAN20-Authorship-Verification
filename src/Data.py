#---------- Imorted Libraries ----------

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams

import re

from collections import defaultdict, Counter

import datetime

import json

import timeit

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

#Not an imported library but the svm model
from Svm import svm



#---------- Variables ----------

#Words such as the, and, or, a, i.e, frequent words which are common for most texts.
sw : set  = set(stopwords.words("english"))

#Tries to convert a word into its dictionary form, i.e, its lemma
lemmatizer = WordNetLemmatizer()



#---------- Clean Data ----------

def remove_stopwords(words : list):
    return [w for w in words if w not in sw]


def normalize_data(data : list):
    for i in range(len(data)):
        #Simple step to normalize the data. Models are not aware that Hi and hi are the same word.
        data[i] = data[i].lower()

        #Reducing a word to its lemma
        data[i] = lemmatizer.lemmatize(data[i])

    return data


def remove_symbols(data : list):
    #A patter which will match any sequence of the included symbols
    pattern = re.compile(r"[\.,:;!?\"<>*-+`']+")

    for i in range(len(data)):
        #Removing symbol sequences matching the pattern
        data[i] = re.sub(pattern, r"", data[i])

    return data


def clean_data(data : list):
    cleaned_text_pair : list =[]
    for text in data:
        #Breaks the entire text into a list of words.
        words : list = word_tokenize(text)

        words = remove_symbols(words)
        words = normalize_data(words)
        words = remove_stopwords(words)
        cleaned_text_pair.append(" ".join(words))


    return cleaned_text_pair

       


#---------- Vectorize data ----------

def retrieve_vector_space_values(vector_space : list, vector_labels : list, vectorized_text_pair : np.array):

    vectors : list = [0]*len(vector_space)

    for vec in range(len(vector_space)):
        try:
            column_index : int = vector_labels.index(vector_space[vec])
            column : np.array = vectorized_text_pair[:, column_index]
            
            vectors[vec] = abs(column[0] - column[1])

        except ValueError as e:
            #The vector does not exist in the vectorized text pair, its value is already 0...
            pass
    return np.array(vectors)
    

#TF-IDF
def tf_ifd_transform(text_pair : list, analyzer : str, ngram_range : tuple):
    tf_idf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    t = tf_idf.fit_transform(text_pair)
    a = t.toarray()
    return a, tf_idf.get_feature_names()


#TF-IDF frequencies
def tf_idf_vectorizer(text_pairs : list, vector_space : list, analyzer : str = "word", n_gram_range : tuple = (1, 1)):
    vectorized_text_pairs : list = [None]*len(text_pairs)
    c = 1

    for i in range(len(text_pairs)):

        frequencies, features = tf_ifd_transform(text_pairs[i], analyzer=analyzer, ngram_range=n_gram_range)
        vectorized_text_pairs[i] = retrieve_vector_space_values(vector_space, features, frequencies)

        if (i == int((len(text_pairs)/10)*c)):
            print(f"{datetime.datetime.now()}: Vectorization at {c}0 %")
            c += 1

    print(f"{datetime.datetime.now()}: Vectorization at 100 %")
    return vectorized_text_pairs



#---------- Get data and related information ----------

def get_text_pairs(clean : bool, path : str, cutoff : int = None):
    id_and_texts : dict = {}

    with open(path, "r") as file:
        jsonlist = list(file)[:10]
        texts : list = [None]*len(jsonlist)
        for i in range(len(jsonlist)):
            entry = json.loads(jsonlist[i])
            text_pair = entry["pair"]

            if (clean):
                text_pair = clean_data(text_pair)

            if (cutoff is not None):
                #Simply using the first and last n=cutoff number of words for each document
                text1 = word_tokenize(text_pair[0])
                text2 = word_tokenize(text_pair[1])
                del(text1[cutoff:-cutoff])
                del(text2[cutoff:-cutoff])
                text_pair = [" ".join(text1), " ".join(text2)]
                

            
            texts[i] = text_pair
        return texts
    

def get_truth(path : str):
    #Boolean values are written with lower case...
    true = True
    false = False

    with open(path, "r") as file:
        jsonlist = list(file)[:10]
        truths : list = [None]*len(jsonlist)
        for i in range(len(jsonlist)):
            entry = json.loads(jsonlist[i])
            truths[i] = entry["same"]
        return truths
    



def get_training_data(shared_words : int = 0,
                    ordered_words : int = 0,
                    n_grams : tuple = None,
                    funcwords : bool = False,
                    largest_word_count : bool = False,
                    cleaning : bool=True,
                    cutoff : int = None):

    if (bool(shared_words) + bool(ordered_words) + bool(n_grams) > 1):
        print("ERROR! More than one dimension specified")
        raise Exception(print(f"shared_word : f{bool(shared_words)}\nordered_words : f{bool(ordered_words)}\nn_grams : f{bool(n_grams)}"))

    vectorized_data : list = []
    labels : list = []
    vector_space : list = []
    vectorized_text_pairs : list = None

    print(f"\n{datetime.datetime.now()}: Collecting data...\n")

    #Two lists contaning text pairs and whether both texts weree written by the same author
    text_pairs = get_text_pairs(cleaning, "D://Data//PAN 2020 Data//pan20-authorship-verification-training-small//pan20-authorship-verification-training-small.jsonl", cutoff)
    truths = get_truth("D://Data//PAN 2020 Data//pan20-authorship-verification-training-small//pan20-authorship-verification-training-small-truth.jsonl")
    
    print(f"{datetime.datetime.now()}: Data collected, extracting vector space...\n")

    if (shared_words):
        vector_space = get_words_n_times_in_all_texts(text_pairs, shared_words)

    elif (ordered_words):
        vector_space = get_n_most_frequent_words(text_pairs, ordered_words)
        
    elif (n_grams):
        vector_space = get_n_most_frequent_ngrams(text_pairs, n_grams[0], n_grams[1])

        #Requires a different TF-IDF vectorizer
        print(f"{datetime.datetime.now()}: vector space extracted, vectorizing data...\n")
        vectorized_text_pairs  = tf_idf_vectorizer(text_pairs, vector_space, analyzer="char", n_gram_range=(n_grams[0], n_grams[0]))

    elif (largest_word_count):
        vector_space = get_wordcount_from_most_verbose_text(text_pairs)
    
    elif (funcwords):
        vector_space = get_function_words()

    if vectorized_text_pairs is None:
        print(f"{datetime.datetime.now()}: vector space extracted, vectorizing data...\n")
        vectorized_text_pairs = tf_idf_vectorizer(text_pairs, vector_space)
    

    print(f"{datetime.datetime.now()}: data collection and vectorization completed")

    X, Y = np.array(vectorized_text_pairs), np.array(truths)
    #vectorized_text_pairs = vectorized_text_pairs.reshape(vectorized_text_pairs.shape[1:])

    print(len(X), len(Y))

    return X, Y, np.array(vector_space)


def get_function_words():
    func_words : list = []
    
    with open("Mosteller_Wallace_function_words.txt", 'r') as file:
        for word in file.readlines():
            #Removes trailing newline character
            func_words.append(word[:-1])

    return func_words
    

def get_n_most_frequent_words(text_pairs : list, n: int):
    counter = defaultdict(int)
    for i in range(len(text_pairs)):
        #A list containing all words in both texts
        words = word_tokenize(text_pairs[i][0]) + word_tokenize(text_pairs[i][1])
        for word in words:
            counter[word] += 1
    
    #Sorts words by their frequency
    ordered = sorted(counter.items(), key=lambda kv : kv[1], reverse=True)
    
    if (n > len(ordered)):
        print(f"\nWARNING! n exceeds total number of distinct words\n")
        return ordered

    return [entry[0] for entry in ordered][:n]


def get_words_n_times_in_all_texts(text_pairs : list, n : int = 1):
    words = set()

    for i in range(len(text_pairs)):
        #Getting all words and their frequncy for both texts
        text1_words = word_tokenize(text_pairs[i][0])
        text2_words = word_tokenize(text_pairs[i][1])
        text1_word_count = Counter(text1_words)
        text2_word_count = Counter(text2_words)

        #Removing all words whose frequency is less than n
        words_more_than_n_text1 = [kv[0] for kv in text1_word_count.items() if kv[1] >= n]
        words_more_than_n_text2 = [kv[0] for kv in text2_word_count.items() if kv[1] >= n]

        #Retrieving words with satisfactory frequency in both texts
        shared_words = set(words_more_than_n_text1).intersection(set(words_more_than_n_text2))

        if len(words) == 0:
            words = shared_words
            
        else:
            #Updating the set of words satisfying the condition
            words = words.intersection(shared_words)

    
    return list(words)


def get_n_most_frequent_ngrams(text_pairs : list, gram_size : int, n : int):
    #Moshe Koopel and Yaron Winter
    n_grams : dict = defaultdict(int)

    #Matches all sequences of whitecase characters including \n, \t, \r, ...
    pattern = re.compile(r"\s+")

    for i in range(len(text_pairs)):
        #Removing whitespaces on the data itself as they are not deemed important
        text1 = re.sub(pattern, r"", text_pairs[i][0])
        text2 = re.sub(pattern, r"", text_pairs[i][1])
        text_pairs[i] = [text1, text2]


        for text in text_pairs[i]:
            for gram in ngrams(text, gram_size):
                n_grams["".join(gram)] += 1

    #Sorts grams by their frequency in descending order
    ordered_grams = sorted([(gram, frequency) for gram, frequency in n_grams.items()], key=lambda kv : kv[1], reverse=True)[:n]

    #Returns the grams in sorted order
    return [x[0] for x in ordered_grams]




if __name__ == '__main__':

    #Uncomment and specify feature set in get training data
    x, y, z = get_training_data(shared_words = 5000)
    svm(x, y, z)

   
