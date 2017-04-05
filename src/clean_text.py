'''
Created on Apr 4, 2017

@author: Fanglin
'''

import nltk
import pandas as p
import pickle
import string
import re

def remove_punc(text):
    return text.translate({ord(c): None for c in string.punctuation})

def remove_num_sym(text):
    '''
    Only keep alphabetical characters and spaces
    '''
    return re.sub(r'[^A-Za-z ]', '', text)

def remove_stopword(text):
    tokens = nltk.word_tokenize(text)
    return [word for word in tokens if word not in nltk.corpus.stopwords.words("english")]

#Import reviewText
path = '../data/'
raw_data = p.read_csv(path + 'elec_sub.csv')
reviewText = raw_data['reviewText'].values

#Export clean_text
output = []
for text in reviewText:
    text = str(text)   #Ensure all texts are strings to avoid TypeError
    text1 = remove_num_sym(text)
    text2 = remove_stopword(text1)
    output.append(text2)
pickle.dump(output, open(path + "clean_text.p", "wb"))