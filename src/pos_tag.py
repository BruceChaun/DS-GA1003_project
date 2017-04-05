'''
Created on Apr 4, 2017

@author: Fanglin
'''

import nltk
import pandas as p
import pickle

def tag(text):
    '''
    @param text: a string of text
    @return tags: a list of tags
    '''
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    return tags

def count_tag(tags, tagname):
    '''
    @param tags: a list of tags
           tagname: a string of tag name, such as 'IN'
    @return count: the number of tags with the tag name, int
    '''
    count = 0
    for tag in tags:
        if tag[1] == tagname:
            count += 1
    return count

tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle').keys()   #Access all possible tag names in the Penn Treebank Tag Set
def tag_dis(tags):
    '''
    @param tags: a list of tags
    @return dis: the distribution of tags among all possible tag names, list
    '''
    dis = []
    for tagname in tagdict:   
        dis.append(count_tag(tags, tagname))   #Maybe there is a faster way to get the distribution
    return dis

#Import reviewText
path = '../data/'
raw_data = p.read_csv(path + 'elec_sub.csv')
reviewText = raw_data['reviewText'].values

#Export pos_tag
output = []
for text in reviewText:
    text = str(text)   #Ensure all texts are strings to avoid TypeError
    tags = tag(text)
    dis = tag_dis(tags)
    output.append(dis)   
df = p.DataFrame(output, columns = tagdict)
pickle.dump(df, open(path + "pos_tag.p", "wb"))
