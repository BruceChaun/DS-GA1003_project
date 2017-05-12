'''
Created on Apr 20, 2017
@author: Fanglin Chen
This module merges all features and splits the whole dataset into training, validation and test sets.
'''

import numpy as np
import pandas as p
import pickle
from sklearn.decomposition import PCA

path = '/Users/chenfanglin/fc1315/ml_project/DS-GA1003_project/'

### Convert pos_tag from count to percentage
pos_tag = pickle.load(open(path + 'pos_tag.p','rb'))
count_sum = pos_tag.sum(axis=1)
pos_tag = pos_tag.div(count_sum, axis=0)

### Import and merge data
elec_sub = p.read_csv(path + 'elec_sub.csv')
elec_sub_other = p.read_csv(path + 'elec_sub_other.csv')
spell = pickle.load(open(path + 'spell.p','rb'))
df_spell = p.DataFrame(spell, columns=['spell'])

data = p.merge(elec_sub, elec_sub_other, on=['userID', 'productID']) 
data = p.concat([data, pos_tag], axis=1)
data = p.concat([data, df_spell], axis=1)
print(data['reviewText'])

### Transform "gender" and "year" into numeric values
data['gender'].replace(to_replace=['female', 'male'], value=[1, 0], inplace=True)
data['year'] = data['year'].str[-4:].astype('int64')

### Export data
data = data[['gender','score','helpful','total','sub_cat1','pos' ,'neg' ,'len', 'voted','total_reviews_product' ,'total_reviews_user', 'review_sequence',
 'score_relative' ,'score_variance', 'year' ,'WC' ,'Analytic','Clout', 'Authentic' ,'Tone' ,'WPS' ,'Sixltr' ,'Dic' ,'function', 'pronoun',
 'ppron', 'i', 'we' ,'you' ,'shehe' ,'they' ,'ipron' ,'article', 'prep' ,'auxverb','adverb', 'conj' ,'negate' ,'verb', 'adj', 'compare' ,'interrog' ,'number',
 'quant' ,'affect' ,'posemo' ,'negemo', 'anx' ,'anger', 'sad' ,'social' ,'family','friend', 'female' ,'male' ,'focuspast' ,'focuspresent' ,'focusfuture','informal' ,
 'swear', 'netspeak' ,'assent', 'nonflu' ,'filler' ,'AllPunc','Period' ,'Comma', 'Colon', 'SemiC', 'QMark' ,'Exclam' ,'Dash', 'Quote', 'Apostro','Parenth' ,'OtherP' ,
 'score.x', 'sd' ,'av_sen', 'score.y', 'Flesch_Kincaid','Gunning_Fog_Index' ,'Coleman_Liau', 'SMOG' ,'Automated_Readability_Index','Average_Grade_Level']]
data.to_csv(path + 'merged.csv', index=False)

### Drop non-numeric values and missing values
merged = p.read_csv(path + 'merged.csv')
merged.drop(['sub_cat1'], axis=1, inplace = True)
merged.dropna(axis=1, how='any', inplace=True)

### PCA (done by Poppy)
pca = PCA(n_components=0.9, svd_solver='full', random_state=16)
df_result = pca.fit(merged).transform(merged)
print(df_result)
print(pca.explained_variance_ratio_) 

### Split data into train, validate and test
merged1 = p.read_csv(path + 'merged1.csv')
print(merged1.shape, merged1.columns.values)

train, validate, test = np.split(merged1.sample(frac=1, random_state=10), [int(.6*len(merged1)), int(.8*len(merged1))])

train.to_csv(path + 'train.csv', index=False)
validate.to_csv(path + 'validate.csv', index=False)
test.to_csv(path + 'test.csv', index=False)



