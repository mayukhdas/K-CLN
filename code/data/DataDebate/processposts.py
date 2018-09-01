# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 03:03:16 2018

@author: mayuk

parse posts
"""
tokenize = lambda doc: doc.lower().split(" ")

from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords
import json
import unicodedata
import cPickle as pkl


with open('test.csv', 'r') as f:
    l = f.readlines()
print len(l)


#from pprint import pprint

with open('title.json') as fl:
    data = json.loads(fl.read().decode("utf-8",'ignore'))

#print data['rows']

l = []
for i in range(len(data['rows'])):
    k = ''.join([x for x in data['rows'][i]])
    k = k.encode('ascii','ignore')
    l.append(k)
    
print l

for i in range(len(l)):
    l[i] = l[i].translate(None, string.punctuation)
    l[i] = l[i].rstrip()
    #l[i] = ' '.join([word for word in l[i].split() if word not in (stopwords.words('english'))])


#print l
print len(l)

all_documents = l

sklearn_tfidf = TfidfVectorizer(stop_words='english',norm='l2',min_df=0, use_idf=True, smooth_idf=False,  sublinear_tf=True, tokenizer=tokenize)#max_features=500,
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
#l = tfidf(all_documents)
feature_names = sklearn_tfidf.get_feature_names()
doc = 0
feature_index = sklearn_representation[doc,:].nonzero()[1]
#tfidf_scores = zip(feature_index, [sklearn_representation[doc, x] for x in feature_index])
#print l
#for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
  #print w, s
print len(feature_names)
#print sklearn_representation[doc,:]

feats = []
for i in range(len(l)):
    f = [0]*len(feature_names)
    f_idx = sklearn_representation[i,:].nonzero()[1]
    for x in f_idx:
        f[x] = sklearn_representation[i, x]
    feats.append(f)
print feats[100]

with open('discTitlesDebate.pkl', 'wb') as f:
    pkl.dump(feats, f)