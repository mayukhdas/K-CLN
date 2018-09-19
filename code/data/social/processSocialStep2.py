# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 15:09:21 2018

@author: mayukh
"""
import csv
tokenize = lambda doc: doc.lower().split(" ")
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords
import json
import numpy as np
import cPickle as pkl
import random
import gzip
from difflib import SequenceMatcher


titles=[]

cols = []
data = []

    

with open('socialmedia-disaster-tweets-DFEshortened.csv', 'rU') as f:
    line = csv.reader(f)
    count =0
    for row in line:
        #print row[len(row)-1]
        if count ==0:
            titles.extend(row)
        elif row[-1] is '':
            continue
        else:
            cols.append(row)
            data.append(row[-3])
        count = count+1

print titles
print len(cols)
#print data

f = gzip.open("socialStep1.pkl.gz",'rb')
labels, rels, otherf = pkl.load(f)




#------------ genfeatures ---------------
l = []
for i in range(len(data)):
    k = ''.join([x for x in data[i] if ord(x)<128])
    k = k.encode("ascii", "ignore")
    l.append(k)
    
#print l

for i in range(len(l)):
    l[i] = l[i].translate(None, string.punctuation)
    l[i] = l[i].rstrip()
    #l[i] = ' '.join([word for word in l[i].split() if word not in (stopwords.words('english'))])


#print l
#print len(l)

all_documents = l

sklearn_tfidf = TfidfVectorizer(stop_words='english',norm='l2',min_df=0, use_idf=True, smooth_idf=False, max_features=500, sublinear_tf=True, tokenizer=tokenize)#max_features=500,
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
#l = tfidf(all_documents)
feature_names = sklearn_tfidf.get_feature_names()
doc = 0
feature_index = sklearn_representation[doc,:].nonzero()[1]
#tfidf_scores = zip(feature_index, [sklearn_representation[doc, x] for x in feature_index])
#print l
#for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
  #print w, s
#print len(feature_names)
#print sklearn_representation[doc,:]
feats = []
for i in range(len(l)):
    f = [0]*len(feature_names)
    f_idx = sklearn_representation[i,:].nonzero()[1]
    for x in f_idx:
        f[x] = sklearn_representation[i, x]
    feats.append(f)
#print len(feats[100])
print len(feats), len(otherf)

finalfeats =[]
#------- final feature vec building ---------------- 
for i in range(len(feats)):
    of = otherf[i]
    f = feats[i]
    of.extend(f)
    finalfeats.append(of)

print finalfeats[2000]

tot = range(0,len(labels))
print tot

train = random.sample(tot, int(len(tot)*0.6))
test = [x for x in tot if x not in train]
valid = random.sample(tot, int(len(train)*0.1))


finalfeats = np.array(finalfeats).reshape((len(finalfeats),len(finalfeats[0])))
labels = np.array(labels)
with open('social.pkl', 'wb') as f:
    pkl.dump((finalfeats,labels,rels,train,valid,test), f)

