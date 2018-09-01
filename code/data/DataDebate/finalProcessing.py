# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:07:16 2018

@author: mayuk
"""

import cPickle as pkl
import numpy as np

with open('featuresDebate.pkl','rb') as f:
    feats = pkl.load(f)
with open('discTitlesDebate.pkl', 'rb') as f1:
    titles = pkl.load(f1)
    
print len(titles[0])


finalfeats = []
for i, row in enumerate(feats):
    x = feats[i]
    x.extend(titles[i])
    finalfeats.append(x)

print len(finalfeats[100])


with open('relsAndLabelsAndSample.pkl', 'rb') as f2:
    r,labels,train,valid,test = pkl.load(f2)

finalfeats = np.array(finalfeats).reshape((len(finalfeats),len(finalfeats[0])))
labels = np.array(labels).reshape((len(labels),1))
with open('debate.pkl', 'wb') as f3:
    pkl.dump((finalfeats,labels,r,train,valid,test),f3)