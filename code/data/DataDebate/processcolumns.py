# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 02:07:38 2018

@author: mayuk
"""
import csv
import numpy as np
import cPickle as pkl
import random

labels=[]
twocols = []
rels = []
with open('columns.csv','r') as f:
    csvR = csv.reader(f, delimiter=',')
    for row in csvR:
        labels.append(int(row[-1]))
        twocols.append([int(row[-3]),int(row[-2])])
    
#print twocols

train=[]
valid=[]
train  = random.sample(range(0,len(labels)),int(len(labels)*0.6))
for i in range(0, len(labels)):
    if i not in train:
        valid.append(i)

test = random.sample(valid,int(len(valid)*0.8))
valid = [x for x in valid if x not in test]

print len(train) + len(valid) + len(test)

print valid

for i, row in enumerate(twocols):
    thisDisc = int(row[0])
    thisAuth = int(row[1])
    sameDisc = []
    sameAuth = []
    for j, row1 in enumerate(twocols):
        currDisc = int(row1[0])
        currAuth = int(row1[1])
        if i != j and thisDisc == currDisc:
            sameDisc.append(j)
        if i != j and thisAuth == currAuth:
            sameAuth.append(j)
    rels.append([sameAuth])

#print rels[1000]
    


r = rels


#for i, row in enumerate(feats):
    #feats[i] = row.extend(titles[i

with open('relsAndLabelsAndSample.pkl', 'wb') as f:
    pkl.dump((r,labels,train,valid,test), f)