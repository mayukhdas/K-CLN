# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:45:26 2018

@author: mayuk
"""

import gzip
import cPickle as pkl

f =gzip.open("../data/social.pkl.gz",'rb')
feats, labels, rel_list, train_ids, valid_ids, test_ids = pkl.load(f)


print [i for i,x in enumerate(labels) if x>0]

avg1 = 0.0
c1 = 0
avg2 = 0.0
c2 = 0
for x in rel_list:
    if len(x[0])>0:
        avg1 = avg1 + len(x[0])
        c1 = c1+1
    if len(x[1])>0:
        avg2 = avg2 + len(x[1])
        c2 = c2+1

print (avg1/float(c1)), (avg2/float(c2))

