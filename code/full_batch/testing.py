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

