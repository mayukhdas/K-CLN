# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:26:45 2018

@author: mxd174630

Sample builder

"""

import gzip
import cPickle
import numpy
import random
import json

def save_sample(train_ids,valid_ids,test_ids, name, loc):
    finalloc = loc + name+'.json'
    data = {}
    
    train_ids_sample = random.sample(train_ids, int(len(train_ids)*1.0))
    valid_ids_sample = random.sample(valid_ids, int(len(valid_ids)*1.0))
    test_ids_sample = random.sample(test_ids, int(len(test_ids)*1.0))
    
    data['train'] = train_ids_sample
    data['valid'] = valid_ids_sample
    data['test'] = test_ids_sample
    with open(finalloc, 'w') as outfile:
        json.dump(dict(data), outfile)
    
    return train_ids_sample,valid_ids_sample,test_ids_sample,finalloc

def load_sample(filepath):
    with open(filepath) as handle:
        dictdump = json.loads(handle.read())
    train = dictdump['train']
    valid = dictdump['valid']
    test = dictdump['test']
    
    return train, valid, test

f = gzip.open('../data/' + 'debate' + '.pkl.gz', 'rb')
feats, labels, rel_list, train_ids, valid_ids, test_ids = cPickle.load(f)
save_sample(train_ids, valid_ids, test_ids, 'debateSample100P', '../data/')


    

