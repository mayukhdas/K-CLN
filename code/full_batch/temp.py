# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import math as m
import cPickle as pkl
y = [3,4,81,0,9,32,6,0,0,0,1]
x = np.asarray(y)
#print("Original array: ")
#print(x)
#print("Their indices are ", [y[i] for i in np.nonzero(x)])
#print "Their indices are ", [y[i] for i in np.nonzero(x)[0]]
nan = float('nan')
s = [[0.2,0.2,0.3],
     [0.1,0.3,0.4],
     [0.1,0.4,0.6],
     [0.2,0.5,0.7],
     [0.5,0.1,0.7],
     [0.4,0.5,0.9],
     [0.3,0.3,0.2],
     [0.2,0.1,0.0]
     ]

s = np.array(s).reshape((8,3))

l = [2,0,1,2,0,2,1,1]
#print l

#l = np.array(l)
l = [i*3+l[i] for i in range(0, len(l))]

#k = np.ones((len(l),3))
#k = np.multiply(k,nan)
#np.put(k,l,1.0)
#print l,s

pkl.dump((l,s), open("testPickle.pkl","wb"))

v1 = []
v2 = []

print v1, v2
with open("testPickle.pkl", "rb") as f:
    v1,v2 = pkl.load(f)

print v1 , v2


#print (True in np.isnan(k))