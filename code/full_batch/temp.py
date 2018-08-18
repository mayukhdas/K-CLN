# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
y = [3,4,81,0,9,32,6,0,0,0,1]
x = np.asarray(y)
#print("Original array: ")
#print(x)
#print("Their indices are ", [y[i] for i in np.nonzero(x)])
#print "Their indices are ", [y[i] for i in np.nonzero(x)[0]]

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
l.extend([0]*(19717-len(l)))
#print l

#l = np.array(l)
l = [i*3+l[i] for i in range(0, len(l))]

k = np.zeros((len(l),3))

np.put(k,l,1.0)


print k


with open("test.txt", "a") as myfile:
    myfile.write("epoch: "+str(k)+"\n")