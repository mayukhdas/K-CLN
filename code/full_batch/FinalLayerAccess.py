# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:22:37 2018

@author: mayuk
"""
import numpy as np

global fprobs
fprobs = np.ones((19717,3)) * 0.0000000003

def init(datalen,classlen):
    global fprobs
    fprobs = np.ones((datalen,classlen))
    factor = (1/classlen) * 0.0000000001
    fprobs = fprobs * factor
    
def setFP(fp):
    global fprobs
    fprobs = fp