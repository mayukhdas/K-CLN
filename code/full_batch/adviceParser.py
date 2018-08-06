# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:37:50 2018

@author: mayukh
"""

# -*- coding: utf-8 -*-

#import cPickle
import gzip
import csv
import numpy as np
from adviceFile import adviceSet

path = '../data/' + "pubmed" + '.pkl.gz'

f = gzip.open(path, 'rb')

# unpickler = cPickle.Unpickler(f)
# print(unpickler.load()[0])
entity_mul=[]
label=[]
advicelist=[]
word_list1=["urinari","insulin","kidney","genotyp","heart","glucose","insulindepend"]
word_list2=["obes","fat","genotyp","heart,glucose"]
word_list0=["genotyp","heart","glucose"]
raw_rels = np.zeros((100,100))

entity_list = []

def adviceFileReader(file):
    lines = [line.rstrip('\n') for line in open(file)]
    return lines

def processLiteral(lit):
    name = str.split(lit,"(")[0]
    terms = str.split(str.replace(str.split(lit,"(")[1],")",""),",")
    return name,terms
    

def parseEntities(f):
    with open(f) as tsv:
        lineC = 0
        for line in csv.reader(tsv, dialect="excel-tab"):
            lineC = lineC+1
            if "NODE" in line or lineC == 2:
                continue
            #print(lineC,"    ",line[0])
            entity_list.append(line[0])
                
           

def parseRel():
    with open("Pubmed-Diabetes.DIRECTED.cites.tab") as tsv:
        count = 0
        for line in csv.reader(tsv, dialect="excel-tab"):
            if ("DIRECTED" in line) or ("NO_FEATURES" in line):
                continue
            print(line)
            for i in range(len(line)):
                if "|" in line[i]:
                    source = str.split(line[i-1],":")[1]
                    sink = str.split(line[i+1],":")[1]
                    sourceidx = entity_list.index(source)
                    sinkidx = entity_list.index(sink)
                    raw_rels[sourceidx, sinkidx] = 1
            count = count+1
        print(count)
# ------------------------------------------------------------------------------------------------------
# Declaring Masks for ADVICE
# ------------------------------------------------------------------------------------------------------
advice_entity_mask = []
advice_entity_label = []
advice_relation_mask = []
    
def parseAdvice(ent,adviceSet):
    for adv in adviceSet:
        
        isAdvGrounded = True
        
        head = adv['h']
        body = adv['b']
        
        targetEnt = '';
        
        print("head")
        pref = head[0]
        if len(head)>1:
            npref = head[1]
        targetEnt = npref[1]
        preflabel = npref[2]
        
        if(targetEnt.startswith("?")):
            isAdvGrounded = None
        
        match = None
        with open(ent) as tsv:
            lineC = 0
            for line in csv.reader(tsv, dialect="excel-tab"):
                lineC = lineC+1
                if "NODE" in line or lineC == 2:
                    continue
                if line[1] == preflabel:
                    match = True
                    
                entitiesInQuestion = {}
                for p in body:
                    if(p[0]=="hasWord"):
                        if(p[1]==targetEnt):
                            if p[2] in line:
                                advice_entity_mask[entity_list.index(line[0])]=1
                            else:
                                advice_entity_mask[entity_list.index(line[0])]=0
                        else:
                            entitiesInQuestion[p[1]] = 0
                    else:
                        if(p[1]==targetEnt):
                            
                         
                        
        print("body")
        
            

print("Entities")
parseEntities("Pubmed-Diabetes.NODE.paper.tab")
raw_rels = np.zeros((len(entity_list),len(entity_list)))
parseRel()
print(np.count_nonzero(raw_rels))
# =============================================================================
# with open("Pubmed-Diabetes.NODE.paper.tab") as tsv:
#     for line in csv.reader(tsv, dialect="excel-tab"):
#         for i in range(len(line)):
#             for word in word_list1:
#                 count = 0
#                 count0=0
#                 if word in line[i]:
#                     count += 1
#             for word1 in word_list2:
#                 if word1 in line[i]:
#                     count+=1
#             if 'label=1' in line[i]:
#                 for word0 in word_list0:
#                     if word0 in line:
#                         count0+=1
#                 if count0 >=1:
#                     label.append(1)
#                 else:
#                     label.append(0)
#             if 'label=2' in line[i]:
#                 label.append(1)
#             if 'label=3' in line[i]:
#                 label.append(2)
#         entity_mul.append(count)
# 
# entity_mul_new=entity_mul[2:]
# for i in range(len(entity_mul_new)):
#     if entity_mul_new[i]==0:
#         entity_mul_new[i]=1
#     if entity_mul_new[i]==1:
#         entity_mul_new[i]=1.5
# =============================================================================
#print len(entity_mul_new),len(label)

# type 1: urinari,insulin, kidney,genotyp,heart,glucose,insulindepend
# type 2: obes, fat,genotyp,heart,glucose
# type 0: genotyp,heart,glucose: change label to 1 or 2

# feats, labels, rel_list, train_ids, valid_ids, test_ids = cPickle.load(f)
#
# print(labels[-1])