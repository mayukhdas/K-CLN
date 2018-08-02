# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:37:50 2018

@author: mayukh
"""

# -*- coding: utf-8 -*-

#import cPickle
import gzip
import csv
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
raw_rels = {}

def adviceFileReader(file):
    lines = [line.rstrip('\n') for line in open(file)]
    return lines

def processLiteral(lit):
    name = str.split(lit,"(")[0]
    terms = str.split(str.replace(str.split(lit,"(")[1],")",""),",")
    return name,terms
    

def parseRel():
    with open("Pubmed-Diabetes.DIRECTED.cites.tab") as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if ("DIRECTED" in line) or ("NO_FEATURES" in line):
                continue
            for i in range(len(line)):
                #print line[i]
                if "|" in line[i]:
                    if raw_rels.has_key(line[i-1]):
                        raw_rels[line[i-1]].append(line[i+1])
                    else:
                        raw_rels.setdefault(line[i-1], [])
                        raw_rels[line[i-1]].append(line[i+1])
                    break
                    
            #print "------------"
        #print raw_rels

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
    
    if(targetEnt.startswith("?")):
        isAdvGrounded = None
    
    
    print("body")
    for p in body:
        print(p)
    
            
        
        



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