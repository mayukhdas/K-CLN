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
word_list2=["obes","fat","genotyp","heart","glucose"]
word_list0=["genotyp","heart","glucose"]
#raw_rels = np.zeros((100,100))

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
            #print(lineC,"    ",line[len(line)-2])
            entity_list.append(line[0])
                
           

# =============================================================================
# def parseRel(relFile):
#     raw_rels = np.zeros((len(entity_list), len(entity_list)))
#     with open(relFile) as tsv:
#         count = 0
#         for line in csv.reader(tsv, dialect="excel-tab"):
#             if ("DIRECTED" in line) or ("NO_FEATURES" in line):
#                 continue
#             #print(line)
#             for i in range(len(line)):
#                 if "|" in line[i]:
#                     source = str.split(line[i-1],":")[1]
#                     sink = str.split(line[i+1],":")[1]
#                     sourceidx = entity_list.index(source)
#                     sinkidx = entity_list.index(sink)
#                     raw_rels[sourceidx, sinkidx] = 1
#             count = count+1
#         print(count)
# =============================================================================
# ------------------------------------------------------------------------------------------------------
# Declaring Masks for ADVICE
# ------------------------------------------------------------------------------------------------------
advice_entity_mask = []
advice_entity_label = []
advice_relation_mask = []
    
def parseAdvice(ent,advice,feats,labels,rel_list):
    advice_entity_mask = np.zeros(len(labels))
    advice_entity_label = np.zeros(len(labels))
    advice_relation_mask = np.zeros((np.array(rel_list)).shape)
    for adv in advice:
        
        isAdvGrounded = True
        
        head = adv['h']
        body = adv['b']
        
        targetEntGiven = '';
        
        print("head")
        if len(head)>0:
            npref = head[0]
        targetEntGiven = npref[1]
        preflabel = npref[2]
        
        if(targetEntGiven.startswith("?")):
            isAdvGrounded = None
        
        match = None
                   
        Target_entities = []
        if isAdvGrounded is True:
            Target_entities.append(targetEntGiven)
        else:
            Target_entities.extend(entity_list)
            
        for index, targetEnt in enumerate(Target_entities):
            entitiesInQuestion = {}
            entitiesInQuestionCon = {}
            if labels[index] == preflabel:
                advice_entity_label[index] = 1
            else:
                advice_entity_label[index] = 0
            for p in body:
                if(p[0]=="hasWord"):
                    if(p[1]==targetEnt) or (p[1]==targetEntGiven):
                        if isAdvGrounded is not None:
                            if hasWordinEntity(ent,p[2],targetEnt):
                                advice_entity_mask[entity_list.index(targetEnt)] = 1
                        else:
                            if hasWordinEntity(ent,p[2],entity_list):
                                advice_entity_mask[:] = 1
                    else:
                        entitiesInQuestion[p[1]] = p[2]
                        entitiesInQuestionCon[p[1]] = None
                else:
                    if targetEnt in p:
                        entityinQ = None
                        for i in range(1,3):
                            if p[i] in entitiesInQuestionCon.keys:
                                entitiesInQuestionCon[p[i]] = True
                                
                            
            if len(entitiesInQuestion) > 0:
                for k in entitiesInQuestion:
                    if not str.startswith(k,"?"):
                        if (hasWordinEntity(ent, entitiesInQuestion[k], k) is True) and (entitiesInQuestionCon[k] is True) and ((entity_list.index(k) in rel_list[entity_list.index(targetEnt)][0]) or (entity_list.index(targetEnt) in rel_list[entity_list.index(k)][0])):
                            rel = entity_list.index(targetEnt)
                            idx = rel_list[rel,0].index(k)
                            advice_relation_mask[rel,0,idx] = 1
                    else:
                        nbrsId = getNeighborList(targetEnt,rel_list)
                        nbrs= [entity_list[i] for i in nbrsId]
                        for n in nbrs:
                            if hasWordinEntity(ent,entitiesInQuestion[k],n):
                                rel = entity_list.index(targetEnt)
                                idx = rel_list[rel,0].index(n)
                                advice_relation_mask[rel,0,idx] = 1
                
                                
def getNeighborList(entity, rel_list):
    print(entity)
    if entity not in entity_list:
        raise Exception("index out of bound")
    elif entity_list.index(entity) > len(rel_list):
        raise Exception("entity not in raw_rel")
    else:
        newL = rel_list[entity_list.index(entity)][0]
        ret = np.asarray(newL)
    return [newL[i] for i in np.nonzero(ret)[0]]

def hasWordinEntity(nodefile,word,entity):
    ret = None
    with open(nodefile) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if (line[0] in entity):
                if word in line:
                    ret = True
    return ret


def getAdvice(nodeFile,relFile,feats,labels,rel_list):
    parseEntities(nodeFile)
    parseRel(relFile)
    parseAdvice(nodeFile,adviceSet,feats,labels,rel_list)
    return advice_entity_label, advice_entity_mask, advice_relation_mask


#raw_rels = np.zeros((len(entity_list),len(entity_list)))
#parseRel()
#print(np.count_nonzero(raw_rels))
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