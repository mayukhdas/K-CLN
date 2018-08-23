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

# unpickler = cPickle.Unpickler(f)
# print(unpickler.load()[0])
advicelist=[]
word_list1=["urinari","insulin","kidney","genotyp","heart","glucose","insulindepend"]
word_list2=["obes","fat","genotyp","heart","glucose"]
word_list0=["genotyp","heart","glucose"]
#raw_rels = np.zeros((100,100))

entity_list = []
entity_whole_list = []


def adviceFileReader(file):
    lines = [line.rstrip('\n') for line in open(file)]
    return lines

def processLiteral(lit):
    name = str.split(lit,"(")[0]
    terms = str.split(str.replace(str.split(lit,"(")[1],")",""),",")
    return name,terms
    

def parseEntities(f):
    entityl = []
    ewl = []
    with open(f) as tsv:
        lineC = 0
        for line in csv.reader(tsv, dialect="excel-tab"):
            lineC = lineC+1
            if "NODE" in line or lineC == 2:
                continue
            #print(line[0],"    ",line[len(line)-1])
            entityl.append(line[0])
            ewl.append([line[0],line[len(line)-1]])
    return entityl,ewl
                
           

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
    
def parseAdvice(ent,advice,feats,labels,rel_list,train):
    global advice_entity_mask
    global advice_entity_label
    global advice_relation_mask
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
       
        #print train
        Target_entities = []
        if isAdvGrounded is True:
            Target_entities.append(targetEntGiven)
        else:
            #print entity_list[train[0]]
            Target_entities.extend([entity_list[i] for i in train])
            #Target_entities.extend([entity_list[i] for i in range(len(labels))])
        #print "list of targets", entity_list[train[10]], Target_entities
            
        for index, targetEnt in enumerate(Target_entities):
            #print(index, " / ", len(Target_entities))
            entitiesInQuestion = {}
            entitiesInQuestionCon = {}
            index = entity_list.index(targetEnt)
            #print labels[index], preflabel
            if str(labels[index]) == str(preflabel):
                advice_entity_label[index] = int(preflabel)
            else:
                advice_entity_label[index] = 0
            for p in body:
                if(p[0]=="hasWord"):
                    if(p[1]==targetEnt) or (p[1]==targetEntGiven):
                        #if isAdvGrounded is not None:
                        #print "target", targetEnt
                        if hasWordinEntity(ent,p[2],targetEnt):
                            advice_entity_mask[entity_list.index(targetEnt)] = 1
                    else:
                        entitiesInQuestion[p[1]] = p[2]
                        entitiesInQuestionCon[p[1]] = None
                else:
                    if targetEnt in p:
                        for i in range(1,3):
                            if p[i] in entitiesInQuestionCon.keys:
                                entitiesInQuestionCon[p[i]] = True
                                
            #print entitiesInQuestion          
            if len(entitiesInQuestion) > 0:
                for k in entitiesInQuestion:
                    if not str.startswith(k,"?"):
                        if (hasWordinEntity(ent, entitiesInQuestion[k], k) is True) and (entitiesInQuestionCon[k] is True) and ((entity_list.index(k) in rel_list[entity_list.index(targetEnt)][0]) or (entity_list.index(targetEnt) in rel_list[entity_list.index(k)][0])):
                            rel = entity_list.index(targetEnt)
                            idx = rel_list[rel,0].tolist().index(entity_list.index(k))
                            advice_relation_mask[rel,0,idx] = 1
                    else:
                        nbrsId = getNeighborList(targetEnt,rel_list)
                        nbrs= [entity_list[i] for i in nbrsId]
                        for n in nbrs:
                            if hasWordinEntity(ent,entitiesInQuestion[k],n):
                                rel = entity_list.index(targetEnt)
                                idx = rel_list[rel,0].tolist().index(entity_list.index(n))
                                advice_relation_mask[rel,0,idx] = 1
                
                                
def getNeighborList(entity, rel_list):
    #print(entity)
    if entity not in entity_list:
        raise Exception("index out of bound")
    elif entity_list.index(entity) > len(rel_list):
        raise Exception("entity not in raw_rel")
    else:
        newL = rel_list[entity_list.index(entity)][0]
        ret = np.asarray(newL)
        #print ret
    return [newL[i] for i in np.nonzero(ret)[0]]


def hasWordinEntity(nodefile,word,entity):
    ret = None
    #print len(entity_whole_list)
    #with open(nodefile) as tsv:
    #for line in csv.reader(tsv, dialect="excel-tab"):
    for line in entity_whole_list:
        #print line[0],entity
        if line[0]==entity:
            if word in line[1]:
                #print "found"
                ret = True
    return ret


def getAdvice(nodeFile,relFile,feats,labels,rel_list, train):
    global entity_list 
    global entity_whole_list
    entity_list, entity_whole_list = parseEntities(nodeFile)
    #print entity_list
    # parseRel(relFile)
    parseAdvice(nodeFile,adviceSet,feats,labels,rel_list,train)
    print np.nonzero(advice_entity_label)
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