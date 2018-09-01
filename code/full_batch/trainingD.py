# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:39:28 2018

@author: mayuk
"""

import numpy
import sys
import prepare_data
import sample_dataD as sd
args = prepare_data.arg_passing(sys.argv)
seed = args['-seed']
numpy.random.seed(seed)

from keras.optimizers import *
from create_model import *
import FinalLayerAccess as fla

#MD + Yang + Dev
sampleFname = None
sampleFname = args['-sample']

dataset = args['-data']
task = 'software'
if 'pubmed' in dataset:
    task = 'pubmed'
elif 'movie' in dataset:
    task = 'movie'
elif 'debate' in dataset:
    task = 'debate'

dataset = '../data/' + dataset + '.pkl.gz'
nodeFile =  '../data/' + 'pubmed-Diabetes.NODE.paper.tab' #MD & Yang
relFile = '../data/' + 'pubmed-Diabetes.DIRECTED.cites.tab' #MD & Yang
modelType = args['-model']
n_layers, dim = args['-nlayers'], args['-dim']
shared = args['-shared']
saving = args['-saving']
nmean = args['-nmean']
yidx = args['-y']

if 'dr' in args['-reg']: dropout = True
else: dropout = False
#feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids = prepare_data.load_data(dataset)
feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, I_adv, W_adv_mask, c_adv_mask = sd.sample_data(dataset,nodeFile,relFile,0.4, sampleFname)
#I_adv = numpy.array(I_adv).reshape((len(I_adv),1))
W_adv_mask = numpy.array(W_adv_mask).reshape((len(W_adv_mask),1))
#W_adv_mask = numpy.repeat(W_adv_mask,feats.shape[-1],axis=1)
c_adv_mask = numpy.array(c_adv_mask)
print(len(train_ids), len(valid_ids), len(test_ids))
labels = labels.astype('int64')
if task == 'movie':
    labels = labels[:, yidx : yidx+1]

def remove(y, not_ids):
    new_y = numpy.copy(y)
    for ids in not_ids:
        new_y[ids] = -1
    return new_y

if type == 'software':
    train_y = remove(labels, [test_ids])
    valid_y = remove(labels, [train_ids])
else:
    train_y = remove(labels, [valid_ids, test_ids])
    valid_y = remove(labels, [train_ids, test_ids])

n_classes = numpy.max(labels)
if n_classes > 1:
    n_classes += 1
    loss = multi_sparse_graph_loss
else:
    loss = graph_loss

if 'movie' in task:
    n_classes = -labels.shape[-1]

#print "I advice :::::::::::::: ",I_adv, len(I_adv),n_classes

I_temp = [i*n_classes+I_adv[i] for i in range(0, len(I_adv))]

#print "I_temp :::", I_temp[19716], len(I_temp)
I_mask = numpy.zeros((len(I_temp),n_classes))
numpy.put(I_mask,I_temp,1.0)
I_ad = numpy.array(I_adv).reshape((len(I_adv),1))
fla.init(len(labels),n_classes) # MD initialize fla
########################## BUILD MODEL ###############################################
print('Building model ...')

# create model: n_layers, hidden_dim, input_dim, n_rel, n_neigh, n_classes, shared

if modelType == 'Highway':
    model = create_highway(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
elif modelType == 'Dense':
    model = create_dense(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1], adv_dim = I_ad.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
else:
    model = create_resNet(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)

model.summary()

# Full batch learning, learning rate should be large
lr = 0.01
opt = {'RMS': RMSprop(lr=lr, rho=0.9, epsilon=1e-8),
       'Adam': Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)}
model.compile(optimizer=opt[args['-opt']], loss=loss)

if 'movie' not in task:
    train_y = numpy.expand_dims(train_y, -1)
    valid_y = numpy.expand_dims(valid_y, -1)

json_string = model.to_json()
fModel = open('models/' + saving + '.json', 'w')
fModel.write(json_string)
fModel.close()

fParams = 'bestModels/' + saving + '.hdf5'
fResult = 'log/' + saving + '.txt'
print("before log")
f = open(fResult, 'w')
f.write('Training log:\n')
f.close()
print("after log")

saveResult = SaveResult([[feats, rel_list, rel_mask, I_ad, W_adv_mask, c_adv_mask, fla.fprobs, I_mask], labels, train_ids, valid_ids, test_ids],
                        task=task, fileResult=fResult, fileParams=fParams)

callbacks=[saveResult, NanStopping()]

his = model.fit([feats, rel_list, rel_mask, I_ad, W_adv_mask, c_adv_mask, fla.fprobs, I_mask], train_y,
                validation_data=([feats, rel_list, rel_mask, I_ad, W_adv_mask, c_adv_mask, fla.fprobs, I_mask], valid_y),
                nb_epoch=1000, batch_size=feats.shape[0], shuffle=False,
                callbacks=callbacks)
