# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import time, os, sys, csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from IPython.core.debugger import Tracer
from IPython import get_ipython
from nilm import NILM_Trainer

import subprocess
import signal
import codecs
import json
from os.path import join
from os import getcwd

from keras.preprocessing import sequence
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, SimpleRNN, LSTM
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint, History

#get_ipython().magic(u'matplotlib notebook')


def trans_watt(watt, std, mean, time_step):
    watt_list = time_step*[watt]
    watt_np = np.array(watt_list, dtype=float)
    watt_np = (watt_np-mean)/float(std)
    watt_np = watt_np.reshape(1, time_step, 1)
    return watt_np

# In[2]:
PATH = getcwd()
path_model = '%s/data/%s/%s'%(PATH, sys.argv[1], sys.argv[2])
p = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
model = load_model('%s/model.h5'%path_model)
t = json.load(p)
time_step = int(t['timesteps'])
#time_step = 500
appl_list = ['FA', 'MI', 'HA']
prediction = []
pid = os.getpid()
f = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.pid', 'w')
f.write(str(pid))
p.close()
f.close()
print ("load model complete");

while True:
    req, ret_pid = raw_input().split()
    pass
    path_req = '%s/data/%s/request/%s'%(PATH, sys.argv[1], req)
    with codecs.open(join(path_req + '.csv'), encoding = 'utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        csv_data = list(reader)
        watt = csv_data[1][0]
        
        input_np = trans_watt(watt, 35.905232617833576, 52.99945, time_step) #after
        #input_np = trans_watt(watt, 33.4766182013, 50.00095, time_step)
        pred_np = model.predict_classes(input_np, verbose=1)
        for i in range(len(appl_list)):
            if pred_np[0][0] & (1<<i):
                prediction.append(appl_list[i])
        print (prediction)
        req = open(path_req + '.req', 'w')
        json.dump({
                'prediction': prediction
                }, req, separators=(',',':'))
        prediction = []
        req.close()
        os.kill(int(ret_pid), signal.SIGTERM)