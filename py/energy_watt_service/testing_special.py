
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import time, os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from IPython.core.debugger import Tracer
from nilm import NILM_Trainer

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

get_ipython().magic(u'matplotlib notebook')


# In[2]:

time_step=10


# In[4]:

model = load_model('/Volumes/MAC_SUB/workspace/개방형_중간점검/energy_api_train/square_training/result/train_201611_16_21_02/model.h5')


# In[5]:

train_df = pd.read_csv('../data/fan_mix_hair/train_50_shuffled.csv')
std = train_df['watt'].std()
mean  = train_df['watt'].mean()


# In[9]:

def trans_watt(watt, std, mean):
    watt_list = time_step*[watt]
    watt_np = np.array(watt_list, dtype=float)
    watt_np = (watt_np-mean)/float(std)
    watt_np = watt_np.reshape(1, time_step, 1)
    return watt_np


# In[28]:

watt = 38
input_np = trans_watt(watt, std, mean)
pred_np = model.predict_classes(input_np, verbose=1)
print(pred_np)


# In[29]:



