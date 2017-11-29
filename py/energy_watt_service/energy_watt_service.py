
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import time, os, sys
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from IPython.core.debugger import Tracer
from IPython import get_ipython
from nilm import NILM_Trainer

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

PATH = getcwd()
f = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.info', 'r')
p = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
t = json.load(p)
path_trained_dir = None
# path_trained_dir = '../lstm/result/separate_100_timestep/train_201611_08_10_22'
hidden_units = int(t['hidden_unit'])
learning_rate = float(t['learning_rate'])
time_step = int(t['timesteps'])
#hidden_units = 10
#learning_rate = 1.0
#time_step=10
appl_list = ['fan', 'massager', 'hairdryer']
appl_num = len(appl_list)
input_dim = 1
nb_classes = pow(2, appl_num)
validation_split = 0.1
epoch = 3000
batch_size = 128
# repeat = 6
shuffle_num = 50
path_train = '%s/data/%s/%s.csv'%(PATH, sys.argv[1], sys.argv[1])
path_test = '%s/data/%s/%s.csv'%(PATH, sys.argv[1], sys.argv[1])

# loss = "mse"
loss = "categorical_crossentropy"
output_activation = "softmax"

def build_gru_model(input_dim, nb_classes, hidden_units):
    print('Building a model ...')
    model = Sequential()
    model.add(GRU(hidden_units, input_dim=input_dim, return_sequences=True, init='he_uniform'))
    model.add(Dropout(0.2))
#     model.add(GRU(hidden_units, input_dim=hidden_units, return_sequences=True, init='he_uniform'))
#     model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(nb_classes)))
    model.add(Activation(output_activation))
    adadelta = Adadelta(lr=learning_rate)
    model.compile(loss=loss, optimizer=adadelta, metrics=["accuracy"])
    print('End of Building a model ...')
    return model

def build_lstm_model(input_dim, nb_classes, hidden_units):
    print('Building a model ...')
    model = Sequential()
    model.add(LSTM(hidden_units, input_dim=input_dim, return_sequences=True, init='he_uniform'))
    model.add(Dropout(0.2))
#     model.add(LSTM(hidden_units, input_dim=hidden_units, return_sequences=True, init='he_uniform'))
#     model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(nb_classes)))
    model.add(Activation(output_activation))
    adadelta = Adadelta(lr=learning_rate)
    model.compile(loss=loss, optimizer=adadelta, metrics=["accuracy"])
    print('End of Building a model ...')
    return model

def build_rnn_model(input_dim, nb_classes, hidden_units):
    print('Building a model ...')
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_dim=input_dim, return_sequences=True, init='he_uniform'))
    model.add(Dropout(0.2))
#     model.add(SimpleRNN(hidden_units, input_dim=hidden_units, return_sequences=True, init='he_uniform'))
#     model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(nb_classes)))
    model.add(Activation(output_activation))
    adadelta = Adadelta(lr=learning_rate)
    model.compile(loss=loss, optimizer=adadelta, metrics=["accuracy"])
    print('End of Building a model ...')
    return model

trainer = NILM_Trainer(path_train, path_test, build_lstm_model)

trainer.init(input_dim, nb_classes, target_dir = '%s/data/%s'%(PATH, sys.argv[1]), epoch=epoch, batch_size=batch_size, hidden_units=hidden_units,
              time_step=time_step, learning_rate=learning_rate, validation_split=validation_split, path_trained_dir=path_trained_dir, key=sys.argv[1])
trainer.load_data()

model, train_history = trainer.train()