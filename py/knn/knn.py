from __future__ import division
import sys
import csv
import matplotlib.pylab as plt
import cPickle
import json
import numpy as np
import codecs
import time
from os import getcwd
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os.path import splitext
from os import listdir
from os import mkdir
from os import path
from numpy import genfromtxt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.externals import joblib

try:
  PATH = getcwd()
  f = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.info', 'r')
  p = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
  t = json.load(p)
  label_name = str(t['target'])
  k = int(t['k'])
  
  f.close()
  p.close()
  
  with codecs.open(join(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'), encoding = 'utf-8-sig') as csv_file:
      #init
      reader = csv.reader(csv_file)
      csv_data = list(reader)
      label_index = csv_data[0].index(label_name)
      temp = []
      dic = {}
      reverse_dic = {}
      
      #base values definition
      features = len(csv_data[0]) - 1
      samples = len(csv_data) - 1
      data = np.empty((samples, features))
      target = np.empty((samples,), dtype=np.int)
      
      #label dictionary & reverse_dictionary
      for i in range(samples):
        temp.append(csv_data[i+1][label_index])
      temp = set(temp)
      for i, name in enumerate(temp):
        dic[name] = i
        reverse_dic[i] = name
  
      #data parsing
      for i in range(samples):
        target[i] = np.asarray(dic[csv_data[i+1][label_index]], dtype=np.int)
        temp = []
        for j in range(features + 1):
          if j is not label_index:
            temp.append(csv_data[i+1][j])
        data[i] = np.asarray(temp, dtype=np.float)
  
  #learning
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(data, target)
  model_name = 'train_'+time.strftime("%Y%m_%d_%H_%M", time.localtime())
  
  dic = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.dic', 'w')
  out = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'w')
  
  P = knn.predict(data)
  retP = []
  for i in range(len(P)):
    retP.append(reverse_dic[P[i]])
  json.dump(reverse_dic, dic, separators=(',',':'))
  
  json.dump({'ntb':
              {
                'model_name' : model_name,
                'samples' : samples,
                'score' : knn.score(data, target),
                'accuracy' : accuracy_score(P, target),
                'recall_score' : recall_score(P, target, average='weighted'),
                'precision_score' : precision_score(P, target, average='weighted'),
              }
            ,
            'tb': 
              {
                'prediction' : retP
              }
            }, out, separators=(',',':'))
                 
  if not path.exists(PATH + '/data/' + sys.argv[1] + '/' + model_name):
    mkdir(PATH + '/data/' + sys.argv[1] + '/' + model_name)
  joblib.dump(knn, PATH + '/data/' + sys.argv[1] + '/' + model_name + '/' + sys.argv[1] + '.pkl')
  dic.close()
  out.close()
  
except:
  print >> sys.stderr, sys.exc_info()[0]