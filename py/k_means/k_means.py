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
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
try:
  PATH = getcwd()
  f = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.info', 'r')
  p = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
  columns = f.read().splitlines()
  t = json.load(p)
  k = int(t['k'])
  f.close()
  p.close()
  
  with codecs.open(join(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'), encoding = 'utf-8-sig') as csv_file:
    reader = csv.reader(csv_file)
    csv_data = list(reader)
    n_samples = len(csv_data) - 1
    n_features = len(csv_data[0])
    data = np.empty((n_samples, n_features))
    
    for i in range(n_samples):
      temp = []
      for j in range(n_features):
        temp.append(csv_data[i+1][j])
      data[i] = np.asarray(temp, dtype=np.float)
      
  k_means = KMeans(n_clusters=k, random_state=0).fit(data)
  P = k_means.predict(data)
  model_name = 'train_'+time.strftime("%Y%m_%d_%H_%M", time.localtime())
  
  out = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'w')
  retP = []
  for i in range(len(P)):
    retP.append(str(P[i]));
  json.dump({'ntb':
              {
                'model_name' : model_name,
                'samples' : n_samples
              }
            ,
            'tb': 
              {
                'predict' : retP
              }
            }, out, separators=(',',':'))
  
  if not path.exists(PATH + '/data/' + sys.argv[1] + '/' + model_name):
    mkdir(PATH + '/data/' + sys.argv[1] + '/' + model_name)
  joblib.dump(k_means, PATH + '/data/' + sys.argv[1] + '/' + model_name + '/' + sys.argv[1] + '.pkl')
  
  out.close()
    
except:
  print >> sys.stderr, sys.exc_info()[0]
