from __future__ import division
import sys
import csv
import matplotlib.pylab as plt
import cPickle
import json
import numpy as np
import codecs
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os.path import splitext
from os import getcwd
from os import listdir
from os import mkdir
from numpy import genfromtxt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.externals import joblib
try:
  PATH = getcwd()
  f = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.info', 'r')
  p = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
  r = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.dic', 'r')
  columns = f.read().splitlines()
  t = json.load(p)
  reverse_dic = json.load(r)
  column_names = t['columns']
  indexs = []
  k = int(t['k'])
  
  f.close()
  p.close()
  
  with codecs.open(join(PATH + '/data/' + sys.argv[1] + '/request/' + sys.argv[2] + '.csv'), encoding = 'utf-8-sig') as csv_file:
      reader = csv.reader(csv_file)
      csv_data = list(reader)
      for name in column_names:
        if name in csv_data[0]:
          indexs.append(csv_data[0].index(name))
      # for j in range(len(csv_data[0])):
      #   if csv_data[0][j] in column_names:
      #     indexs.append(column_names.index(csv_data[0][j]))
          # indexs.append(j)
          #indexs=[2,0,3,1]
      features = len(column_names)
      samples = len(csv_data) - 1
      temp = []
  
      data = np.empty((samples, features))
      for i in range(samples):
        temp = []
        for j in indexs:
          temp.append(csv_data[i+1][j])
          print "aa",temp
        data[i] = np.asarray(temp, dtype=np.float)
  
  knn = joblib.load(PATH + '/data/' + sys.argv[1] + '/pkl/' + sys.argv[1] + '.pkl')
  P = knn.predict(data)
  retP = []
  for i in range(len(P)):
    retP.append(reverse_dic[str(P[i])])
  print P[i]
  req = open(PATH + '/data/' + sys.argv[1] + '/request/' + sys.argv[2] + '.req', 'w')
  json.dump({
            'prediction': retP
            }, req, separators=(',',':'))
  req.close()
except:
  print >> sys.stderr, sys.exc_info()[0]
  