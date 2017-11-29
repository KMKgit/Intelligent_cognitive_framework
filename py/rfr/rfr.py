import sys
import json
import numpy as np
import pandas as pd
import time
from os import getcwd
from os import mkdir
from os import path
from keras.utils.np_utils import to_categorical
from sklearn import linear_model, datasets, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import preprocessing
import logging
logging.basicConfig(level=logging.DEBUG)

class rfr(object):
  
  def setColumns(self):
    self.PATH = getcwd()
    f = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.info', 'r')
    p = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
    self.columns = f.read().splitlines()
    
    t = json.load(p)
    self.label_name = "label"
    self.n_estimators = bool(t['n_estimators'])
    # self.criterion = bool(t['criterion'])
    self.random_state = bool(t['random_state'])
    self.n_jobs = 
    # self.predict_col = int(t['predict_col'])
    # self.max_depth = 'None'
    # self.label_name = 'label'
    # self.label_name = t['target']
    
  def pre_processing_for_data(self, data_file_path):
    data_pd = pd.read_csv(data_file_path)
    features_pd = data_pd
    features_pd = features_pd.drop(self.label_name,1)
    self.data_dim = len(features_pd.columns)
    self.samples = len(features_pd)
    X_val = np.array(features_pd)
    label_pd = data_pd[self.label_name]
    label_list = list(set(np.reshape(label_pd.values,(-1,))))
    Y_val = np.array(label_pd)
    return X_val, Y_val
    
  def training(self):
    self.setColumns()
    X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
    X_train = preprocessing.scale(X_train)
    clf = RandomForestRegressor(random_state=self.random_state, n_estimators=self.n_estimators, verbose=True)
    clf.fit(X_train, Y_train)
    
    score = clf.score(X_train, Y_train)
    P = clf.predict(X_train)
    accuracy = metrics.mean_squared_error(Y_train,clf.predict(X_train))
    
    model_name = 'train_'+time.strftime("%Y%m_%d_%H_%M", time.localtime())
    #save
    out = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'w')
    predict = []
    for i in P:
        predict.append(i)
    json.dump({'ntb':
              {
                'model_name' : model_name,
                'samples' : self.samples,
                'accuracy' : accuracy
                
              }
            ,
            'tb': 
              {
                #'predict' :
              }
            }, out, separators=(',',':'))
    
    if not path.exists(self.PATH + '/data/' + sys.argv[1] + '/' + model_name):
        mkdir(self.PATH + '/data/' + sys.argv[1] + '/' + model_name)
    joblib.dump(clf, self.PATH + '/data/' + sys.argv[1] + '/' + model_name + '/' + sys.argv[1] + '.pkl')
    
    out.close()
    
  def test(self):
    self.setColumns()
    X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
    X_train = preprocessing.scale(X_train)
    f = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'r')
    t = json.load(f)
    tt = t['ntb']
    clf = joblib.load(self.PATH + '/data/' + sys.argv[1] + '/' + str(tt['model_name']) + '/' + sys.argv[1] + '.pkl')
    score = clf.score(X_train, Y_train)
    P = clf.predict(X_train)
    accuracy = metrics.mean_squared_error(Y_train, clf.predict(X_train))
    
    test = open(self.PATH + '/data/' + sys.argv[1] + '/test/' + sys.argv[1] + '.test', 'w')
    json.dump({'ntb':
              {
                'samples' : self.samples,
                'accuracy' : accuracy
                
              }
            ,
            'tb': 
              {
                #'predict' :
              }
            }, test, separators=(',',':'))
    
    test.close()
    
  def request(self):
    self.setColumns()
    X_train, _ = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
    X_train = preprocessing.scale(X_train)
    f = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'r')
    t = json.load(f)
    tt = t['ntb']
    clf = joblib.load(self.PATH + '/data/' + sys.argv[1] + '/' + str(tt['model_name']) + '/' + sys.argv[1] + '.pkl')
      
    P = clf.predict(X_train)
      
    request = open(self.PATH + '/data/' + sys.argv[1] + '/request/' + sys.argv[1] + '.request', 'w')
    predict = []
    for i in P:
      predict.append(i)
    json.dump({'ntb':
              {
                'samples' : self.samples,
              }
            ,
            'tb': 
              {
                'predict' : predict
              }
            }, request, separators=(',',':'))
    request.close()
      
def main():
  myrfr = rfr()
  myrfr.training()
  myrfr.test()
  myrfr.request()
  
if __name__ == "__main__":
    main()
