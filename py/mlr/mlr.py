# -*- coding: utf-8 -*-
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.externals import joblib
from sklearn import preprocessing
import logging
logging.basicConfig(level=logging.DEBUG)

class mlr(object):

    def setColumns(self):
        self.PATH = getcwd()
        f = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.info', 'r')
        p = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
        self.columns = f.read().splitlines()
        
        t = json.load(p)
        self.label_name = t['target']
        self.fit_intercept = bool(t['m_fit_intercept'])
        # self.label_name = 'label'
        # self.fit_intercept = True
    
    def pre_processing_for_data(self, data_file_path):
      
        #read data
        data_pd = pd.read_csv(data_file_path)
        #assign features 
        features_pd = data_pd
       
        if self.label_name in features_pd:
            features_pd = features_pd.drop(self.label_name,1)
        
        self.data_dim = len(features_pd.columns)
        self.samples = len(features_pd)
        #crate input data
        X_val = np.array(features_pd)
        if self.label_name is not None:
            if self.label_name in data_pd:
                label_pd = data_pd[self.label_name]
                #calculate label list
                label_list = list(set(np.reshape(label_pd.values,(-1,))))
                Y_val = np.array(label_pd)
            else:
                Y_val = 0
        else:
            Y_val = 0
      
        return X_val, Y_val
      
    def training(self):
        self.setColumns()
        X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
        X_train = preprocessing.scale(X_train)
        linereg = LinearRegression(fit_intercept=self.fit_intercept)
        linereg.fit(X_train, Y_train)
        
        score = linereg.score(X_train, Y_train)
        P = linereg.predict(X_train)
        accuracy = metrics.mean_squared_error(Y_train,linereg.predict(X_train))
        
        model_name = 'train_'+time.strftime("%Y%m_%d_%H_%M", time.localtime())
        
        out = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'w')
        predict = []
        for i in P:
            predict.append(i)
        json.dump({'ntb':
                    {
                        'model_name' : model_name,
                        'samples' : self.samples,
                        'score' : score
                    }
                  ,
                'tb': 
                      {
                        #'predict' :
                      }
                  }, out, separators=(',',':'))
                  
        if not path.exists(self.PATH + '/data/' + sys.argv[1] + '/' + model_name):
            mkdir(self.PATH + '/data/' + sys.argv[1] + '/' + model_name)
        joblib.dump(linereg, self.PATH + '/data/' + sys.argv[1] + '/' + model_name + '/' + sys.argv[1] + '.pkl')
        
        out.close()
        
    def test(self):
        self.setColumns()
        X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
        X_train = preprocessing.scale(X_train)
        clf = joblib.load(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[3] + '/' + sys.argv[1] + '.pkl')
        
        score = clf.score(X_train, Y_train)
        P = clf.predict(X_train)
        accuracy = metrics.mean_squared_error(Y_train, clf.predict(X_train))
        
        test = open(self.PATH + '/data/' + sys.argv[1] + '/test/' + sys.argv[2] + '.test', 'w')
        json.dump({
                    'ntb':
                    {
                            'samples' : self.samples,
                            'accuracy' : accuracy/100
                    },
                    'tb': 
                    {
                        #'predict' :
                    }
                }, test, separators=(',',':'))
        
        test.close()
        
    def request(self):
        self.setColumns()
        X_train, _ = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/request/' + sys.argv[2] + '.csv'))
        X_train = preprocessing.scale(X_train)
        clf = joblib.load(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[3] + '/' + sys.argv[1] + '.pkl')
          
        P = clf.predict(X_train)
          
        req = open(self.PATH + '/data/' + sys.argv[1] + '/request/' + sys.argv[2] + '.req', 'w')
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
                }, req, separators=(',',':'))
        req.close()
            
def main():
    mymlr = mlr()
    mymlr.training()
    # mymlr.test()
    # mymlr.request()
    
if __name__ == "__main__":
    main()