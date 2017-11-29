import numpy as np
import pandas as pd
import sys
import json
import time
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Dense, Dropout, Activation
from os import getcwd, environ, listdir, mkdir, path
from keras.optimizers import rmsprop
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.externals import joblib
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
#from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline



class logistic(object):
 
    def setColumns(self):
    
        # Load Datad
        self.PATH = getcwd()
        f = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.info', 'r')
        p = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
        columns = f.read().splitlines()
        t = json.load(p)
        
        self.in_learning_rate = float(t['learning_rate'])
        self.label_name = t['target']
        #self.in_n_iter  = int(t['n_iter'])
        #self.in_n_components = int(t['n_components'])
        #self.in_logistic_c = float(t['logistic_c'])
        #self.in_learning_rate = 0.01
        #self.in_n_iter  = 30
        #self.in_n_components = 63
        #self.in_logistic_c = 6000.0
        #self.label_name = 'label'

        #f.close()
        #p.close()

    def pre_processing_for_data(self, data_file_path):
      
        #read data
        data_pd = pd.read_csv(data_file_path)
        #assign features 
        features_pd = data_pd
        #features_pd = features_pd.drop('label',1)
       
        if self.label_name in features_pd:
            features_pd = features_pd.drop(self.label_name,1)
        
        self.data_dim = len(features_pd.columns)
        self.in_n_components = len(features_pd.columns)
        self.samples = len(features_pd)
        #crate input data
        X_val = np.array(features_pd)
        #adjust_offset_value = len(X_val)-(len(X_val) % self.timesteps)
        #X_val = X_val[0:adjust_offset_value]
        #X_val = X_val.reshape(-1, self.timesteps, self.data_dim)
        #assign label
        #label_pd = data_pd['label']
        if self.label_name is not None:
            if self.label_name in data_pd:
                label_pd = data_pd[self.label_name]
                #calculate label list
                label_list = list(set(np.reshape(label_pd.values,(-1,))))
                #create output data
                #Y_val = []
                #loop for number of data
                #for count, i  in enumerate(label_pd):
                #    idx = find_matching_index(label_list,i)
                #    Y_val.append(np.reshape(vectorized_Y_data(idx,len(label_list)),(-1,)))
                Y_val = np.array(label_pd)
                #Y_val = to_categorical(Y_val)
            else:
                Y_val = 0
        else:
            Y_val = 0
      
        #print(np.shape(X_val))
        #print(np.shape(Y_val))
        return X_val, Y_val

    
    def training(self):
        #init network configuration
        self.setColumns()
        #preprocessing data
        #X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/py/rbm/test.csv'))
        X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
        
        
        #create model
        logistic = linear_model.LogisticRegression()
        
        logistic.C = self.in_learning_rate
       
        logistic.fit(X_train, Y_train)
        
        acc, recall, precision, f1_score = cal_matrics(logistic, X_train, Y_train)

        
        model_name = 'train_'+time.strftime("%Y%m_%d_%H_%M", time.localtime())
        #save
        out = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'w')
        json.dump({'ntb':
                  {
                    'model_name' : model_name,
                    'samples' : self.samples,
                    'accuracy' : acc,
                    'recall' : recall,
                    'precision' : precision,
                    'f1_score' : f1_score
                  }
                ,
                'tb': 
                  {
                    #'predict' :
                  }
                }, out, separators=(',',':'))
        
        if not path.exists(self.PATH + '/data/' + sys.argv[1] + '/' + model_name):
            mkdir(self.PATH + '/data/' + sys.argv[1] + '/' + model_name)
        joblib.dump(logistic, self.PATH + '/data/' + sys.argv[1] + '/' + model_name + '/' + sys.argv[1] + '.pkl')
        
        out.close()
      
    def test(self):
        
        #init network configuration
        self.setColumns()
        #preprocessing data
        #X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/py/rbm/test.csv'))
        X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
        
        logistic_classifier = joblib.load(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[3] + '/' + sys.argv[1] + '.pkl')
        
        # Trainig result
        #RBM_result = metrics.classification_report(
        #    Y_train,
        #    classifier.predict(X_train))
        acc, recall, precision, f1_score = cal_matrics(logistic_classifier, X_train, Y_train)
        
        #print()
        #print("Logistic regression using RBM features:\n%s\n" % Logistic_result)
        
        test = open(self.PATH + '/data/' + sys.argv[1] + '/test/' + sys.argv[2] + '.test', 'w')
        json.dump({'ntb':
                      {
                        'samples' : self.samples,
                        'accuracy' : acc,
                        'recall' : recall,
                        'precision' : precision,
                        'f1_score' : f1_score
                    }
                    ,
                    'tb': 
                    {
                        #'predict' :
                    }
                }, test, separators=(',',':'))
        test.close()
        
    def request(self):

        #init network configuration
        self.setColumns()
        #preprocessing data
        #X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/py/rbm/test.csv'))
        X_train, _ = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/request/' + sys.argv[2] + '.csv'))
        
        logistic_classifier = joblib.load(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[3] + '/' + sys.argv[1] + '.pkl')
        
        # Trainig result
        P = logistic_classifier.predict(X_train)
        
        prediction_array = []
        for i in P:
            prediction_array.append(i)
            
        
        req = open(self.PATH + '/data/' + sys.argv[1] + '/request/' + sys.argv[2] + '.req', 'w')
        json.dump({'ntb':
                  {
                    'samples' : self.samples
                  }
                ,
                'tb': 
                  {
                    'predict' : prediction_array
                  }
                }, req, separators=(',',':'))
        req.close()


def cal_matrics(model, X, Y):
    
    predict = model.predict(X)
    
    acc = metrics.accuracy_score(
        Y,
        predict, normalize=False,sample_weight=None)

    recall = metrics.recall_score(
        Y,
        predict, average='micro')
        
    precision = metrics.precision_score(
        Y,
        predict, average='micro')
        
    f1_score = metrics.f1_score(
        Y,
        predict,average='micro')
        
    return acc, recall, precision, f1_score


def find_matching_index(src, dst):
    for i in range(0, len(src)):
        if src[i] == dst:
            return i
    return -1

def vectorized_Y_data(j,label_num):
    e = np.zeros((label_num, 1))
    e[j] = 1.0
    return e[:]

def main():
    mylogistic = logistic()
    mylogistic.training()
#    mylogistic.test()
#    mylogistic.request()

if __name__ == "__main__":
    main()


