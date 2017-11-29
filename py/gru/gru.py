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

class gru(object):
    
    def setColumns(self):
        
        self.PATH = getcwd()
        f = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.info', 'r')
        p = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
        self.columns = f.read().splitlines()
      
        t = json.load(p)
        self.label_name = str(t['label_name'])
        self.learning_rate = float(t['learning_rate'])
        self.batch_size = int(t['batch_size'])
        self.hidden_layer = int(t['hidden_layer'])
        self.hidden_unit = t['hidden_unit']
        self.dropout = t['dropout']
        self.epoch = int(t['epoch'])
        #self.timesteps = int(t['timesteps'])
        
        #self.learning_rate = 0.01
        #self.batch_size = 50
        #self.hidden_layer = 2
        #self.hidden_unit = [32, 32]
        #self.dropout = [0.5, 0.5]
        #self.epoch = 3
        self.timesteps = 1
        
        self.in_activation = "softmax"
        self.loss_function = "categorical_crossentropy"
        
    def pre_processing_for_data(self, data_file_path):
      
        #read data
        data_pd = pd.read_csv(data_file_path)
        #assign features 
        features_pd = data_pd
        #features_pd = features_pd.drop('label',1)
       
        if self.label_name in features_pd:
            features_pd = features_pd.drop(self.label_name,1)
        
        self.data_dim = len(features_pd.columns)
        self.samples = len(features_pd)
        #crate input data
        X_val = np.array(features_pd)
        adjust_offset_value = len(X_val)-(len(X_val) % self.timesteps)
        X_val = X_val[0:adjust_offset_value]
        X_val = X_val.reshape(-1, self.timesteps, self.data_dim)
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
                Y_val = to_categorical(Y_val)
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
        X_train, Y_train = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
        
        #create model
        model = Sequential()
        #input layer
        model.add(GRU(output_dim = int(self.hidden_unit[0]), 
                    return_sequences=True, 
                    input_shape=(self.timesteps, self.data_dim)))
        #hidden layer
        for i in range(0, self.hidden_layer):
            if i == self.hidden_layer-1:
                model.add(GRU(int(self.hidden_unit[i])))
            else:
                model.add(GRU(int(self.hidden_unit[i]), return_sequences=True))
                model.add(Dropout(float(self.dropout[i])))
        #output layer
        model.add(Dense(len(Y_train[0]), activation=self.in_activation))
        #set cost-function, optimizser, metrics
        model.compile(loss=self.loss_function, optimizer=rmsprop(lr = self.learning_rate), metrics=['accuracy'])
        #do training
        model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.epoch, validation_data=(X_train, Y_train))
        #save model
        out = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'w')
        P =  model.predict_classes(X_train, verbose=0)
        score = model.evaluate(X_train, Y_train, verbose=0)
        model_name = 'train_'+time.strftime("%Y%m_%d_%H_%M", time.localtime())
        
        json.dump({'ntb':
                    {
                        'model_name' : model_name,
                        'samples' : self.samples,
                        'score' : score[0],
                        'accuracy' : score[1],
                        #'recall_score' : recall_score(P, Y_train, average='weighted'), # it's not working because of multi-dimension
                        #'precision_score' : precision_score(P, Y_train, average='weighted') # it's not working becuse of multi-dimension
                
                    }
                  ,
        
                  }, out, separators=(',',':'))
        
        if not path.exists(self.PATH + '/data/' + sys.argv[1] + '/' +  model_name):
            mkdir(self.PATH + '/data/' + sys.argv[1] + '/' + model_name)
        model.save(self.PATH + '/data/' + sys.argv[1] + '/' + model_name + '/' + 'model' + '.h5')  
        #model.save('./test_py/gru/weight.h5')
        del model
        out.close()


    def test(self):
        #init network configuration
        self.setColumns()
        #preprocessing data
        X_test, Y_test = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.csv'))
        #load latest train model
        f = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'r')
        t = json.load(f)
        model_path = self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[3] + '/' + 'model.h5'
        model = load_model(model_path)
        
        #save test result
        #test = open(self.PATH + '/data/' + sys.argv[1]  + '/test/' + 'test.test', 'w')
        test = open(self.PATH + '/data/' + sys.argv[1] + '/test/' + sys.argv[2] + '.test', 'w')
        P = model.predict_classes(X_test, verbose=0)
        score = model.evaluate(X_test, Y_test, verbose=0)
        
        json.dump({'ntb':
                    {
                      'samples' : self.samples,
                      'score' : score[0],
                      'accuracy' : score[1]
                    }
                  ,
                  'tb': 
                    {
                    }
                  }, test, separators=(',',':'))
        del model
        test.close()


    def request(self):
        
        #initialize data
        self.setColumns()
        #configure data path
        X_request, _ = self.pre_processing_for_data(str(self.PATH + '/data/' + sys.argv[1] + '/request/' + sys.argv[2] + '.csv'))
        #load latest train model
        f = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'r')
        t = json.load(f)
        model_path = self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[3] + '/' + 'model.h5'
        model = load_model(model_path)
        
        #save requset result
        #argv[2]
        req = open(self.PATH + '/data/' + sys.argv[1]  + '/request/' + sys.argv[2] +'.req', 'w')
        P = model.predict_classes(X_request, verbose=0)
        #score = model.evaluate(X_request, Y_request, verbose=0)
        
        prediction_array = []
        for i in P:
            prediction_array.append(i)
            
        json.dump({'ntb':
                    {
                      'samples' : self.samples,
                    }
                  ,
                  'tb': 
                    {
                        'prediction' : prediction_array
                    }
                  }, req, separators=(',',':'))
        del model
        req.close()

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
    mygru = gru()
    mygru.training()
    #mygru.test()
    #mygru.request()
    
if __name__ == "__main__":
    main()