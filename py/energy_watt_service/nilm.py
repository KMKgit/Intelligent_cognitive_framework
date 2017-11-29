#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import time, os, sys
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from IPython.core.debugger import Tracer

import json
from os import getcwd

from keras.preprocessing import sequence
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint, History

class NILM_Trainer:    
    # hyper parameters
    learning_rate = None
    input_dim = None
    hidden_units = None
    time_step=None
    nb_classes = None
    validation_split = None
    epoch=None
    batch_size = None
    # path
    path_train = None
    path_test = None
    path_result = None
    path_history = None
    path_hyper_parameter = None
    path_compare_log = None
    path_attack_acc = None
    path_trained_dir = None
    path_trained_model = None
    path_save_model = None
    result_dir = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    train_df = None
    test_df = None
    cur_time = None
    std_list = None # to fix ========================================
    mean_list = None # to fix ========================================
    
    appl_list = None
    build_model = None
    model = None
    
    def __init__(self, path_train, path_test, build_model):
        self.path_train = path_train
        self.path_test = path_test 
        self.build_model = build_model 
        self.attack_cat_zero_list = []
        self._max_dict = {}
        self._total_columns = []
        
    def _trans_onehot(self, l, max_size):
        print("start trans_onehot")
        l_size = len(l)
        res = np.zeros((l_size, max_size))
        for i, item in enumerate(l):
            res[i][item[0]]=1
        print("end trans_onehot")
        return res
    
    def _gen_max_dict(self, header='infer'):
        print('Max Ditionary generating ...')
        self._max_dict = {}
        print("Loading a dataset [%s]..."%self.path_train)
        self.train_df = pd.read_csv(self.path_train, sep=',', header=header, low_memory=False)
        self.appl_list = self.train_df.columns

        print("Loading a dataset [%s]..."%self.path_test)
        self.test_df = pd.read_csv(self.path_test, sep=',', header=header, low_memory=False)

        input_cols = ['watt']  
        self.std_list = {} # to fix ========================================
        self.mean_list = {} # to fix ========================================        
        for col in input_cols: # to fix ========================================
            self.std_list[col] = self.train_df[col].std() # to fix ========================================
            self.mean_list[col] = self.train_df[col].mean() # to fix ========================================
        print('Std Mean Ditionary generated ...')
    
    def _process_data(self, df, header='infer'):

        columns = []
        for i in range(self.nb_classes):
            columns.append('case'+str(i))
        columns.append('watt')       

        result_np = np.zeros((len(df), len(columns)))
        result_df = pd.DataFrame(result_np, columns=columns)

        for i in range(self.nb_classes):
            
            bin_str = bin(i)[2:].zfill(len(df.columns)-1)
            query_list = []
            for j in range(len(bin_str)): 
                q = '%s==%d'%(self.appl_list[j], int(bin_str[j]))
                query_list.append(q)
            query = " and ".join(query_list)
            idx = df.query(query).index
            result_df.loc[idx, 'case%d'%i] = 1
        result_df['watt'] = df['watt']    

        use_columns = columns
        result_df.columns = columns
        cols = use_columns # header list
        input_cols = cols[-1:]
        output_cols = cols[:-1] # 'attack_cat'

        print("Normalizing ...")
        
        for col in input_cols: # to fix ========================================
            result_df[col] = result_df[col].fillna(0) # to fix ========================================
            result_df[col] = (result_df[col]-self.mean_list[col])/float(self.std_list[col])  # to fix ========================================     

        input_df = result_df[input_cols]
        output_df = result_df[output_cols]

        rows_len = len(result_df)
               
        print("row_len : ",rows_len)
        print("time_step : ", self.time_step)

        rows_len = rows_len/self.time_step*self.time_step
        
        print("input_dim", self.input_dim)
        
        if self.input_dim is None:
            self.input_dim = 1
        
        
        X = input_df.values[:rows_len]
        X = X.reshape(-1, self.time_step, self.input_dim)

        y = output_df.values[:rows_len]
        y = y.reshape(-1, self.time_step, self.nb_classes)

        print('shape of X: ',X.shape)
        print('shape of y: ',y.shape)
        print('End of load_data() ...')

        return X, y
    
    def load_data(self):
        self._gen_max_dict()
        self.X_train, self.y_train = self._process_data(self.train_df)
        self.X_test, self.y_test = self._process_data(self.test_df)
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    # 학습 관련
    def init(self, input_dim, nb_classes, target_dir, epoch=500, batch_size=128, hidden_units=100, time_step=100, learning_rate=0.001, validation_split=0.1, path_trained_dir=None, key=sys.argv[1]):
        self.cur_time = time.localtime()
        PATH = getcwd()
        self.result_dir = '%s/data/%s/train_%s'%(PATH, key, time.strftime("%Y%m_%d_%H_%M", self.cur_time))
        #mkdir(self.result_dir)
        os.mkdir( self.result_dir, 0755 )
        self.path_line_chart = '%s/train_history_chart.png'%(self.result_dir)
        self.path_hyper_parameter = '%s/hyper_parameter.csv' % (self.result_dir)
        self.path_result = '%s/result.txt'%(self.result_dir)
        self.path_compare_log = '%s/compare_log.csv'%(self.result_dir)
        self.path_attack_acc = '%s/attack_acc.csv'%(self.result_dir)
        self.path_save_model = '%s' % (self.result_dir) 
        self.path_history = '%s/history.txt' % (self.result_dir)                
        print ("%s is created"%self.result_dir)
        
        self.input_dim = input_dim
        self.nb_classes = nb_classes
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.time_step = time_step
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.path_trained_dir = path_trained_dir        

        print ("hyper parameters initialized")
        
        hyper_parameter_df = pd.DataFrame({'input_dim': [self.input_dim],
                                           'nb_classes': [self.nb_classes],
                                           'epoch': [self.epoch],
                                           'batch_size': [self.batch_size],
                                           'hidden_units': [self.hidden_units],
                                           'time_step': [self.time_step],
                                           'learning_rate': [self.learning_rate],
                                           'validation_split': [self.validation_split],
                                           'path_trained_dir': [self.path_trained_dir],})
        hyper_parameter_df.to_csv(self.path_hyper_parameter)
        
        self.model = self.build_model(self.input_dim, self.nb_classes, self.hidden_units)
        if path_trained_dir:
            print ('Loading a previous model')
            self.path_trained_model = '%s/model.h5' % (self.path_trained_dir)
            self.model = load_model(self.path_trained_model)
        print ('End of Building a model ...')
    
    def _save_history(self, train_history):        
        with open(self.path_history, 'w') as f:
            f.write (str(train_history.history))

    def train(self):
        PATH = getcwd()
        out = open(PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.out', 'w')
        json.dump({
            'ntb' : {
                'model_name' :  'train_' + time.strftime("%Y%m_%d_%H_%M", self.cur_time)
            }
        }, out, separators=(',',':'))
        out.close()
        print("Training ...")
        
        history_pointer = History()
        checkpointer = ModelCheckpoint(filepath=self.path_save_model+'/model.h5',
                                        verbose=1, save_best_only=True)
        train_history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, nb_epoch=self.epoch, validation_split=self.validation_split, verbose=1, shuffle=False, callbacks=[checkpointer, history_pointer])
        self.model.save(self.path_save_model+'/model.h5')
        self.model.save_weights(self.path_save_model+'/weight.h5')
        self._save_history(train_history)
        
        print("weights saved in %s"%self.path_save_model)
        print("End of Training ...")
        return self.model, train_history
      
    # 테스트 관련
    def _square_to_num(self, data):
        res = np.zeros(shape=(data.shape[0],1))

        for i in range(data.shape[0]):
            category = np.argmax(data[i])
            res[i] = category

        return res
    
    def __calc_conf(self, pred_df, actual_df):
        merge_df = pd.concat([pred_df, actual_df], axis=1)
        TP_cnt = len(merge_df.query('pred!=%d and pred==true'%0))
        FP_cnt = len(merge_df.query('pred!=%d and pred!=true'%0))
        TN_cnt = len(merge_df.query('pred==%d and pred==true'%0))   
        FN_cnt = len(merge_df.query('pred==%d and pred!=true'%0))

        return TP_cnt, FP_cnt, TN_cnt, FN_cnt

    def _calc_acc_metrics(self, TP_cnt, FP_cnt, TN_cnt, FN_cnt):    
        precision = TP_cnt/float(TP_cnt+FP_cnt) if TP_cnt+FP_cnt!=0 else 0
        recall = TP_cnt/float(TP_cnt+FN_cnt) if TP_cnt+FN_cnt!=0 else 0
        accuracy = (TP_cnt+TN_cnt)/float(TP_cnt+TN_cnt+FP_cnt+FN_cnt) if TP_cnt+TN_cnt+FP_cnt+FN_cnt!=0 else 0
        f1_score = 2*recall*precision/(recall+precision) if recall+precision!=0 else 0
        FPR = FP_cnt/float(FP_cnt+TN_cnt) if FP_cnt+TN_cnt!=0 else 0
        FNR = FN_cnt/float(FN_cnt+TP_cnt) if FN_cnt+TP_cnt!=0 else 0
        far = (FPR+FNR)/2

        return precision, recall, accuracy, f1_score, far
    
    def _save_compare_log(self, pred_np, actual_np):
        pred = pd.DataFrame(pred_np, dtype=int, columns=["pred"])
        test = pd.DataFrame(actual_np, dtype=int, columns=["true"])
        compare_df = pd.concat([pred, test], names=["pred", "true"], axis=1)
        
        if self.path_compare_log is None:
            self.path_compare_log =  str(self.PATH + '/data/' + sys.argv[1] + '/test/' + sys.argv[2] + '.test')
        
        compare_df.to_csv(self.path_compare_log, index=False)
        return compare_df
        
    #def rquest(self, path_trained_dir, watt):
        #model = load_model('%s/model.h5'%path_trained_dir)
        #watt=10
        #watt_list = self.time_step*[watt]
        #watt_np = np.array(watt_list, np.type=float)
        #watt_np = watt_np.reshape(1, self.time_step, 1)
        #pred_np = model.predict_classes(watt_np, verbose=1)
        #print ("shape of result: ", pred_np.shape)
    
    def load_params(self):
        self.PATH = getcwd()
        p = open(self.PATH + '/data/' + sys.argv[1] + '/' + sys.argv[1] + '.param', 'r')
      
        t = json.load(p)
        #if 'label_name' in t:
        #    self.label_name = str(t['label_name'])
        #else:
        #    self.label_name = None
        self.learning_rate = float(t['learning_rate'])
        #self.batch_size = int(t['batch_size'])
        #self.hidden_layer = int(t['hidden_layer'])
        self.hidden_unit = t['hidden_unit']
        #self.dropout = t['dropout']
        #self.epoch = int(t['epoch'])
        self.time_step = int(t['timesteps'])
        print("self.timeSteps", self.time_step)
        #self.learning_rate = 0.01
        #self.batch_size = 50
        #self.hidden_layer = 2
        #self.hidden_unit = [32, 32]
        #self.dropout = [0.5, 0.5]
        #self.epoch = 3
        #self.timesteps = 1
        
        #self.in_activation = "softmax"
        #self.loss_function = "categorical_crossentropy"
        
    
    def test(self, path_model, path_result):
        print ('Loading a previous model')
        model = load_model('%s/model.h5'%path_model)
        print ("model loading completed")
        self.appl_list = ['fan', 'massager', 'hairdryer']
        appl_num = len(self.appl_list)
        self.nb_classes = pow(2, appl_num)
        
        print("param path : ",self.path_test)
        
        self.test_df = pd.read_csv(self.path_test, sep=',', header='infer', low_memory=False)
        
        self._gen_max_dict()
        self.load_params()
        self.X_test, self.y_test = self._process_data(self.test_df)
        
        pred_np = model.predict_classes(self.X_test, verbose=1)
        print ("shape of result: ", pred_np.shape)
        total_cnt = len(pred_np)*len(pred_np[0])
        pred_np = pred_np.reshape((pred_np.shape[0]*pred_np.shape[1], 1))        
        y_test_np = self.y_test.reshape((self.y_test.shape[0]*self.y_test.shape[1], self.nb_classes))
        actual_np = self._square_to_num(y_test_np)
        
        
        compare_df = self._save_compare_log(pred_np, actual_np)
        
        
        TP_cnt, FP_cnt, TN_cnt, FN_cnt = self.__calc_conf(compare_df['pred'], compare_df['true'])
        precision, recall, accuracy, f1_score, far = self._calc_acc_metrics(TP_cnt, FP_cnt, TN_cnt, FN_cnt)
        
        test = open(path_result, 'w')
        json.dump({'ntb':
                    {
                      'samples' : len(self.X_test),
                      'Precision' : precision,
                      'Recall' : recall,
                      'Accuracy' : accuracy,
                      'F1 score' : f1_score
                    }
                  ,
                  'tb': 
                    {
                    }
                  }, test, separators=(',',':'))
        test.close()
        
        
        #with open(path_result, 'w') as f:
        #    f.write ("TP: %s\n" % TP_cnt)
        #    f.write ("TN: %s\n" % TN_cnt)
        #    f.write ("FP: %s\n" % FP_cnt)
        #    f.write ("FN: %s\n" % FN_cnt)
        #    f.write ("===============================================================\n")
        #    f.write ("Precision: %f\n" % precision)
        #    f.write ("Recall: %f\n" % recall)
        #    f.write ("Accuracy: %f\n" % accuracy)
        #    f.write ("F1 score: %f\n" % f1_score)
        #    f.write ("FAR: %f\n" % far)