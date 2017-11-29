import json
import time
import urllib2
import random
import sys
import json

from os import getcwd
from termcolor import colored



#lines = sys.stdin.readlines()[1::]
#print lines

while True:
  try:
    PATH = getcwd()
    apiKey = '1bwlBNfpovUrI7o1'
    method = 'LSTM'
    model = 'train_201701_31_09_59'
    inp = {};
    for i in range(1, 64):
      inp.update({'feature' + str(i):i})
    
    queryData = json.dumps({
      'apiKey': apiKey,
      'method': method,
      'model' : model,
      'inp': inp
    })
    clen = len(queryData)
    
    print queryData
    print 
    urlf = open(PATH + '/request_restful_data/url', 'r')
    t = json.load(urlf)
    urlf.close()
    url = t['url'] + ':8080/run'
    #url = 'http://52.79.71.50:8080/run'
    
    req = urllib2.Request(url, queryData, {'Content-Type': 'application/json', 'Content-Length': clen})
    f = urllib2.urlopen(req)
    response = f.read()
    f.close()

    obj = json.loads(response)

    print colored(type(obj), 'green')
    print colored(obj, 'green')
    time.sleep(5)
  except:
    print 'Error'
    time.sleep(5)
