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
    apiKey = 'Ml0r3rcjFkyV77it'
    method = 'SVR'
    model = 'train_201701_31_12_54'
    inp = {'Hour' : 4, 'Day' : 1, 'DayOfWeek' : 6, 'Month' : 1, 'Humidity' : 53,'Temp' : -12.3};

    queryData = json.dumps({
      'apiKey': apiKey,
      'method': method,
      'model' : model,
      'inp': inp
    })
    print queryData
    print 
    urlf = open(PATH + '/request_restful_data/url', 'r')
    t = json.load(urlf)
    url = t['url'] + ':8080/run'
    clen = len(queryData)
    urlf.close()
    
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
