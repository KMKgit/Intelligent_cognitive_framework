# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import time, os, sys, csv
from IPython.core.debugger import Tracer
from IPython import get_ipython
import json
import subprocess

from os.path import join
from os import getcwd


#get_ipython().magic(u'matplotlib notebook')
PATH = getcwd()
apikey = sys.argv[1]
model_name = sys.argv[2]
cmd = '(while [ 1 ]; do sleep 1; done) | python py/energy_watt_service/backend_request_energy_watt_service.py %s %s'%(apikey, model_name)
subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)