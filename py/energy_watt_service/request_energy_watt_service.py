# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import time, os, sys, csv
from IPython.core.debugger import Tracer
from IPython import get_ipython

import subprocess
import json
import threading
import Queue
from os.path import join
from os import getcwd
from getpass import getpass

PATH = getcwd()
apikey = sys.argv[1]
req_name = sys.argv[2]
pid = open(PATH + '/data/' + apikey + '/' + apikey + '.pid', 'r').read()
request_pid = os.getpid()

echo_path = 'echo %s %d > /proc/%d/fd/0'%(req_name, request_pid, int(pid))
#echo_path = 'echo %s %d > /proc/%d/fd/0'%(req_name, request_pid, int(pid))
p = subprocess.Popen(echo_path, shell=True, stdout=subprocess.PIPE)

time.sleep(5);