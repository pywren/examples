#from gevent import monkey

#monkey.patch_socket()
#monkey.patch_ssl()

import time
import boto3 
import uuid
import numpy as np
import time
import pywren
import subprocess
import logging
from scipy.ndimage.filters import convolve as convolveim
from scipy.signal import convolve as convolvesig


MAT_N = 1024

def compute_flops(loopcount):
    
    a = random.random((100, 100, 100))
    b = random.random((10,10,10))

    conv1 = convolveim(a,b, mode = 'constant')
#conv2 = convolvesig(a,b, mode = 'same')

    return conv1 # FLOPS / (t2-t1)

if __name__ == "__main__":

    fh = logging.FileHandler('simpletest.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(pywren.wren.formatter)
    pywren.wren.logger.addHandler(fh)

    t1 = time.time()

    wrenexec = pywren.default_executor()
    fut = wrenexec.call_async(compute_flops, 10)
    print fut.callset_id

    res = fut.result() 
    #print res/1e9, "GFLOPS"

