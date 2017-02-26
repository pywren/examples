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
import os



MAT_N = 1024*32

def compute_flops(loopcount):
    
    A = np.arange(MAT_N**2, dtype=np.float64).reshape(MAT_N, MAT_N)
    B = np.arange(MAT_N**2, dtype=np.float64).reshape(MAT_N, MAT_N)

    t1 = time.time()
    for i in range(loopcount):
        c = np.sum(np.dot(A, B))

    FLOPS = 2 *  MAT_N**3 * loopcount
    t2 = time.time()
    return os.environ, FLOPS / (t2-t1)

if __name__ == "__main__":


    t1 = time.time()

    wrenexec = pywren.default_executor()
    fut = wrenexec.call_async(compute_flops, 4, 
                              extra_env={"OMP_NUM_THREADS" : "", 
                                         "DEBUG_ENV" : "HELLOWORLD"})
    print fut.callset_id

    env, flops = fut.result() 
    print flops/1e9, "GFLOPS"

