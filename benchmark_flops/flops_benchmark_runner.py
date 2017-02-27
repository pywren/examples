from ruffus import * 
import numpy as np
import time
import sys
sys.path.append("../")
import exampleutils
import pywren

import exampleutils
import cPickle as pickle


import flops_benchmark

LOOPCOUNT = 20
MATN = 4096

def ruffus_params():
    for workers in [1, 30, 100, 300, 600, 1000, 1500, 2000, 2800]:
        for seed in range(10):
        
            outfile = "microbench.{}.{}.pickle".format(workers, seed)
            yield None, outfile, workers

@files(ruffus_params)

def run_exp(infile, outfile, workers):

    res = flops_benchmark.benchmark(LOOPCOUNT, workers, MATN)
    res['loopcount'] = LOOPCOUNT
    res['workers'] = workers
    res['MATN'] = MATN
    pickle.dump(res, open(outfile, 'w'), -1)


if __name__ == "__main__":
    pipeline_run([run_exp])

