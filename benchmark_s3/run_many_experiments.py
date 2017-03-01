from ruffus import * 
import numpy as np
import time
import sys
sys.path.append("../")
import exampleutils
import pywren

import exampleutils
import cPickle as pickle


import s3_benchmark


def ruffus_params():
    for workers in [1, 10, 30, 100, 300, 600, 1000, 2000, 2800]:
        for seed in range(5):
            for mb_per_file in [1000]: # 10, 100, 1000]:
        
                prefix = "s3_microbench.{}.{}.{}".format(workers, seed, 
                                                             mb_per_file)
                outfile = prefix + ".pickle"
                yield None, outfile, workers, seed, mb_per_file

BUCKET_NAME = 'jonas-pywren-benchmark'
AWS_REGION='us-west-2'

@files(ruffus_params)

def run_exp(infile, outfile, workers, seed, mb_per_file):
    # write simply writes that number of keys
    print("running {} {} {} {}".format(outfile, workers, seed, mb_per_file))
    t1 = time.time()
    write_results = s3_benchmark.write(BUCKET_NAME, mb_per_file, workers, 
                                      key_prefix="", region=AWS_REGION)
    t2 = time.time()
    read_results = s3_benchmark.read(BUCKET_NAME, workers, 
                                     write_results['keynames'], 
                                     read_times=1, 
                                     region=AWS_REGION)
    t3 = time.time()
    pickle.dump({'write_results' : write_results, 
                 'read_results' : read_results, 
                 'write_start_timestamp' : t1,
                 'read_start_timestamp' : t2, 
                 'read_done_timestamp' : t3, 
                 'workers' : workers, 
                 'seed' : seed, 
                 'mb_per_file' : mb_per_file}, 
                open(outfile, 'w'), -1)

    print("workers=", workers, "seed=", seed, "mb_per_file=", mb_per_file, "runtime:{:3.0f}".format(t3-t1))
                 
if __name__ == "__main__":
    pipeline_run([run_exp])

