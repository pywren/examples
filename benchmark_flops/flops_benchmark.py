import numpy as np
import time
import pywren
import cPickle as pickle
import click

from compute import compute_flops

def benchmark(loopcount, workers, matn):

        
    t1 = time.time()
    N = workers

    iters = np.arange(N)
    
    def f(x):
        return {'flops' : compute_flops(loopcount, matn)}

    pwex = pywren.default_executor()
    futures = pwex.map(f, iters)

    print "invocation done, dur=", time.time() - t1
    print "callset id: ", futures[0].callset_id

    local_jobs_done_timeline = []
    result_count = 0
    while result_count < N:
        fs_dones, fs_notdones = pywren.wait(futures, pywren.wren.ALWAYS)
        result_count = len(fs_dones)
        
        local_jobs_done_timeline.append((time.time(), result_count))
        est_flop = 2 * result_count * loopcount * matn**3
        
        est_gflops = est_flop / 1e9/(time.time() - t1)
        print "jobs done: {:5d}    runtime: {:5.1f}s   {:8.1f} GFLOPS ".format(result_count, 
                                                                           time.time()-t1, 
                                                                           est_gflops)
        
        if result_count == N:
            break

        time.sleep(1)
    print "getting results" 
    results = [f.result() for f in futures]
    print "getting status" 
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]

    all_done = time.time()
    total_time = all_done - t1
    print "total time", total_time
    est_flop = result_count * 2 * loopcount * matn**3
    
    print est_flop / 1e9/total_time, "GFLOPS"
    res = {'total_time' : total_time, 
           'est_flop' : est_flop, 
           'run_statuses' : run_statuses, 
           'invoke_statuses' : invoke_statuses, 
           'callset_id' : futures[0].callset_id, 
           'local_jobs_done_timeline' : local_jobs_done_timeline, 
           'results' : results}
    return res

@click.command()
@click.option('--workers', default=10, help='how many workers', type=int)
@click.option('--outfile', default='flops_benchmark.pickle', 
              help='filename to save results in')
@click.option('--loopcount', default=6, help='Number of matmuls to do.', type=int)
@click.option('--matn', default=1024, help='size of matrix', type=int)
def run_benchmark(workers, outfile, loopcount, matn):
    res = benchmark(loopcount, workers, matn)
    res['loopcount'] = loopcount
    res['workers'] = workers
    res['MATN'] = MATN
    
    pickle.dump(res, open(outfile, 'w'), -1)

if __name__ == "__main__":
    run_benchmark()
