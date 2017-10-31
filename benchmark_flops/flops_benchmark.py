import numpy as np
import time
import pywren
import pickle as pickle
import click
import pandas as pd

from compute import compute_flops


def benchmark(loopcount, workers, matn, verbose=False):
    t1 = time.time()
    N = workers

    iters = np.arange(N)

    def f(x):
        return {'flops': compute_flops(loopcount, matn)}

    pwex = pywren.lambda_executor()
    futures = pwex.map(f, iters)

    print("invocation done, dur=", time.time() - t1)
    print("callset id: ", futures[0].callset_id)

    local_jobs_done_timeline = []
    result_count = 0
    while result_count < N:
        fs_dones, fs_notdones = pywren.wait(futures)
        result_count = len(fs_dones)

        local_jobs_done_timeline.append((time.time(), result_count))
        est_flop = 2 * result_count * loopcount * matn ** 3

        est_gflops = est_flop / 1e9 / (time.time() - t1)
        if verbose:
            print("jobs done: {:5d}    runtime: {:5.1f}s   {:8.1f} GFLOPS ".format(result_count,
                                                                                    time.time() - t1,
                                                                                    est_gflops))

        if result_count == N:
            break

        time.sleep(1)
    if verbose:
        print("getting results")
    results = [f.result(throw_except=False) for f in futures]
    if verbose:
        print("getting status")
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]

    all_done = time.time()
    total_time = all_done - t1
    print("total time", total_time)
    est_flop = result_count * 2 * loopcount * matn ** 3

    print(est_flop / 1e9 / total_time, "GFLOPS")
    res = {'total_time': total_time,
           'est_flop': est_flop,
           'run_statuses': run_statuses,
           'invoke_statuses': invoke_statuses,
           'callset_id': futures[0].callset_id,
           'local_jobs_done_timeline': local_jobs_done_timeline,
           'results': results}
    return res


def results_to_dataframe(benchmark_data):
    callset_id = benchmark_data['callset_id']

    func_df = pd.DataFrame(benchmark_data['results']).rename(columns={'flops': 'intra_func_flops'})
    statuses_df = pd.DataFrame(benchmark_data['run_statuses'])
    invoke_df = pd.DataFrame(benchmark_data['invoke_statuses'])

    est_total_flops = benchmark_data['est_flop'] / benchmark_data['workers']
    results_df = pd.concat([statuses_df, invoke_df, func_df], axis=1)
    Cols = list(results_df.columns)
    for i, item in enumerate(results_df.columns):
        if item in results_df.columns[:i]: Cols[i] = "toDROP"
    results_df.columns = Cols
    results_df = results_df.drop("toDROP", 1)
    results_df['workers'] = benchmark_data['workers']
    results_df['loopcount'] = benchmark_data['loopcount']
    results_df['MATN'] = benchmark_data['MATN']
    results_df['est_flops'] = est_total_flops
    return results_df


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
    res['MATN'] = matn

    pickle.dump(res, open(outfile, 'wb'), -1)


if __name__ == "__main__":
    run_benchmark()
