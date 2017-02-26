

import pywren

if __name__ == "__main__":
    import logging
    #logging.basicConfig(level=logging.DEBUG)

    def test_add(x):
        return x + 7

    wrenexec = pywren.default_executor()
    x = [1, 2, 3, 4]
    N = len(x)
    futures = wrenexec.map(test_add, x, invoke_pool_threads=2)

    fs_dones, fs_notdones = pywren.wait(futures)
    result_count = len(fs_dones)
    f = futures[0]

    print f.result(throw_except=False)
    print f._call_invoker_result
    print [f.result() for f in futures]
