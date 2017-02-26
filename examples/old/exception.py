import pywren
import subprocess
import sys
import traceback

def throw_exception(x):
    1 / 0
    return 10


if __name__ == "__main__":
    wrenexec = pywren.default_executor()

    fut = wrenexec.call_async(throw_exception, None)

    try:
        throw_exception(1)
    except Exception as e:
        print "first exception"
        exc_type_true, exc_value_true, exc_traceback_true = sys.exc_info()


    try:
        print fut.result()
    except Exception as e:
        print "second exception"
        exc_type_wren, exc_value_wren, exc_traceback_wren = sys.exc_info()

    print exc_type_wren == exc_type_true
    print type(exc_value_wren) == type(exc_value_true)
