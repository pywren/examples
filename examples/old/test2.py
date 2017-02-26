from __future__ import print_function
import pickle
import subprocess
import traceback 
import sys

def subprocess_fail():

    out_filename = "test.pickle"
    try:
        subprocess.check_output("dummycommand", shell=True)

    except Exception as e:
        pickle.dump(e, open(out_filename, 'wb'))

    # now try and reload

    pickle.load(open(out_filename, 'rb'))


class CantPickle(object):
    def __init__(self, foo, dump_fail=False, load_fail=False):
        self.foo = foo
        self.dump_fail = dump_fail
        self.load_fail = load_fail
    
    def __getstate__(self):
        print("getstate called")
        if self.dump_fail:
            raise Exception("cannot pickle dump this object")

        return {'foo' : self.foo, 
                'dump_fail' : self.dump_fail, 
                'load_fail' : self.load_fail}

    def __setstate__(self, arg):
        print("setstate called")
        if arg['load_fail']:
            raise Exception("cannot pickle load this object")

        self.load_fail = arg['load_fail']
        self.dump_fail = arg['dump_fail']
        self.foo = arg['foo']


cp = CantPickle(7, dump_fail = True)

out_filename = "test.pickle"

pickle.dump(cp, open(out_filename, 'wb'), -1)

pickle.load(open(out_filename, 'rb'))

