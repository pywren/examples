from pywren.cloudpickle import cloudpickle, serialize
from cStringIO import StringIO
import pandas as pd
import sklearn.linear_model
import cPickle as pickle

def f(x):
    m = sklearn.linear_model.Lasso()
    return str(x)
args = [pd.Series([1, 2, 3])]



        
s = StringIO()
cp = cloudpickle.CloudPickler(s, 2)
cp.dump(f)
print len(cp.modules), len(s.getvalue())


s = StringIO()
cp = cloudpickle.CloudPickler(s, 2)

cp.dump(args)
print len(cp.modules), len(s.getvalue())


s = StringIO()
cp = cloudpickle.CloudPickler(s, 2)

cp.dump(f)
cp.dump(args)
print len(cp.modules), len(s.getvalue())
a = pickle.loads(s.getvalue())
# THIS ONLY RESTORES THE ORIGINAL FUNCTION


# now try with serializer
ser = serialize.SerializeIndependent()
list_of_strs, mod_paths = ser([f, args])
print [len(a) for a in list_of_strs]



