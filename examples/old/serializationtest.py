from __future__ import print_function
import pywren
import hashlib

if __name__ == "__main__":
    import logging
    #logging.basicConfig(level=logging.DEBUG)

    def test_add(x):
        return x + 3 + 4

    def test_serialize(func, data):
        serializer = pywren.cloudpickle.serialize.SerializeIndependent()
        func_and_data_ser, mod_paths = serializer([func] + data)

        func_str = func_and_data_ser[0]
        data_strs = func_and_data_ser[1:]
        return func_str, data_strs

    x = [1, 2, 3, 4]
    func_str, data_strs = test_serialize(test_add, x)
    print("function", hashlib.sha256(func_str).hexdigest())
    for di, d in enumerate(data_strs):
        print("d[{}]={}".format(di, hashlib.sha256(d).hexdigest()))
        
