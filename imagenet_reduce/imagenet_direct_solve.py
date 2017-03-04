import numpy as np
import time
import cPickle as pickle
import sklearn.preprocessing

downsampled_X_train = np.load("working.dir/imagenet-train-all-scaled-tar.tarfile_keys.featuredata.X_train.npy", mmap_mode='r')
downsampled_y_train = np.load("working.dir/imagenet-train-all-scaled-tar.tarfile_keys.featuredata.y_train.npy", mmap_mode='r')

ohe = sklearn.preprocessing.OneHotEncoder()
y_oh = ohe.fit_transform(downsampled_y_train.reshape(-1, 1)).todense().astype(np.float32)

total_times = []
for i in range(10):
    X = downsampled_X_train.copy()
    t1 = time.time()
    XtX = np.dot(X.T, X)
    t2 = time.time()
    Xty = np.dot(X.T, y_oh)
    t3 = time.time()
    w = np.linalg.solve(XtX, Xty)
    t4 = time.time()

    print "dot(X^T, X) took {:3.1f} sec".format(t2-t1)
    print "dot(X^T, y_oh) took {:3.1f} sec".format(t3-t2)
    print "solve took {:3.1f} sec".format(t4-t3)

    total_times.append(t4-t1)
print np.mean(total_times), np.std(total_times)

