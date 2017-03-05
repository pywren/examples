import numpy as np
import time
import cPickle as pickle
import sklearn.preprocessing
import util

X_train = np.load("imagenet-train-all-scaled-tar.tarfile_keys.features_bulk.npy",  mmap_mode='r')
y_meta = pickle.load(open("imagenet-train-all-scaled-tar.tarfile_keys.labels.meta.pickle", 'r'))

y_train = y_meta['y']

X_train = X_train # [::10]
y_train = y_train #[::10]

y_oh = util.to_onehot(y_train)

print util.direct_solve(X_train, y_oh)

