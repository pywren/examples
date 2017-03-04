import numpy as np

import sklearn.linear_model

import cPickle as pickle

def chunk(l, chunk_size):
    """
    Chunk a list into sublists of size chunk_size 
    """
    N = len(l)
    res = []
    pos = 0
    while pos < N:
        res_sub = []
        for i in range(chunk_size):
            res_sub.append(l[pos])

            pos += 1

            if pos == N:
                break
        res.append(res_sub)
    return res


downsampled_X_train = np.load("working.dir/imagenet-train-all-scaled-tar.tarfile_keys.downsample.featuredata.X_train.npy", mmap_mode='r')
downsampled_y_train = np.load("working.dir/imagenet-train-all-scaled-tar.tarfile_keys.downsample.featuredata.y_train.npy", mmap_mode='r')
downsampled_X_test = np.load("working.dir/imagenet-train-all-scaled-tar.tarfile_keys.downsample.featuredata.X_test.npy", mmap_mode='r')
downsampled_y_test = np.load("working.dir/imagenet-train-all-scaled-tar.tarfile_keys.downsample.featuredata.y_test.npy", mmap_mode='r')

m = sklearn.linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0.01, 
                                       l1_ratio=0.015, n_jobs=32)
unique_labels = np.unique(downsampled_y_train)

TRAIN_N = len(downsampled_y_train)
BATCH_SIZE = 1000
EPOCH_N = 20
for ei in range(EPOCH_N):
    idx = np.random.permutation(TRAIN_N)

    for ci, c in enumerate(chunk(idx, BATCH_SIZE)):
        
    
        m.partial_fit(downsampled_X_train[c], downsampled_y_train[c], 
                      unique_labels)
        print ei, ci
    pickle.dump(m, open("sgd.{:08d}.pickle".format(ei), 'w'), -1)
