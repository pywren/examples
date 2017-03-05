import tarfile
import os
import boto
import cStringIO
import cPickle as pickle
from ruffus import * 
import scipy.misc
import pywren
import time
from  scipy.misc import imread
import numpy as np
import random 
import time
import gist
from sklearn.model_selection import train_test_split
from util import centerscale
import pandas as pd
import re
import util
import scipy.linalg
import sklearn.metrics as metrics
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder

WORKING_DIR = "working.dir"
tw = lambda x : os.path.join(WORKING_DIR, x)

def get_filelocs_from_tar(bucket, key):
    """
    Walk through the tarfile header blocks and get contained file
    offsets
    """
    conn = boto.connect_s3(is_secure=False) # THIS IS HORRIBLE WHAT ARE WE THINKING? 

    b = conn.get_bucket(bucket)
    c = b.get_key(key)
    s = c.get_contents_as_string()

    sfid = cStringIO.StringIO(s)
    #filename = "/tmp/n13044778-scaled.tar"
    tf = tarfile.open(fileobj=sfid)

    file_locations = []
    for name in tf.getnames():
        tf_info = tf.getmember(name)
        if tf_info.isfile():
            data_size = tf_info.size
            data_offset = tf_info.offset_data
            name_filename = os.path.basename(name)
            file_locations.append((name_filename, data_offset, data_size))
    return file_locations

def get_key_region(bucket, key, offset, length):
    """
    get "length" bytes at offset for a particular key. 
    Useful for extracting files from inside tars
    """
    conn = boto.connect_s3(is_secure=False) # THIS IS HORRIBLE WHAT ARE WE THINKING? 

    b = conn.get_bucket(bucket)
    c = b.get_key(key)
    start_byte = offset
    end_byte = offset + length - 1
    headers={'Range' : 'bytes={}-{}'.format(start_byte, end_byte)}
    s = c.get_contents_as_string(headers=headers)
    return s

def params():
    for s  in ["imagenet-train-all-scaled-tar", 
               "imagenet-validation-all-scaled-tar"
    ]:
        yield None, "{}.tarfile_keys.pickle".format(s), s, s

@files(params)
def get_tarfiles(infile, outfile, bucket, prefix):
    
    conn = boto.connect_s3(is_secure=False) # THIS IS HORRIBLE WHAT ARE WE THINKING? 

    b = conn.get_bucket(bucket)
    res = []
    for k in b.list(prefix=prefix):
        if ".tar" in k.key:
            res.append((bucket, k.key))
    pickle.dump(res, open(outfile, 'w'))

@transform(get_tarfiles, suffix(".pickle"), ".image_offsets.pickle")
def get_file_offsets(infile, outfile):
    t1 = time.time()
    
    keys = pickle.load(open(infile, 'r'))
    
    def get((b, k)):
        return b, k, get_filelocs_from_tar(b, k)
    

    wrenexec = pywren.default_executor()
    futures = wrenexec.map(get, keys)

    res = [f.result() for f in futures]
    t2 = time.time()
    pickle.dump({'records' : res, 
                 'runtime' : t2-t1}, 
                open(outfile, 'w'))
    print "get_file_offsets runtime=", t2-t1

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

def load_image(bucket, key, offset, length):
    """
    Load an image at a particular offset in a key
    """
    im_str = get_key_region(bucket, key, offset, length)

    img = imread(cStringIO.StringIO(im_str))
    return img

def simple_pixel_stats(img):
    img = img.astype(np.float32)
    return np.mean(img), np.var(img)


def gist_features(img):
    assert img.dtype == np.uint8
    a = centerscale(img, 256)
    assert a.shape == (256, 256, 3)
    b = gist.extract(a)
    return b

def split_string_region(s, recordlist):
    """
    split a string at indicated points
    """
    res = []
    for offset, length in recordlist:
        res.append(s[offset:offset+length])
    return res


def downsample_features(img):
    assert img.dtype == np.uint8
    a = centerscale(img, 64)
    assert a.shape == (64, 64, 3)
    return a.astype(np.float32).flatten()

@transform(get_file_offsets, 
           suffix(".image_offsets.pickle"), 
           (".features_bulk.npy", ".job_stats.pickle"))
def process_images_bulk(infile, (outfile_features, outfile_stats)):

    
    offsets = pickle.load(open(infile, 'r'))
    records = offsets['records']

    for b, k, img_offset_list in records:
        # sanity check
        offsets = [im_data[1] for im_data in img_offset_list]
        assert (np.diff(offsets) > 0).all()

    records_split = []
    NUM_CHUNKS = 3
    JOB_N = 100000
    for b, k, img_offset_list in records:
        for c in chunk(img_offset_list, len(img_offset_list)/NUM_CHUNKS + 2):
            records_split.append((b, k, c))

    def improc((bucket, key, img_offset_list)):
        """
        Featurize a particular bucket/key at the images listed
        at various offsets
        """

        first_offset = img_offset_list[0][1]
        last_offset = img_offset_list[-1][1]
        last_length = img_offset_list[-1][2]
        s = get_key_region(bucket, key, first_offset, last_offset + last_length)
        
        reslist = []
        base_os = first_offset 
        for filename, offset, length in img_offset_list:
            im_str = s[offset-base_os:offset-base_os+length]
            img = imread(cStringIO.StringIO(im_str))
            if len(img.shape) == 2: # sometimes grayscale wtf? 
                img_new = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img_new[:, :, 0] = img
                img_new[:, :, 1] = img
                img_new[:, :, 2] = img
                img = img_new

            res = downsample_features(img)

            reslist.append((filename, res))
        return bucket, key, reslist
    
    wrenexec = pywren.default_executor()
    # shuffle so we're not always hitting the same key

    t1 = time.time()
    records_to_process = records_split[:JOB_N]
    futures = wrenexec.map(improc, records_to_process)

    RECORD_N = len(records_to_process)
    image_N = np.sum([len(r[2]) for r in records_to_process])
    fut_notdone = futures
    result_count = 0
    metadata_list = []
    run_statuses = []
    invoke_statuses = []
    while result_count < RECORD_N:
        fut_done, fut_notdone = pywren.wait(fut_notdone, pywren.wren.ALWAYS, 
                                            THREADPOOL_SIZE=128, 
                                            WAIT_DUR_SEC=2)
        if len(fut_done) > 0 and result_count == 0:
            # first pass through
            res_bucket, res_key, reslist = fut_done[0].result()
            f = reslist[0]
            out_shape = (image_N, f[1].shape[0])
            print "output shape is {}".format(out_shape)
            imdata = np.lib.format.open_memmap(outfile_features, mode='w+', 
                                               dtype=np.float32,
                                               shape=out_shape)
                                                  
            
            next_image_pos = 0

        for f in fut_done:
            res_bucket, res_key, res_data = f.result()
            run_statuses.append(f.run_status)
            invoke_statuses.append(f.invoke_status)
            for filename, features in res_data:
                metadata_list.append((res_bucket, res_key, filename))
                imdata[next_image_pos] = features
                next_image_pos += 1

        result_count += len(fut_done)
        print("just completed {}, total completed {}, waiting for {}".format(len(fut_done), result_count, 
                                                                             len(fut_notdone)))
    t2 = time.time()
    pickle.dump({'image_n' : image_N, 
                 'metadata' : metadata_list, 
                 'run_statuses' : run_statuses, 
                 'invoke_statuses' : invoke_statuses, 
                 'runtime' : t2-t1}, 
               open(outfile_stats, 'w'), -1)

    # print "featurize runtime=", t2-t1, "{:3.1f} img/sec".format(IMAGE_N/(t2-t1))
    # print "Total images", IMAGE_N
    # print "total runners", len(res)
    # print "successfully ran gist"
    

@transform(process_images_bulk, suffix(".features_bulk.npy"), 
           ".labels.meta.pickle")
def process_features((infile_features, infile_meta), outfile):
    print "infile_features=", infile_features, "infile_meta=", infile_meta, "outfile=", outfile

    meta = pickle.load(open(infile_meta, 'r'))
    metadata = meta['metadata']
    
    dataframe = pd.DataFrame(metadata, columns=["bucket", "key", "filename"])
    dataframe['tarfile'] = dataframe.key.apply(lambda x : x.split("/")[1])
    

    label_re = re.compile(".*(n\d+)-.+.tar")
    label_str = dataframe['tarfile'].apply(lambda x : label_re.match(x).group(1))
    dataframe['label_str'] = label_str

    labels_to_labelnum = {k : v for v, k in enumerate(np.unique(label_str))}
    label_num = dataframe.label_str.apply(lambda x : labels_to_labelnum[x])
    label_num = np.array(label_num)
    dataframe['label_num'] = label_num
    pickle.dump({'infile_features' : infile_features, 
                 'infile_meta' : infile_meta, 
                 'labels_to_labelnum' : labels_to_labelnum, 
                 'dataframe' : dataframe, 
                 'y_label' : dataframe['label_str'], 
                 'y' : np.array(label_num)}, 
                open(outfile, 'w'))


@mkdir(WORKING_DIR)
@subdivide(process_features, formatter(),
            # Output parameter: Glob matches any number of output file names
            tw("{basename[0]}.*.chunk"),
            # Extra parameter:  Append to this for output file names
            tw("{basename[0]}"))
def subdivide_fold_files(infile, output_files, output_base):
    # split into chunks to get parallel dist matrix calculations
    # 
    import sklearn.metrics
    d = pickle.load(open(infile, 'r'))
    X = np.load(d['data_filename'])
    y = np.load(d['label_filename'])

    fold_random_state = 10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=fold_random_state)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_train_filename = output_base + ".X_train.npy"
    X_test_filename = output_base + ".X_test.npy"
    y_train_filename = output_base + ".y_train.npy"
    y_test_filename = output_base + ".y_test.npy"
    np.save(X_train_filename, X_train)
    np.save(y_train_filename, y_train)
    np.save(X_test_filename, X_test)
    np.save(y_test_filename, y_test)

    CHUNK_SIZE = 1000

    TEST_N = X_test.shape[0]
    TOTAL_CHUNKS = int(np.ceil(TEST_N / float(CHUNK_SIZE)))
    pos = 0
    config_i = 0
    while config_i < TOTAL_CHUNKS:
        config = {'X_train_filename' : X_train_filename, 
                  'X_test_filename' : X_test_filename, 
                  'y_train_filename' : y_train_filename, 
                  'y_test_filename' : y_test_filename, 
                  'config_i' : config_i, 
                  'range' : (pos, pos+CHUNK_SIZE), }
        chunk_filename = output_base + ".{}.chunk".format(config_i)
        pickle.dump(config, open(chunk_filename, 'w'))
        print "chunk_filename=", chunk_filename
        pos += CHUNK_SIZE
        config_i += 1


@transform(subdivide_fold_files, suffix(".chunk"), ".knn")
def run_knn(infile, outfile):
    """
    compute knn on top k 
    """

    import sklearn
    t1 = time.time()
    d = pickle.load(open(infile, 'r'))
    X_train_filename = d['X_train_filename']
    X_test_filename = d['X_test_filename']
    y_train_filename = d['y_train_filename']
    y_test_filename = d['y_test_filename']

    X_train = np.load(X_train_filename, mmap_mode='r')
    y_train = np.load(y_train_filename, mmap_mode='r')
    X_test = np.load(X_test_filename, mmap_mode='r')
    y_test = np.load(y_test_filename, mmap_mode='r')

    config_i = d['config_i']
    row_min, row_max = d['range']
    X_test = X_test[row_min:row_max]
    y_test = y_test[row_min:row_max]
    TEST_N = X_test.shape[0]

    top_K = 30
    top_labels_pos = np.zeros((TEST_N, top_K), dtype=np.uint32)
    top_dist_vals = np.zeros((TEST_N, top_K), dtype=np.float32)
    top_labels = np.zeros((TEST_N, top_K), dtype=np.uint32)

    dists = sklearn.metrics.pairwise.pairwise_distances(X_train, X_test)
    # arg partition only returns the top K to the left of the pivot 
    top_k_pos = np.argpartition(dists, kth=top_K-1, axis=0)[:top_K]

    # need to resort top K 
    for i in range(TEST_N):
        a = top_k_pos[:, i]
        b = np.argsort(dists[a, i])

        top_labels_pos[i] = a[b]
        top_dist_vals[i] = dists[a[b], i]
        top_labels[i] = y_train[a[b]]

    t2 = time.time()
    print "runtime was", t2-t1
    pickle.dump({'infile' : infile, 
                 'config_i' : config_i, 
                 'TEST_N' : TEST_N, 
                 'top_labels_pos' : top_labels_pos, 
                 'top_dist_vals' : top_dist_vals, 
                 'top_labels' : top_labels, 
                 'top_K' : top_K, 
                 'runtime' : t2-t1}, 
                open(outfile, 'w'))

def learnPrimal(trainData, labels, W=None, reg=0.1, TOT_FEAT=1):
    '''Learn a model from trainData -> labels '''

    trainData = trainData.reshape(trainData.shape[0],-1)
    n = trainData.shape[0]
    X = np.ascontiguousarray(trainData, dtype=np.float64).reshape(trainData.shape[0], -1)
    if (W is None):
        W = np.ones(n)[:, np.newaxis]

    sqrtW = np.sqrt(W)
    X *= sqrtW
    XTWX = X.T.dot(X)
    XTWX /= float(TOT_FEAT)
    idxes = np.diag_indices(XTWX.shape[0])
    XTWX[idxes] += reg
    # generate one-hot encoding
    y = np.eye(max(labels) + 1)[labels]
    XTWy = X.T.dot(W * y)
    model = scipy.linalg.solve(XTWX, XTWy)
    return model


@follows(process_features)
@files(("imagenet-train-all-scaled-tar.tarfile_keys.features_bulk.npy", 
        "imagenet-train-all-scaled-tar.tarfile_keys.labels.meta.pickle"), 
       "lstsq_direct_solve.pickle")
def lstsq_direct_solve((X_filename, y_filename), out_filename):


    X_train = np.load(X_filename,  mmap_mode='r')
    y_meta = pickle.load(open(y_filename, 'r'))

    y_train = y_meta['y']

    X_train = X_train # [::10]
    y_train = y_train #[::10]

    res = learnPrimal(X_train, y_oh)

    # res= util.direct_solve(X_train, y_oh)
    pickle.dump({'X_filename' : X_filename, 
                 'y_filename' : y_filename, 
                 'res' : res}, 
                open(out_filename, 'w'))

@follows(lstsq_direct_solve)
@files(("imagenet-train-all-scaled-tar.tarfile_keys.features_bulk.npy", 
       "imagenet-train-all-scaled-tar.tarfile_keys.labels.meta.pickle", 
       "imagenet-validation-all-scaled-tar.tarfile_keys.features_bulk.npy", 
       "imagenet-validation-all-scaled-tar.tarfile_keys.labels.meta.pickle", 
       "lstsq_direct_solve.pickle"), "evaluate.pickle")
def evaluate_lstsq((train_data, train_meta, 
                    test_data, test_meta, 
                    model_filename), output_filename):
    
    X_train = np.load(train_data, mmap_mode='r')
    y_train_meta = pickle.load(open(train_meta, 'r'))
    
    X_test = np.load(test_data, mmap_mode='r')
    y_test_meta = pickle.load(open(test_meta, 'r'))
    
    train_labelnum_to_label = {k : v for v, k in y_train_meta['labels_to_labelnum'].items()}

    model = pickle.load(open(model_filename, 'r'))
    w = model['res']['w'] 
    pred_scores = X_test_w = np.dot(X_test, w)

    pred_label_pos = np.array(np.argmax(pred_scores, axis=1)).flatten()

    pred_label = [train_labelnum_to_labpel[p] for p in pred_label_pos]

    y_test_pred_num = [y_test_meta['labels_to_labelnum'][i] for i in pred_label]
    print np.sum(y_test_pred_num == y_test_meta['y'])/float(len(y_test_pred_num))
    
    

if __name__ == "__main__":
    pipeline_run([get_tarfiles, get_file_offsets, process_images_bulk, 
                  process_features, lstsq_direct_solve, 
                  evaluate_lstsq]) # process_features])
    #checksum_level=0)
                  #process_features, subdivide_fold_files, run_knn], multiprocess=18)
    #load_image
    #pipeline_printout(target_tasks=[get_tarfiles, get_file_offsets, process_images_bulk], checksum_level=0)
