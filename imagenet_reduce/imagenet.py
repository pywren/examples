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
import opt

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


def downsample_features(img, size):
    assert img.dtype == np.uint8
    a = centerscale(img, size)
    assert a.shape == (size, size, 3)
    return a.astype(np.float32).flatten()

def downsample_fft(img, size):
    a = centerscale(img, size)
    assert a.shape == (size, size, 3)
    res = []
    for i in range(3):
        f = np.fft.fft2(a[:, :, i])
        res.append(np.abs(f).astype(np.float32).flatten())
        res.append(np.angle(f).astype(np.float32).flatten())
    return np.concatenate(res)


FEATURIZE_LIST = ['downsample_8', 'downsample_32', 
                  'downsample_64', 'downsample_fft_8', 
                  'downsample_fft_32']

def exp_params():
    for featurizer in FEATURIZE_LIST:
        for phase in ['train', 'validation']:
            infile = "imagenet-{}-all-scaled-tar.tarfile_keys.image_offsets.pickle".format(phase)
            outfile_base = "{}.{}".format(featurizer, phase)
            yield (infile, (outfile_base + ".features_bulk.npy", 
                           outfile_base + ".job_stats.pickle"), 
                   phase, featurizer)


@files(exp_params)
def process_images_bulk(infile, (outfile_features, outfile_stats), 
                        phase, featurizer):

    
    offsets = pickle.load(open(infile, 'r'))
    records = offsets['records']


    
    if featurizer == 'downsample_8':
        featurize = lambda x: downsample_features(x, 8)
        NUM_CHUNKS = 3
    elif featurizer == 'downsample_32':
        featurize = lambda x: downsample_features(x, 32)
        NUM_CHUNKS = 3
    elif featurizer == 'downsample_64':
        featurize = lambda x: downsample_features(x, 64)
        NUM_CHUNKS = 3
    elif featurizer == 'downsample_fft_8':
        featurize = lambda x: downsample_fft(x, 8)
        NUM_CHUNKS = 3
    elif featurizer == 'downsample_fft_32':
        featurize = lambda x: downsample_fft(x, 32)
        NUM_CHUNKS = 5
    else:
        ValueError("unknown featurizer {}".format(featurizer))


    for b, k, img_offset_list in records:
        # sanity check
        offsets = [im_data[1] for im_data in img_offset_list]
        assert (np.diff(offsets) > 0).all()

    records_split = []

    JOB_N = 10000000
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

            res = featurize(img)
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

MODEL_LIST = ['lstsq']
def solve_params():
    for model in MODEL_LIST:
        for featurizer in FEATURIZE_LIST:
            X_train_filename = "{}.train.features_bulk.npy".format(featurizer) 
            y_train_filename = "{}.train.labels.meta.pickle".format(featurizer)
            X_test_filename = "{}.validation.features_bulk.npy".format(featurizer) 
            y_test_filename = "{}.validation.labels.meta.pickle".format(featurizer)
            outfile = "{}.{}.model.pickle".format(featurizer, model)
            yield (X_train_filename, y_train_filename,
                   X_test_filename, y_test_filename), outfile, model

@follows(process_features)
@files(solve_params)
def model_solve((X_train_filename, y_train_filename, 
                 X_test_filename, y_test_filename), out_filename, model_name):


    X_train = np.load(X_train_filename,  mmap_mode='r')
    y_meta = pickle.load(open(y_train_filename, 'r'))

    y_train = y_meta['y']

    if model_name == 'lstsq':
        mf = opt.MultiLeastSquares(np.logspace(-3, 10, 14))
        models = mf.multifit(X_train, y_train)
        model_configs = mf.get_configs()
        #m.fit(X_train, y_train)
    else:
        raise ValueError("Unknown model_name {}".format(model_name))


    # res= util.direct_solve(X_train, y_oh)
    pickle.dump({'X_train_filename' : X_train_filename, 
                 'y_train_filename' : y_train_filename, 
                 'X_test_filename' : X_test_filename, 
                 'y_test_filename' : y_test_filename, 
                 'model_name': model_name, 
                 'models' : models, 
                 'model_configs' : model_configs}, 
                open(out_filename, 'w'))

@follows(model_solve)
@transform(model_solve, suffix(".model.pickle"), 
           ".evaluate.pickle")
def evaluate_model(infile, outfile):
    
    model_data = pickle.load(open(infile, 'r'))
    X_train = np.load(model_data['X_train_filename'], mmap_mode='r')
    y_train_meta = pickle.load(open(model_data['y_train_filename'], 'r'))
    
    X_test = np.load(model_data['X_test_filename'], mmap_mode='r')
    y_test_meta = pickle.load(open(model_data['y_test_filename'], 'r'))
    
    train_labelnum_to_label = {k : v for v, k in y_train_meta['labels_to_labelnum'].items()}
    res = []
    for m, mc in zip(model_data['models'], 
                     model_data['model_configs']):

        pred_scores = m.predict_proba(X_test)

        pred_label_pos = np.array(np.argmax(pred_scores, axis=1)).flatten()

        pred_label = [train_labelnum_to_label[p] for p in pred_label_pos]

        y_test_pred_num = [y_test_meta['labels_to_labelnum'][i] for i in pred_label]

        accuracy = np.sum(y_test_pred_num == y_test_meta['y'])/float(len(y_test_pred_num))
        
        r = {'pred_socres' : pred_scores, 
             'pred_label' : pred_label, 
             'y_test_pred_num' : y_test_pred_num, 
             'y_test_meta' : y_test_meta['y'], 
             'accuracy' : accuracy, 
             'mc' : mc}
        
        print "Top-1 accuracy = {:3.1f}% for {}".format(accuracy*100, mc)
        res.append(r)
    pickle.dump({'res' : res}, 
                open(outfile, 'w'))


if __name__ == "__main__":
    pipeline_run([get_tarfiles, get_file_offsets, process_images_bulk, 
                  process_features, 
                  model_solve, 
                  evaluate_model, 
    ]) # process_features])
    #checksum_level=0)
                  #process_features, subdivide_fold_files, run_knn], multiprocess=18)
    #load_image
    #pipeline_printout(target_tasks=[get_tarfiles, get_file_offsets, process_images_bulk], checksum_level=0)
