import scipy.ndimage
from scipy.ndimage import zoom
import numpy as np
import boto
import sklearn.preprocessing
import time

def centerscale(img, PIXN):
    """
    center and scale the image and then crop
    """
    H, W, C = img.shape
    if H < PIXN or W < PIXN: 
        D = min(H, W)
        z = float(PIXN) / D

        img = zoom(img, (z, z, 1), order=1)
        H, W, C = img.shape
    elif H > PIXN and W > PIXN:
        D = min(H, W)
        z =  float(PIXN) / D

        img = zoom(img, (z, z, 1), order=1)
        H, W, C = img.shape
            
        
    assert H >= PIXN
    assert W >= PIXN
    # return centered img
    H_border = H - PIXN 
    W_border = W - PIXN 
    out_img =  img[H_border/2:H_border/2 + PIXN, 
                   W_border/2 : W_border/2 + PIXN]
    if out_img.shape != (PIXN, PIXN, C):
        print out_img.shape, PIXN, H_border, W_border
        raise ValueError()
    return out_img


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


def to_onehot(y_train):
    ohe = sklearn.preprocessing.OneHotEncoder()
    y_oh = ohe.fit_transform(y_train.reshape(-1, 1)).todense().astype(np.float32)
    return y_oh

def direct_solve(X, y):
    t1 = time.time()
    XtX = np.dot(X.T, X)
    t2 = time.time()
    Xty = np.dot(X.T, y)
    t3 = time.time()
    w = np.linalg.solve(XtX, Xty)
    t4 = time.time()

    print "dot(X^T, X) took {:3.1f} sec".format(t2-t1)
    print "dot(X^T, y_oh) took {:3.1f} sec".format(t3-t2)
    print "solve took {:3.1f} sec".format(t4-t3)
    return {'w': w, 
            'dot(X^T, X)' : t2-t1, 
            'dot(X^T, y)' : t3-t2, 
            'solve' : t4-t3}

            
