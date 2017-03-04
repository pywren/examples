import scipy.ndimage
from scipy.ndimage import zoom
import numpy as np
import boto

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
