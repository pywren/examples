from __future__ import print_function
import pywren
import subprocess
import sys
import boto3

def get_size((bucket, key)):
    s3 = boto3.resource('s3')
    a = s3.meta.client.head_object(Bucket=bucket, Key=key)
    return a['ContentLength']    


if __name__ == "__main__":
    

    wrenexec = pywren.default_executor()
    fut = wrenexec.call_async(get_size, sys.argv[1:])
    print(fut.callset_id)

    res = fut.result() 
    print(res)
