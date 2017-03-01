# pywren examples

This is a repository of pywren examples, showing how to run various
example code and generate many of the plots used in blog posts.
Most examples have an explanation in a `README.md`, a script to run, and
often a [Jupyter notebook](http://jupyter.org/) for interactively examining
results. 

Note that these examples, in addition to requiring the latest pywren, 
often require additional packages like Jupyter/iPython, Matplotlib, Seaborn, 
and the Ruffus pipeline manager. 


All pywren examples can be found [in our examples github repository](https://github.com/pywren/examples) 
most often as [Jupyter/IPython notebooks](http://jupyter.org/)

### Hello World

[Hello world](hello_world/hello_world.ipynb) is a simple example to
get you up and running with pywren.

### TFLOPS on microservices

An example of how to achieve over 40 TFLOPS of numerical performance
using pure-Python code running on thousands of simultaneous
cores. This example is based on our [original blog post](http://pywren.io/pywren.html) and
our
[recent paper](https://arxiv.org/abs/1702.04024). [[code]]flops_benchmark)

### GB/s from S3
We can achieve up to 80 GB/sec read and 60 GB/sec write performance to S3 in this
benchmark example, based on our [original blog post](http://pywren.io/pywren_s3.html). We have
notebooks that [show how to benchmark](benchmark_s3/s3_benchmark.ipynb) and then [how to measure scaling](benchmark_s3/s3_scaling.ipynb). [[code]](benchmark_s3/). 

### Measuring Lambda's recycling
[coming soon]

### Running a parameter server
[coming soon]

### Large-scale reduction
[coming soon]

### Robust Kalman Filtering
[coming soon]

### Inverse problems with sweep
[coming soon]


