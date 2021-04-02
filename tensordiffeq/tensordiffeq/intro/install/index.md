# Installing TensorDiffEq

Here are some baseline instructions for getting TensorDiffEq online and running on your system. We have multiple potential deployment 
environments that can potentially be utilized, from CPUs on a local machine (i.e. a windows PC or MacBook) all the way up to data center 
scale computations. Here we will break the install down into CPU and GPU implementations.

## Installing TensorDiffEq on a CPU-only Enviromnent

TensorDiffEq is freely available for install on [PyPi](https://pypi.org/project/tensordiffeq/) and can 
be installed using `pip`:

```code
pip install tensordiffeq
```

This will automatically install the latest stable version of Tensorflow with TensorDiffEq, but does not guarantee CUDA and its affiliated dependencies
will be installed for GPU. 

TensorDiffEq's developers develop on `tf-nightly`, meaning that the latest stable install version of Tensorflow should not provide any sort of 
version errors. However - if your local system (i.e. a supercomputing facility, etc.) has not upgraded to Tensorflow 2.x, more than likely TensorDiffeq 
will be unsupported. A version of Tensorflow > 2.0 is required on your local machine if you intend on executing TensorDiffEq on that machine. The `pip install` command 
should install all these dependencies automatically if you have administrator privileges on your machine, meaning that for most attempting to install via
`pip`, these concerns will not pose a problem.

It is important to note that while TensorDiffEq is scalable on the larger end, it is perfectly capable of running on a local machine on CPU. 
A unique feature of Tensorflow - and, by extension, TensorDiffEq - is that the software will automatically detect whether your hardware is GPU-compatible,
identify the number of workers available, and automatically utilize all of them, unless otherwise specified. With Tensorflow, the user does not need to worry about specific 
installations or setups to ensure that their code runs on their system, and we take that same approach with the TensorDiffEq package interface. 
Therefore, *very* minimal modifications to your code are required to distribute across multiple GPU workers, or run on a quad-core processor found in a personal computer. 


## Installing TensorDiffEq on a GPU Enviroment 


In order to fully utilize Tensorflow on GPU - and fully utilize the features of TensorDiffEq - all the upstream CUDA dependencies must be installed.
This can be a somewhat tedious and error prone task. It is *highly* recommended that you work with Tensorflow's [containerized distributions](https://www.tensorflow.org/install/docker)
for easy implementation of TensorDiffEq's solvers. 

*Typically* installing via `pip` does not pose problems when installing inside a Tensorflow docker container. However, depending on the age of your container build, some level
of CUDA dependency issues may arise. If this occurs, rebuild your docker container using the `latest` version via 

```{code}
docker run --gpus all -it tensorflow/tensorflow:latest-gpu bash

```

This will drop you into a GPU-enabled Tensorflow container, where you can run

```{code}
pip install tensordiffeq
```

and run scripts as applicable. 
