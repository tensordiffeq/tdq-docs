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



This will automatically install the latest stable version of Tensorflow, but dies not guarantee CUDA and its affiliated dependencies
will be installed for GPU. 

It is important to note that while TensorDiffEq is scalable on the larger end, it is perfectly capable of running on a local machine on CPU. 
A unique feature of Tensorflow - and, by extension, TensorDiffEq - is that the software will automatically detect whether your hardware is GPU-compatible,
identify the number of workers available, and automatically utilize all of them, unless otherwise specified. The user does not need to worry about specific 
installations or setups to ensure that their code runs on their system, and we take that same attitude into consideration with the package. 
Therefore, we adopt a similar attitude, and *very* minimal modifications to your code are required to distribute across multiple GPU workers. 

in order to fully utilize Tensorflow on GPU - and fully utilize the features of TensorDiffEq - all the upstream CUDA dependencies must be installed.
This can be a somewhat tedious and error prone task. It is *highly* recommended that you work with Tensorflow's [containerized distributions]()