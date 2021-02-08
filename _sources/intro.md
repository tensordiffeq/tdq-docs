TensorDiffEq 
============================
## Official Documentation

### *Under Construction* - bear with us as we build out these docs! 
#### To check recent activity and track progress check out the [repository](https://github.com/tensordiffeq/tdq-docs)
![Docker Release](https://github.com/tensordiffeq/TensorDiffEq/workflows/Docker%20Release/badge.svg)
![Conda Build](https://github.com/tensordiffeq/TensorDiffEq/workflows/Conda%20Build/badge.svg)
![Package Build](https://github.com/tensordiffeq/TensorDiffEq/workflows/Package%20Build/badge.svg)
![Package Release](https://github.com/tensordiffeq/TensorDiffEq/workflows/Package%20Release/badge.svg)
![pypi](https://img.shields.io/pypi/v/tensordiffeq)
![downloads](https://img.shields.io/pypi/dd/tensordiffeq)
![python versions](https://img.shields.io/pypi/pyversions/tensordiffeq)

## I AM A CHANGE

TensorDiffEq is a python package built on top of Tensorflow to provide scalable and efficient
PINN solvers. TensorDiffEq's primary purpose is for scalable solving of PINNs (inference) and 
inverse problems (discovery). 

Additionally, TensorDiffEq is the only package that fully supports and implements [Self-Adaptive PINN](https://arxiv.org/abs/2009.04544) solvers 
and is the only Multi-GPU PINN solution suite that is fully open-source. 

Many choices for your scientific machine learning solution exist, use TensorDiffEq if you require
- A meshless PINN solver that can distribute over multiple workers (GPUs) for
  forward problems (inference) and inverse problems (discovery)
- Scalable domains - Iterated solver construction allows for N-D spatio-temporal support
  - support for N-D spatial domains with no time element is included
- Self-Adaptive Collocation methods for forward and inverse PINNs
- Intuitive user interface allowing for explicit definitions of variable domains, 
  boundary conditions, initial conditions, and strong-form PDEs 
  
What makes TensorDiffEq different?
- Completely open-source
- [Self-Adaptive Solvers](https://arxiv.org/abs/2009.04544) for forward and inverse problems, leading to increased accuracy of the solution and stability in training, resulting in 
  less overall training time 
- Multi-GPU distributed training for large or fine-grain spatio-temporal domains
- Built on top of Tensorflow 2.0 for increased support in new functionality exclusive to recent TF releases, such as [XLA support](https://www.tensorflow.org/xla), 
[autograph](https://blog.tensorflow.org/2018/07/autograph-converts-python-into-tensorflow-graphs.html) for efficent graph-building, and [grappler support](https://www.tensorflow.org/guide/graph_optimization)
  for graph optimization* - with no chance of the source code being sunset in a further Tensorflow version release
  
- Intuitive interface - defining domains, BCs, ICs, and strong-form PDEs in "plain english"


Thanks to our affiliates:  



[comment]: <> (![Army Research Lab Logo]&#40;images/ARL-logo.jpg&#41;)

[comment]: <> (![Texas A&M Engineering]&#40;images/engineering-logo.png&#41;  )


```{image} images/devcom.png
:alt: Army Research Lab Logo
:height: 65px
:align: center
```  


```{image} images/engineering-logo.png
:alt: Texas A&M Engineering
:height: 65px
:align: center
```  


```{image} images/tamids.png
:alt: Texas A&M Institute of Data Science
:height: 65px
:align: center
```

