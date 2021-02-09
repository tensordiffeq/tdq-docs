# Compiling your Problem

Once you've defined your problem, it must be compiled such that TensorDiffEq can build the loss function described
by the boundary conditions, initial conditions, and physics defined in the previous sections.

Additionally, here is where we will define the neural network size and depth. Currently, most PINN approaches use dense fully connected
neural networks for function approximation. Fully-connected Neural Networks have some level of theoretical backing
that they will converge to a solution of the underlying function {cite}`pinkus1999approximation, chen1993approximations`, and this
theoretical backing has extended into the PINN framework {cite}`shin2020convergence`. With that being said, currently the only type of network supported in
TensorDiffEq is the fully-connected MLP network.


```{bibliography} references.bib
:style: unsrt
```