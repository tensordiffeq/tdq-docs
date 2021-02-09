# Compiling your Problem

Once you've defined your problem, it must be compiled such that TensorDiffEq can build the loss function described
by the boundary conditions, initial conditions, and physics defined in the previous sections.

Additionally, here is where we will define the neural network size and depth. Currently, most PINN approaches use dense fully connected
neural networks for function approximation. Fully-connected Neural Networks have some level of theoretical backing
that they will converge to a solution of the underlying function {cite}`pinkus1999approximation,chen1993approximations`, and this
theoretical backing has extended into the PINN framework {cite}`shin2020convergence`. With that being said, currently the only type of network supported in
TensorDiffEq is the fully-connected MLP network.

### Layer Sizes

TensorDiffEq uses the [Keras API](https://keras.io/) for neural network construction. All you need to do is define a list of layer
sizes for your neural network. So, for a network with an `[x,t]` input, 4 layers deep, with 128 nodes, one would define
a layer size list of `[2,128,128,1]`.

For our problem we have been building in the previous sections, we can define layer sizes as such:

```{code} python
layer_sizes = [2, 128, 128, 128, 128, 1]
```

Or, if your problem is a function of `[x,y,t]`, then you could define the exact same network with an input layer with 3 nodes, i.e.

```{code} python
layer_sizes = [3, 128, 128, 128, 128, 1]
```

### Compile the Model

```{bibliography} ../../references.bib
:style: unsrt
```