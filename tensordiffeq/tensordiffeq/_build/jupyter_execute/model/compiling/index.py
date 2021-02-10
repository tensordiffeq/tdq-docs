# Compiling your Problem

Once you've defined your problem, it must be compiled such that TensorDiffEq can build the loss function described
by the boundary conditions, initial conditions, and physics defined in the previous sections.

### Layer Sizes

Here is where we will define the neural network size and depth. Currently, most PINN approaches use dense fully connected
neural networks for function approximation. Fully-connected Neural Networks have some level of theoretical backing
that they will converge to a solution of the underlying function {cite}`pinkus1999approximation,chen1993approximations`, and this
theoretical backing has extended into the PINN framework {cite}`shin2020convergence`. With that being said, currently the only type of network supported in
TensorDiffEq is the fully-connected MLP network.

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

In order to compile the model, we first initialize the model we are interested in. Currently, forward solutions of PINNs are performed by
the `CollocationSolverND()` method.

#### Collocation Solver

The primary method of solving forward problems in TensorDiffEq is the collocation solver. This methodology identifies points
in the domain of the problem and collocates them to the solution via a loss function. Therefore, this is a natural application
for a neural network function approximation.

##### Instantiate the Model

The `CollocationSolverND()` solver can be initialized in the following way:

```{code} python
CollocationSolverND(assimilate=False)
```

Args:
- `assimilate` - a `bool` that describes whether the `CollocationSolverND` will be used for data assimilation

Note that very little in the solver is truly initialized when creating the `CollocationSolverND` instance, most comes later in the `compile` call.

##### Methods

```{code} python
compile(layer_sizes, f_model, domain,b bcs, isAdaptive=False, col_weights=None, u_weights=None, g=none, dist=False)
```

Args:
- `layer_sizes` - a `list` of `ints` describing the size of the input, hidden, and output layers of the FC MLP network
- `f_model` a `fun` describing the physics of the problem. More info is provided in [this section](../../physics/index.ipynb)

Model compilation is truly where the rubber meets the road in defining an inference model in TensorDiffEq. We compile the model using the `compile` method on the
`CollocationSolverND` method.




```{bibliography} ../../references.bib
:style: unsrt
```