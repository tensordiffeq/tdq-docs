# Discovery Model

Next, we will assess the discovery model, where we can perform parameter estimation of values in a PDE system.
This is commonly referred to as "inverse modeling," as opposed to
forward modeling (or inference) where we provide all the input information to the `f_model` and solve the whole solution via a
neural network function approximation, forcing the residual term of the `f_model` constrained output to `0.0` via an MSE loss.

This case is almost a more natural application of neural networks, as it is a physics-constrained methodology of supervised learning. We have data output from
a simulation or an experiment, and we seek to map parameters of a known PDE system to that data. In this case, the neural network works via a multi-faceted loss
function, where the residual of the `f_network` is being forced to `0.0` via an MSE loss. However, the key difference is that we are also using the estimated parameters to find that loss.
Therefore, we also take the loss wrt the target variables (the estimated parameters, in this case) and optimize those terms via an
MSE loss against the training data. The result is a training methodology that not only trains the $u(\textbf{X}, t)$ solution (or just $u(\textbf{X})$, depending on your problem), but also
estimates parameters within the model concurrently.

In this section we will dig into the `DiscoveryMode()` in TensorDiffEq. We will discuss the pertinent methods, as well as list a [few examples](../discovery-example/index.ipynb)
to get you off the ground in training your models.

## Initialize

```{code} python
DiscoveryModel()
```

The `DiscoveryModel` class initialized without any arguments.

## Methods

The `DiscoveryModel` class has methods to pass in the data for the problem, as well as methods to fit and predict similar to the `CollocationModelND()` class.

### Compiling the Model

First we must feed in the data and fitting parameters using the `compile` method.

```{code} python
compile(layer_sizes, f_model, X, u_star, var, col_weights=None)
```

Args:
- `layer_sizes` - a `list` of `ints` describing the width and depth of your MLP network used for approximation. See
[here](../../model/compiling/index.html#layer-sizes) for more information
- `f_model` - a `func` describing the physics model. The `f_model` for a `DiscoveryModel` must contain the input variable `vars_`
as the *second* input, before the input variables to the PDE system. See the example [here](../discovery-example/index.html). The definition
of the variables themselves will be in the `var` input
- `X` - a `list` of `[N,1]` arrays of input data, one for each variable (i.e. a list of `[x,t]` where `x` and `t` are `[N,1]` arrays)
- `u_star` - an `array` containing the exact solution data, with at point to point correlation to the data input to `X`
- `var` - a `list` of `tf.Variables`, preinitialized, for training the parameters you are interested in in your model
- `col_weights` - a `tf.Variable` array of size `[N,1]` for collocation weights for each data point in the experimental data

Once compiled, the model is ready to begin training the parameters in `var` to fit the data presented in the `[X, u_star]` pairs. As mentioned above,
this training is performed concurrently to training a $u(\textbf{X}, t)$ network for the problem presented in `f_model`.

### Fitting the model

Once compiled, we can fit the model.

```{code} python
fit(tf_itter)
```

Args:
- `tf_iter` - the number of iterations of training to be completed by the optimizer. Optimizer defaults to `tf.Keras.Optimizers.Adam(.005)` but can be [modified](../../advanced/index.md). 


