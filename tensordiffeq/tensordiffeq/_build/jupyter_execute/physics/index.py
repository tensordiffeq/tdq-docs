# Building the Physics

Physics are paramount to the PINN concept. The fundamental idea of PINNs is that there could be little to no data available to
solve the problem at hand. The solution is to build a model capable of operating in the middle ground - somewhere in between a rote data-driven model and one in which the entire problem
is completely defined. The data requirement for PINNs is drastically reduced from a black-box machine learning model, leaning instead on the
physics of the problem at hand to bridge the gap to identifying a solution.

The fantastic part about PINN solvers, such as those we are building in this section, is that there is no data requirement whatsoever to train
the solution network. The training, in this case, is minimizing the residual between the output of the network as it goes through the PDE we define.
We are simply seeking to push that residual to zero. On the boundaries we do explicitly define solutions, which could be considered as
"data," however these are known quantities when solving any PDE, and are required to generate a unique solution for your problem.

In order to add the physics to a problem in TensorDiffEq, a function describing the relationships between all the partial derivatives
must be generated. In order to get the partials, we take advantage of the `tf.gradients` function, which generates the gradients of a
function with respect to an input. This provides an intuitive interface to build a strong-form PDE to represent the
physics in your model.

### Building Physics in TensorDiffEq

A simple example from [Raissi et. al](https://maziarraissi.github.io/PINNs/) is the viscous Burger's equation, useful for modeling shock waves in
a medium.

To implement the Burger's equation in TensorDiffEq, we define a `f_model` that contains the strong form of the PDE as follows:

```{code-block} python
def f_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1)) # this line is required regardless of the model
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    f_u = u_t + u * u_x - (0.05 / tf.constant(math.pi)) * u_xx
    return f_u
```

the `f_model` is then fed into the solver as-is, with no special processing required.

Defining a higher-dimensional model in TensorDiffEq is just as straightforward, for instance a 2D heat diffusion
equation could be described in the following way:

```{code-block} python
def f_model(u_model, x, y, t):
    u = u_model(tf.concat([x, y, t], 1)) # this line must be modified to include the new dimension `y`
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_y = tf.gradients(u, y)
    u_yy = tf.gradients(u_y, y)
    u_t = tf.gradients(u, t)

    f_u = u_t - c*(u_xx + u_yy)

    return f_u
```


Once you define the physics of the model, you are ready to compile the model and begin training to generate a solution
approximation network for your PDE.

