# Boundary Conditions

Currently, TensorDiffEq contains built-in support for Dirichlet and Periodic BCs, with expanded support coming in the near future

The boundary conditions described generate additional terms in the loss function to allow for enforcement of those boundaries in the final
solution. Therefore, one simple needs to describe all the boundaries in your problem, put them into a `list` so that TensorDiffEqs solvers can iterate
through them, and allow the solution to train.

Boundary conditions have similar structure, and dont return anything once initialized. Once fed into the solver, however,
the boundary conditions described by the user are enforced in the solution.

### Dirichlet BCs

Dirichlet BCs are initialized in the following fashion:

```{code-block} python
dirichletBC(domain, val, var, target)
```

Args:
- `domain` - a `domain` object containing the variables in the domain
- `val` - a `float` containing the value to be enforced at the boundary
- `var` - a `str` indicating which variable should be enforced by the value in `val`
- `target` - a `str` indicating whether the value listed in `val` will be targeting the `upper` or `lower` boundary on `var`

to create a simple dirichletBC, one could define upper and lower boundary values on the `x` value from the [IC definition section](../ic/index.ipynb)

```{code-block} python
upper_x = dirichlectBC(Domain, val=0.0, var='x', target="upper")
lower_x = dirichlectBC(Domain, val=0.0, var='x', target="lower")
```

This will force the upper and lower boundary value of 0.0 on the upper and lower boundaries of `x` for a 1D spatio-temporal problem.

```{note}
Currently TensorDiffEq doesn't support functions as Dirichlet BCs. In the near future this will be a feature and will contain a similar
interface as the [IC function definition](../ic/index.ipynb). Bear with us as we add features to help you solve your problems!
```

### Periodic BCs

Periodic BCs in TensorDiffEq allow for fine-grain control over the depth to which the derivatives at your boundaries go.
Many periodic BC implementations in other PINN solvers only allow for the zero-order derivative (no derivative) as a member of the loss function in
their solvers. TensorDiffEq allows for arbitrary depth and fine-grain control over which derivatives are included in the
final calculations. We first define a derivative model as such:

```{code-block} python
def deriv_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    return u, u_x
```

which solves down to the first order level at the boundary, allowing for added continuity. However, we aren't limited to only one level of derivative
in the periodic BC, and can instead define an arbitrary amount, such as a 4th-order derivative:

```{code-block} python
def deriv_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_xxx = tf.gradients(u_xx, x)[0]
    u_xxxx = tf.gradients(u_xxx, x)[0]
    return u, u_x, u_xx, u_xxx, u_xxxx
```

A similar form can be used to define higher-order derivatives at boundaries in higher dimensions as well, such as the
following to define a periodic BC on a 2D domain:

```{code-block} python
def deriv_model(u_model, x, y, t):
    u = u_model(tf.concat([x, y, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yx = tf.gradients(u_y, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    u_xy = tf.gradients(u_x, y)[0]
    return u, u_x, u_y, u_xx, u_yy, u_xy, u_yx
```

All the items listed in the `return` will be included iteratively in the loss function of your PINN solver, and adding higher order
derivatives adds little to no additional computational cost.
