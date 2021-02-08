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
