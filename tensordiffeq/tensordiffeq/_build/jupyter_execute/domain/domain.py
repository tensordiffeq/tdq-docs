# Generating a Domain

A `Domain` object is the first essential component of definig a problem in TensorDiffEq. The domain object
contains primatives for defining the problem scope used later in your definitions of boundary conditions,
initial conditions, and eventually to sample collocation points that are fed into your PINN solver.

### Instantiate
```{code-block} python
DomainND(var, time_var = None)
```

Args:
- `var` - a `list` of variables intended for use in the domain
- `time_var` - a `str`. if your problem is temporal or spatiotemporal then list the time variable here

### Methods

###### Adding variables to your domain

In order to build out a domain object, we must define the different variables working together in the problem.
A unique aspect of tensorDiffEq is that it is dimension-agnostic - there is no limit to the number of dimensions
you can add to the problem.

Usage:
```{code-block} python
add(token, vals, fidel)
```


Args:
- `token` - A `str` by which the variable will be referenced, usually a dimension of the problem such as
`"x"` or `"y"`
- `vals` - a `list` of inputs corresponding to `[min, max]` of the target domain
- `fidel` - An `int` defining the level of fidelity of the evenly spaced samples along this simensions boundary points

```{note}
TensorDiffEq uses *meshless* solvers, i.e. the domain is not solved using evenly spaced meshs across the domain, as in FEA.
The `fidel` metric defined here is to facilitate generation of training points for training the solution at the boundaries of your domain.
```

Example:
Usage:
```{code-block} python
Domain = DomainND(['x', 't'], time_var = 't')
Domain.add('x', [-1.0, 1.0], 256)
Domain.add('t', [0.0, 1.0], 101)
```



Usage:
```{code-block} python
generate_collocation_points(N_f)
```
