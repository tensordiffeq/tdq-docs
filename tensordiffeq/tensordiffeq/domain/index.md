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
- `time_var` - a `str` indicating which variable in the list of `var` is the time variable. If your problem is temporal or spatiotemporal then list the time variable here

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
- `fidel` - An `int` defining the level of fidelity of the evenly spaced samples along this dimension boundary points

```{note}
TensorDiffEq uses *meshless* solvers, i.e. the domain is not solved using evenly spaced meshs across the domain, as in FEA.
The `fidel` metric defined here is to facilitate generation of training points for training the solution at the boundaries of your domain.
```

Example:

```{code-block} python
Domain = DomainND(['x', 't'], time_var = 't')
Domain.add('x', [-1.0, 1.0], 256)
Domain.add('t', [0.0, 1.0], 101)
```


###### Generation of Collocation Points

Collocation points for solving an ND PINN problem are automatically generated using the bounds specified in the `DomainND` object. All you need to do is specify how many you would like.
If you have a large domain and require a lot of collocation points across your domain, TensorDiffEq was designed to handle your problem specifically. More information for solving large problems
with large or very coarse domains are covered later in the section on [GPU best practices]()

```{code-block} python
generate_collocation_points(N_f)
```

Args:
- `N_f` is an `int` describing the number of collocation points desired within the domain defined in your `DomainND` object

Example:
```{code-block} python
Domain = DomainND(['x', 't'], time_var = 't')
Domain.add('x', [-1.0, 1.0], 256)
Domain.add('t', [0.0, 1.0], 101)
Domain.generate_collocation_points(50000)
```

```{note}
The collocation points generated are not returned, they reside in the `DomainND` object. Therefore, one does not need to allocate
the output of `generate_collocation_points` to a variable. Once the `DomainND` object is passed into the solver the collocation points
will be found automatically and used for generating a solution.
```
