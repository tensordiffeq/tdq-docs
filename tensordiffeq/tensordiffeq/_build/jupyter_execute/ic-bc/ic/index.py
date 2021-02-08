# Initial Conditions

Paramount to any IVP are the initial conditions on which your problem is based. If your problem is temporal, i.e. evolving with time
(as is the case with most PDEs) the initial values are required to discern the exact solution for the problem you are interested in.
The same is true when solving a PDE via a collocation PINN solver. Here we will discuss how to implement initial condition functions
in TensorDiffEq.

##### Defining a function of Domain variables
The primary method of setting an initial condition in TensorDiffEq is to simply define a method that generates the physical function
describing the initial values of your problem. This is a powerful function of TensorDiffEq as it allows for any sort of function, continuous,
discrete, linear, nonlinear, piecewise, etc. This allows for modeling discontinuinities at the initial boundary, giving the user a
wide range of complex possibilities.

In order to build an initial boundary in TensorDiffEq, a python method with inputs corresponding to the variables of the problem `Domain`
can be defined, i.e.

```{code-block} python
def func_ic(x):
    return -np.sin(math.pi * x )
```
This defines an initial condition that is only dependent on `x`, of the form $u(x,0) = -sin(\pi x)$. However, as mentioned, there is no requirement that
the initial condition is a linear or even a continuously differentiable function. One could define a piecewise function in the form below:

```{code-block} python
def func_ic(x):
    if x >= 0.5:
        out = 1.0
     else:
        out = 0.0
    return out
```

The above would be valid input to TensorDiffEqs solvers, and would still allow for self-adaptive solving at the initial boundary.
