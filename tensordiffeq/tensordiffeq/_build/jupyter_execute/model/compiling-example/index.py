# Collocation Solver Example

Lets walk through a whole script and get an idea of how to solve a simple burgers equation example, similar to
the example seen in [Raissi et. al](https://maziarraissi.github.io/PINNs/).

## Full Script

The full script, up to model training, is below:

```{code} python
Domain = DomainND(["x", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 256)
Domain.add("t", [0.0, 1.0], 100)

N_f = 20000
Domain.generate_collocation_points(N_f)


def func_ic(x):
    return -np.sin(x * math.pi)

init = IC(Domain, [func_ic], var=[['x']])
upper_x = dirichletBC(Domain, val=0.0, var='x', target="upper")
lower_x = dirichletBC(Domain, val=0.0, var='x', target="lower")

BCs = [init, upper_x, lower_x]


def f_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    f_u = u_t + u * u_x - (0.05 / tf.constant(math.pi)) * u_xx
    return f_u


layer_sizes = [2, 128, 128, 128, 128, 1]

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs)
model.fit(tf_iter=1000, newton_iter=1000)
```

## Script Walkthrough

Let's take a look step-by-step.

First we [create the domain](../../physics/index.ipynb) and generate collocation points

```{code} python
Domain = DomainND(["x", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 256)
Domain.add("t", [0.0, 1.0], 100)

N_f = 20000
Domain.generate_collocation_points(N_f)
```

Next we [generate the IC](../../ic-bc/ic/index.ipynb) and the [BCs](../../ic-bc/bc/index.ipynb), and concatenate them into a `list` to
input into the Collocation Solver

```{code} python
def func_ic(x):
    return -np.sin(x * math.pi)

init = IC(Domain, [func_ic], var=[['x']])
upper_x = dirichletBC(Domain, val=0.0, var='x', target="upper")
lower_x = dirichletBC(Domain, val=0.0, var='x', target="lower")

BCs = [init, upper_x, lower_x]
```