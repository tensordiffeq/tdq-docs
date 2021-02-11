# Self-Adaptive PINN Example

Next, let's jump into a [Self-Adaptive PINN](https://arxiv.org/pdf/2009.04544.pdf) example, where we demonstrate some of the capabilities of the self-adaptive training.
You may notice that the interface doesn't change too much, and all we need to do is define weight vectors in the form of
tf.Variables for the collocation initial condition weights.

A full example is shown below for the Allen-Cahn PDE:

```{code} python
Domain = DomainND(["x", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 512)
Domain.add("t", [0.0, 1.0], 201)

N_f = 50000
Domain.generate_collocation_points(N_f)


def func_ic(x):
    return x ** 2 * np.cos(math.pi * x)


# Conditions to be considered at the boundaries for the periodic BC
def deriv_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_xxx = tf.gradients(u_xx, x)[0]
    u_xxxx = tf.gradients(u_xxx, x)[0]
    return u, u_x, u_xxx, u_xxxx


init = IC(Domain, [func_ic], var=[['x']])
x_periodic = periodicBC(Domain, ['x'], [deriv_model])

BCs = [init, x_periodic]


def f_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    c1 = tdq.utils.constant(.0001)
    c2 = tdq.utils.constant(5.0)
    f_u = u_t - c1 * u_xx + c2 * u * u * u - c2 * u
    return f_u


col_weights = tf.Variable(tf.random.uniform([N_f, 1]), trainable=True, dtype=tf.float32)
u_weights = tf.Variable(100 * tf.random.uniform([512, 1]), trainable=True, dtype=tf.float32)

layer_sizes = [2, 128, 128, 128, 128, 1]

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs, isAdaptive=True,
                col_weights=col_weights, u_weights=u_weights)
model.fit(tf_iter=10000, newton_iter=10000)

```

Lets break this script up and discuss it a bit. First we define the `domain` and everything associated in it, in this case
we have a problems that is only dependent on `x` and `t`.

```{code} python
Domain = DomainND(["x", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 512)
Domain.add("t", [0.0, 1.0], 201)

N_f = 50000
Domain.generate_collocation_points(N_f)
```

Notice how this problem we take more collocation points than the [last example](../compiling-example/index.ipynb) with its simpler
example.

Next up lets take a look at defining the [initial condition](../../ic-bc/ic/index.ipynb) and the
[periodic BC derivative model](../../ic-bc/bc/index.ipynb#derivative-models). Then we drop those conditions into a list to drop them
into the solver.

```{code} python
def func_ic(x):
    return x ** 2 * np.cos(math.pi * x)


# Conditions to be considered at the boundaries for the periodic BC
def deriv_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_xxx = tf.gradients(u_xx, x)[0]
    u_xxxx = tf.gradients(u_xxx, x)[0]
    return u, u_x, u_xxx, u_xxxx


init = IC(Domain, [func_ic], var=[['x']])
x_periodic = periodicBC(Domain, ['x'], [deriv_model])

BCs = [init, x_periodic]
```