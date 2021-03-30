
# GPU Considerations

Now lets take some of these examples, make them a little "larger" (by adding more collocation points, for instance), and solve the 
across multiple GPUs. 

An extremely unique feature of TensorDiffEq is that the exact same code that implements a solver on a small scale CPU platform 
 can implement a solver on a massive scale. The only major difference is scaling up the number of collocation points and implementing 
a distributed solver in TensorDiffEq. 

Even more powerful - this only requires the modification of one line of code - a single boolean value to be modified. See the below Allen-Cahn example:

## Full example 

```{code} python
Domain = DomainND(["x", "t"], time_var='t')

Domain.add("x", [-1.0, 1.0], 512)
Domain.add("t", [0.0, 1.0], 201)

N_f = 1000000 # 1m collocation points
Domain.generate_collocation_points(N_f)


def func_ic(x):
    return x ** 2 * np.cos(math.pi * x)


# Conditions to be considered at the boundaries for the periodic BC
def deriv_model(u_model, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    return u, u_x

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

layer_sizes = [2, 128, 128, 128, 128, 1]

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs, dist=True)
model.fit(tf_iter=1000)
```

We note here that the only difference in the code comes in the `compile` call, where we add the argument `dist=True`. 
This enables TensorDiffEq to adopt a [`tf.distribute.MirroredStrategy()`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
data-parallelism approach to training. In this case, the collocation points are distributed evenly across all available workers. This is one of 
the most powerful aspects of TensorDiffEq, being able to scale readily without modification of the code. The same physics model, neural network model, and 
optimizer (except for L-BFGS, at the time of this writing) can be run on a small model on a local machine, and can also be scaled up to run on an
enterprise-level data center with $N$ GPUs. 


## Notes and Best-Practices for GPUs
- graph-mode L-BFGS is typically faster on a single-GPU, given sufficient model size. On a CPU, empirically,
it has been demonstrated that eager-mode L-BFGS is actually faster than graph-mode

We are interested in
 community feedback! If you notice something interesting here, open a PR on these docs and let us know!

