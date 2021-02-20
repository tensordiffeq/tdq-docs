# Discovery Model Example

Here we will attempt to learn the parameters provided in [this example](../../model/compiling-example/index.md) (i.e. .0001 and 5.0) from the 
data provided. We consider that data, in this case, to be "experimental" data, however we do know it to be high-fidelity data from a  simulation of the 
AC PDE system. 

First, we present the whole script to train a discovery model, then we will break it apart and examine the major chunks. 

## Full Example
```{code} python 
# Put params into a list
params = [tf.Variable(0.0, dtype=tf.float32), tf.Variable(0.0, dtype=tf.float32)]


# Define f_model, note the `vars` argument. Inputs must follow this order!
def f_model(u_model, var, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    c1 = var[0]  # tunable param 1
    c2 = var[1]  # tunable param 2
    f_u = u_t - c1 * u_xx + c2 * u * u * u - c2 * u
    return f_u


# Import data, same data as Raissi et al

data = scipy.io.loadmat('AC.mat')

t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = data['uu']
Exact_u = np.real(Exact)

# generate all combinations of x and t
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]

x = X_star[:, 0:1]
t = X_star[:, 1:2]

print(np.shape(x))
# append to a list for input to model.fit
X = [x, t]

# define MLP depth and layer width
layer_sizes = [2, 128, 128, 128, 128, 1]

# initialize, compile, train model
model = DiscoveryModel()
model.compile(layer_sizes, f_model, X, u_star, params) 

# train loop
model.fit(tf_iter=10000)

```

Let's break this apart and look at its pieces.

### Defining parameters and `f_model` for estimation
First we define the `tf.Variable` objects for the parameters and the new `f_model`. Note that the structure and syntax is largely the same as the [`CollocationSolverND` example](../../model/compiling-example/index.md), with a few notable exceptions.

```{code} python
# Put params into a list
params = [tf.Variable(0.0, dtype=tf.float32), tf.Variable(0.0, dtype=tf.float32)]


# Define f_model, note the `vars` argument. Inputs must follow this order!
def f_model(u_model, var, x, t):
    u = u_model(tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    c1 = var[0]  # tunable param 1
    c2 = var[1]  # tunable param 2
    f_u = u_t - c1 * u_xx + c2 * u * u * u - c2 * u
    return f_u
```

Above, we can see that the parameters must be `tf.Variables,` initialized as above, with a `tf.float32` data type. 
You must initialize as many of these as there are parameters to estimate. Those variables 
must then be added to a `list` for training at a later step. Here we initialize the parameters to `0.0` to start. As a heuristic, this 
works well since the $u$ network also needs to get somewhat close to a `0.0` residual solution before the trianing of the 
parameters can really start to take root.

Concurrently, we generate the new `f_model`. As discussed earlier, the new `f_model` contains an additional input from its 
[`CollocationSolverND` cousin](../../model/compiling-example/index.md) - the `var` input. This input is where the `list` of 
`tf.Variables` goes. Inside the `f_model` definition, that list is then partitioned out piecewise into the PDE. This allows the 
tensorflow tracing to reach into the `f_model` function and backpropagate against those values, resulting in training of the parameters 
as well as the $u$ network itself.

### Importing data and generating input
```{code} python
# Import data, same data as Raissi et al
data = scipy.io.loadmat('AC.mat')

t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = data['uu']
Exact_u = np.real(Exact)

# generate all combinations of x and t
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]

x = X_star[:, 0:1]
t = X_star[:, 1:2]

# append to a list for input to model.fit
X = [x, t]
```

In this case, the input `x` and `t` sequences were held in the file. These took a similar form to [`np.linspace` objects](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html), i.e. were vectors of even spacing across the `x` and `t` 
dimensions, independently. Therefore, we needed to generate all possible combinations 
of `x` and `t` to use these. If you are in need of a multidimensional meshgrid generator (past 2D) that returns a list of all possible combinations
of `np.linepace` type arrays, check out [this github gist](https://gist.github.com/levimcclenny/e87dd0979e339ea89a9885ec05fe7c10) to get the input in the format tensordiffeq requires. This multimesh generation is 
included in tdq base and is available by combining the function `multimesh` with `flatten_and_stack`, both in `tensordiffeq.utils`. Note that you need 
an `X, u_sol` pair for each of your data points. So, if you have a 1D (with time) problem, then you need an input pair that of the form `[x,t]` and 
a target `u_sol` value. Essentially, we are performing supervised learning of the parameters, therefore we need some target value for each input coordinate in 
the domain where we have data available. 


```{code} python 
# define MLP depth and layer width
layer_sizes = [2, 128, 128, 128, 128, 1]
```
