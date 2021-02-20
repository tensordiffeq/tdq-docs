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

# define MLP depth and layer width
layer_sizes = [2, 128, 128, 128, 128, 1]

# generate all combinations of x and t
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]

x = X_star[:, 0:1]
t = X_star[:, 1:2]

print(np.shape(x))
# append to a list for input to model.fit
X = [x, t]

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
must then be added to a `list` for training at a later step. 

Concurrently, we generate the new `f_model`. As discussed earlier, the new `f_model` contains an additional input from its 
[`CollocationSolverND` cousin](../../model/compiling-example/index.md) - the `var` input. This input is where the `list` of 
`tf.Variables` goes. Inside the `f_model` definition, that list is then partitioned out piecewise into the PDE. This allows the 
tensorflow tracing to reach into the `f_model` function and backpropogate against those values, resulting in training of the parameters 
as well as the $u$ network itself.