# Discovery Model Example

Here we will attempt to learn the parameters provided in [this example](../../model/compiling-example/) (i.e. .0001 and 5.0) from the 
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

