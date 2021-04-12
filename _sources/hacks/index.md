# Custom Neural Network Architectures

## In Development *(as of 7 April 2021)*
The fact that TensorDiffEq is built on top of Keras allows for some unique properties when training PINNs.
So far, a (non-exhaustive) list includes:
- easy modification of neural network architecture with ANY Keras layers (whether or not they are useful for your model is left up to you)
- easy modification of optimizers using the `tf.keras.optimizers` bank of solvers
- exporting and re-importing a model for later use or transfer learning

Here we will discuss a few of these options and how to execute these 'hacks'

## Modification of Neural Network Architecture 

By default, TensorDiffEq will build a fully-connected network using the layer sizes and lengths you define in 
the `layer sizes` parameter, which is fed into the `model.compile` call. However, once the mode has been compiled,
that network can be overwritten with any Keras neural network. Here we demonstrate how to do so, adding batch norm layers 
to the network. 
Referencing the example [here](../model/compiling-example/index.html), we can modify the neural network as such:


```{code} python
# need to include keras.layers and Sequential API
from tf.keras import layers, Sequential

layer_sizes = [2, 128, 128, 128, 128, 1]

model_bn = tf.keras.Sequential(
    [
        layers.Dense(2, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"),        
            layers.BatchNormalization(),
            layers.Dense(128, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"), 
            layers.BatchNormalization(),
            layers.Dense(128, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"),
            layers.BatchNormalization(),           
            layers.Dense(1, activation=None),
    ]
)

model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs)
# overwrite the default NN with our new one defined above
model.u_model = model_bn
model.fit(tf_iter=1000, newton_iter=1000)
```

This will fit your network with batchnorm as the PDE approximation network, allowing more stability during training and 
reducing the likelihood of vanishing gradients in the training.