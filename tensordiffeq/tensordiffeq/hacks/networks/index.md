# Custom Neural Network Architectures



## Modification of Neural Network Architecture 

By default, TensorDiffEq will build a fully-connected network using the layer sizes and lengths you define in 
the `layer_sizes` parameter, which is fed into the `model.compile` call. However, once the model has been compiled,
that network can be overwritten with any Keras neural network. Here we demonstrate how to do so, adding batch norm layers 
to the network. 
Referencing the example [here](../../model/compiling-example/index.md), we can modify the neural network as such:


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

This will fit your custom network (i.e., with batch norm) as the PDE approximation network, allowing more stability and 
reducing the likelihood of vanishing gradients in the training.