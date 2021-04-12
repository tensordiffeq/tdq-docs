# Modification of Keras Optimizers

Due to the fact that TensorDiffEq is based on Keras, we have access to the full swath of optimizers pushed out by the Tensorflow 
team for neural network training. A full list is available [here](https://keras.io/api/optimizers/), and includes:
- SGD
- RMSprop
- Adam
- Adadelta
- Adagrad
- Adamax
- Nadam
- Ftrl

Modification of these optimizers away from the baseline default parameters in TensorDiffEq is relatively straightforward, 
allowing the user to identify which optimizer is best for their specific problem. 

One can modify the optimizers in the `CollocationSolverND` object by either changing the 
`tf_optimizer` object or the `tf_optimizer_weights` object and replacing them with a new instance of 
a `tf.keras.optimizers` object, annotated above. As an example, this is how one could modify the built-in `Adam` optimizer with an 
`SGD` optimizer for training:

```{code} python
model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs)
model.tf_optimizer = tf.keras.optimizers.SGD(lr=.001)
model.fit(tf_iter=2000)
```

Additionally, one could replace the learning rate parameter in the `tf_optimizer` object with 
a different learning rate if that is desired in the default `Adam` optimizer in the same way.

```{code} python
model = CollocationSolverND()
model.compile(layer_sizes, f_model, Domain, BCs)
model.tf_optimizer = tf.keras.optimizers.Adam(lr=.001)
model.fit(tf_iter=2000)
```

It is important to note that replacing the optimizers in this way does not guarantee they will converge. The question
of PINN training stability and convergence isa heavily researched and ongoing conversation. 
