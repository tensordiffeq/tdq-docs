# Overview

## In Development *(as of 14 April 2021)*
The fact that TensorDiffEq is built on top of Keras allows for some unique properties when training PINNs.
So far, a (non-exhaustive) list includes:
- easy modification of neural network architecture with ANY Keras layers (whether or not they are useful for your model is left up to you)
- easy modification of optimizers using the `tf.keras.optimizers` bank of solvers
- exporting and re-importing a model for later use or transfer learning

Here we will discuss a few of these options and how to execute these 'hacks'