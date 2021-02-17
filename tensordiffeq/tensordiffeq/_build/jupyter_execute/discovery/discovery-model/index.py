# Discovery Model

Next, we will assess the discovery model, where we can perform parameter estimation of values in a PDE system.
This is commonly referred to as "inverse modeling," as opposed to
forward modeling (or inference) where we provide all the input information to the `f_model` and solve the whole solution via a
neural network function approximation, forcing the residual term of the `f-model` constrained output to `0.0` via an MSE loss.

This case is almost a more natural application of neural networks, as it is a physics-constrained methodology of supervised learning. We have data output from
a simulation or an experiment, and we seek to map parameters of a known PDE system to that data. In this case, the neural network works via a multi-faceted loss
function, where the residual of the `f_network` is being forced to `0.0` via an MSE loss. However, the key difference is that we are also using the estimated parameters to find that loss.
Therefore, we also take the loss wrt the target variables (the estimated parameters, in this case) and optimize those terms via an
MSE loss against the training data. The result is a training methodology that not only trains the $u(\textbf{X}, t)$ solution (or just $u(\textbf{X})$, depending on your problem), but also
estimates parameters within the model concurrently.

In this section we will dig into the `DiscoveryMode()` in TensroDiffEq. We will discuss the pertinent methods, as well as list a [few examples](../discovery-example/index.ipynb)
to get you off the ground in training your models.
