# Initial Conditions

Paramount to any IVP are the initial conditions on which your problem is based. If your problem is temporal, i.e. evolving with time
(as is the case with most PDEs) the IVP is required to discern the exact solution for the problem you are interested in.
The same is true when solving a PDE via a collocation PINN solver. Here we will discuss how to implement initial condition functions
in TensorDiffEq.


