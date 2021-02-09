# Building the Physics

Physics are paramount to the PINN concept. The fundamental idea of PINNs is that there could be little to no data available to
solve the problem at hand. The solution is to bridge the gap between a rote data-driven model and one in which the entire problem
is completely defined. The data requirement for PINNs is drastically reduced from a black-box machine learning model, leaning instead on the
physics of the problem at hand to bridge the gap to identifying a solution.

The fantastic part about PINN solvers, such as those we are building in this section, is that there is no data requirement whatsoever to train
the solution network. The training, in this case, is minimizing the residual between the output of the network as it goes through the PDE we define.
Therefore, we are simply seeking to push that residual to zero. On the boundaries we do explicitly define solutions, which could be considered as
"data," however these are known quantities when solving any PDE, and are required to generate a unique solution for your problem.



