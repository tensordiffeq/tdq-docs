# Generating a Domain

A `Domain` object is the first essential component of definig a problem in TensorDiffEq. The domain object
contains primatives for defining the problem scope used later in your definitions of boundary conditions,
initial conditions, and eventually to sample collocation points that are fed into your PINN solver.

#### Usage:
```{code-block} python
DomainND(var, time_var = None)
```

#### Methods

Usage:
```{code-block} python
add(token, vals, fidel)
```

Args:
- `token` - A `str` by which the varialbe will be referenced, usually a dimension of the problem such as
`"x"` or `"y"`
- `vals` - a `list` of inputs corresponding to `[min, max]` of the target domain
- `fidel` - An `int` defining the level of fidelity of the evenly spaced samples along this simensions boundary points

```{note}
TensorDiffEq uses *meshless* solvers, i.e. the domain is not solved using evenly spaced meshs across the domain, as in FEA.
The `fidel` metric defined here is to allow the generation of the training points for the boundaries in the loss function of the PINN solver
```


where `token` is the token identifier, i.e.

## Markdown + notebooks

As it is markdown, you can embed images, HTML, etc into your posts!

![](https://myst-parser.readthedocs.io/en/latest/_static/logo.png)

You an also $add_{math}$ and

$$
math^{blocks}
$$

or

$$
\begin{aligned}
\mbox{mean} la_{tex} \\ \\
math blocks
\end{aligned}
$$

But make sure you \$Escape \$your \$dollar signs \$you want to keep!

## MyST markdown

MyST markdown works in Jupyter Notebooks as well. For more information about MyST markdown, check
out [the MyST guide in Jupyter Book](https://jupyterbook.org/content/myst.html),
or see [the MyST markdown documentation](https://myst-parser.readthedocs.io/en/latest/).

## Code blocks and outputs

Jupyter Book will also embed your code blocks and output in your book.
For example, here's some sample Matplotlib code:

from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
data = np.array(data).T
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

fig, ax = plt.subplots(figsize=(10, 5))
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot']);

There is a lot more that you can do with outputs (such as including interactive outputs)
with your book. For more information about this, see [the Jupyter Book documentation](https://jupyterbook.org).