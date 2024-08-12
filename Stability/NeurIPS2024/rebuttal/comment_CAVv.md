## Discretization and Euler-Maruyama Method
In the supplementary material we included a note on the discretization, which goes as follows.

**Discretize the time variable:** Let $t_n = n \Delta t$ where $\Delta t = 1$ (unit time step).

The Euler-Maruyama method for a stochastic differential equation $\mathrm{d}{\mathbf{x}} = \left(-\mathbf{x} + f(\mathbf{W}_{\text{in}} \mathbf{I}(t) + \mathbf{W} \mathbf{x} + \mathbf{b})\right)\mathrm{d}{t} + \sigma(\mathbf{x})\mathrm{d}{W}$ is given by:
$$\mathbf{x}_{t+1} = \mathbf{x}_t + \left( -\mathbf{x}_t + f(\mathbf{W}_{\text{in}} \mathbf{I}_t + \mathbf{W} \mathbf{x}_t + \mathbf{b}) \right) \Delta t + \sigma \Delta W_t.$$

**Simplify the equation with $\Delta t = 1$:**
$$
\begin{aligned}
 \mathbf{x}_{t+1} &= \mathbf{x}_t + \left( -\mathbf{x}_t + f(\mathbf{W}_{\text{in}} \mathbf{I}_t + \mathbf{W} \mathbf{x}_t + \mathbf{b}) \right) + \sigma \Delta W_t, \\
 \mathbf{x}_{t+1} &= f(\mathbf{W}_{\text{in}} \mathbf{I}_t + \mathbf{W} \mathbf{x}_t + \mathbf{b}) + \sigma \Delta W_t.
 \end{aligned}
$$

**Introduce the noise term** $\zeta_t = \sigma \Delta W_t$, which represents the discrete-time noise term.
Thus, we have derived the discrete-time equation:

$$\mathbf{x}_t = f(\mathbf{W}_{\text{in}} \mathbf{I}_t + \mathbf{W} \mathbf{x}_{t-1} + \mathbf{b}) + \zeta_t.$$

**Additional Comment:** We would like to further comment on our remark about the dependence of the discretization on the activation function used. It is a well-known fact that a simple forward Euler rule is known to be rather inaccurate and unstable for integrating stiff ODE systems. While it is true that the choice of activation function can influence the discretization process, it’s important to note that, in practice, this dependency is usually not problematic.

We included this form to make the paper relevant to computational neuroscientists that make use of discrete time RNNs for modeling.
Finally, we would like to propose some additional experiments that will show that differences between continuous-time RNNs and their discrete-time counterparts.

[1] Monfared, Z., & Durstewitz, D. (2020). Transformation of ReLU-based recurrent neural networks from discrete-time to continuous-time. In International Conference on Machine Learning (pp. 6999-7009). PMLR.
