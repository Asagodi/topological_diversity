## Discretization and Euler-Maruyama Method
In the supplementary material we included a note on the discretization.

**Discretize the time variable:** Let \( t_n = n \Delta t \) where \( \Delta t = 1 \) (unit time step).

**Apply the Euler-Maruyama method:** The Euler-Maruyama method for a stochastic differential equation \( \dm{\vx} = a(\vx)\dm{t} + b(\vx)\dm{W} \) is given by:

\[ \vx_{t+1} = \vx_t + a(\vx_t) \Delta t + b(\vx_t) \Delta W_t. \]

**Substitute the drift and diffusion terms into the Euler-Maruyama formula:**

Drift term: \( a(\vx, t) = -\vx + f(\win \vI(t) + \vW \vx + \vb) \)  
Diffusion term: \( b(\vx, t) = \sigma \)

\[ \vx_{t+1} = \vx_t + \left( -\vx_t + f(\win \vI_t + \vW \vx_t + \vb) \right) \Delta t + \sigma \Delta W_t. \]

**Simplify the equation with \(\Delta t = 1\):**

\[
\begin{align}
 \vx_{t+1} &= \vx_t + \left( -\vx_t + f(\win \vI_t + \vW \vx_t + \vb) \right) + \sigma \Delta W_t, \\
 \vx_{t+1} &= f(\win \vI_t + \vW \vx_t + \vb) + \sigma \Delta W_t.
 \end{align}
\]

**Introduce the noise term** \(\zeta_t = \sigma \Delta W_t\), which represents the discrete-time noise term.

Thus, we have derived the discrete-time equation:

\[ \vx_t = f(\win \vI_t + \vW \vx_{t-1} + \vb) + \zeta_t. \]

**Additional Comment:** We would like to further comment on our remark about the dependence of the discretization on the activation function used. To add some context, [1] notes that a simple forward Euler rule is known to be rather inaccurate and unstable for integrating stiff ODE systems. While it is true that the choice of activation function can influence the discretization process, it’s important to note that, in practice, this dependency is usually not problematic.

[1] Monfared, Z., & Durstewitz, D. (2020, November). Transformation of ReLU-based recurrent neural networks from discrete-time to continuous-time. In International Conference on Machine Learning (pp. 6999-7009). PMLR.
