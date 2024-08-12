We thank the reviewer for the additional comments.


## Bifurcation and stability analysis
> Bifurcation and stability analysis in Appendix (S2) are not only restricted to a low-dimensional system but also to a system with specific parameters

### Analytically tractable examples
We would first like to clarify that in Supplementary Sections 1 and 2 that the provided examples are illustrative rather than exhaustive.
Recognizing that the supplementary material previously lacked clear structure and signposting, we will revise the text to better convey the motivation behind the examples and to provide a clearer explanation of the analysis.
The analysis can furthermore be easily extended to a more general form W = [w11 w12; w21 w22] that has a bounded line attractor, through a coordinate transformation.


### Extension of bounded line attractor analysis to higher dimensional systems
We agree that this analysis in S2 can be extended to higher dimensions, and we will incorporate this remark into the supplementary text.
An extension of these results of a low-dimensional system can be easily echieved by addition of dimensions that have an attractive flow normal to the low-dimensional continuous attractor or invariant manifold. 

In fact, all continuous attractors are center manifolds.
Therefore, we can always analyse a continuous attractor as a center manifold.
However, not all center manifolds are continuous attractors.
So all in all, the results can be extended to higher dimensional systems through reference to the normal hyperbolicity of the involved invariant manifolds.
We believe that using center manifold theory is superfluous to extend results from low- to high-dimensional systems.



## Perturbation and dimensionality
> Regarding the perturbation ...
We have now changed the statement in Sec.3.1 to be $l\neq 0$ for clarity.


## Discretization of SDE
We appreciate the suggestion to better explain the discretization procedure, i.e., the steps for going from Eq.(7) to Eq.(6). 


### Steps
In the supplementary material we will include a note on the discretization, which goes as follows.

**Discretize the time variable:** Let $t_n = n \Delta t$.

The Euler-Maruyama method for a stochastic differential equation $$\mathrm{d}{\mathbf{x}} = \left(-\mathbf{x} + f(\mathbf{W}_{\text{in}} \mathbf{I}(t) + \mathbf{W} \mathbf{x} + \mathbf{b})\right)\mathrm{d}{t} + \sigma\mathrm{d}{W}_t$$ is given by :
$$\mathbf{x}_{n+1} = \mathbf{x}_n + \left( -\mathbf{x}_n + f(\mathbf{W}_{\text{in}} \mathbf{I}_n + \mathbf{W} \mathbf{x}_n + \mathbf{b}) \right) \Delta t + \sigma \Delta W_n,$$
with $\Delta W_{n}=W_{(n+1)\Delta t}-W_{n\Delta t}\sim \mathcal{N}(0,\Delta t).$

**Subsitute $\Delta t = 1$:**
$$
\begin{aligned}
 \mathbf{x}_{t+1} &= \mathbf{x}_t + \left( -\mathbf{x}_t + f(\mathbf{W}_{\text{in}} \mathbf{I}_t + \mathbf{W} \mathbf{x}_t + \mathbf{b}) \right) + \sigma \Delta W_t, \\
 &= f(\mathbf{W}_{\text{in}} \mathbf{I}_t + \mathbf{W} \mathbf{x}_t + \mathbf{b}) + \sigma \Delta W_t.
 \end{aligned}
$$

**Introduce the noise term** $\zeta_t = \sigma \Delta W_t$, which represents the discrete-time noise term.
Thus, we have derived the discrete-time equation:

$$\mathbf{x}_t = f(\mathbf{W}_{\text{in}} \mathbf{I}_t + \mathbf{W} \mathbf{x}_{t-1} + \mathbf{b}) + \zeta_t.$$


We thought that $\Delta t=1$ would simplify the presentation, however, it seems to be misleading the readers.
In our numerical experiments, we actually used a $\Delta t<1$. 
We will update our manuscript to include $\Delta t$ for clarity.



### Integration scheme
Numerical integration of a stochastic differential equation is an extensive field by itself [1a].
We chose to use the simplest Euler-Maruyama discretization form, because this leads to the standard RNN form, even though it is inferior to other methods.
Computational neuroscientists often train RNNs as models of neural computation [2a,3a]
and interpret them as dynamical systems.
Our experiments connects to existing literature.
In future studies, it would be interesting to perform experiments with Neural SDEs [4a].







[1a] Kopell, N. (1996). Global center manifolds and singularly perturbed equations: A brief (and biased) guide to (some of) the literature. Dynamical Systems and Probabilistic Methods in Partial Differential Equations: 1994 Summer Seminar on Dynamical Systems and Probabilistic Methods for Nonlinear Waves, June 20-July 1, 1994, MSRI, Berkeley, CA, 31, 47.

[2a] https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/

[3a] Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. nature, 503(7474), 78-84.

[4a] Sussillo, D., & Barak, O. (2013). Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. Neural computation, 25(3), 626-649.

[5a] Monfared, Z., & Durstewitz, D. (2020). Transformation of ReLU-based recurrent neural networks from discrete-time to continuous-time. In International Conference on Machine Learning (pp. 6999-7009). PMLR.



