We thank the reviewer for the additional comments.


## Bifurcation and stability analysis
> Bifurcation and stability analysis in Appendix (S2) are not only restricted to a low-dimensional system but also to a system with specific parameters

### Analytically tractable examples
We would first like to clarify that in Supplementary Sections 1 and 2 that the provided examples are illustrative rather than exhaustive.
Recognizing that the supplementary material previously lacked clear structure and signposting, we will revise the text to better convey the motivation behind the examples and to provide a clearer explanation of the analysis.
The analysis can furthermore be easily extended to a more general form $W = [w_{11} w_{12}; w_{21} w_{22}]$ that has a bounded line attractor, through a coordinate transformation.


### Extension of bounded line attractor analysis to higher dimensional systems
We agree that this analysis in S2 can be extended to higher dimensions, and we will incorporate this remark into the supplementary text.
An extension of these results of a low-dimensional system can be easily achieved by 'addition' of dimensions that have an attractive flow normal to the low-dimensional continuous attractor or invariant manifold.



More generally, the results from a low dimensional system can indeed be extended to higher dimensional systems through reduction methods from center manifold theory.
On the center manifold the singular perturbation problem (as is the case for continuous attractors) restricts to a regular perturbation problem [1a].
Furthermore, relying on the Reduction Principle [2a], one can always reduce all systems (independent of dimension) to the same canonical form, given that they have diffeomorphic invariant manifold. We thank the reviewer for pointing this out and will add a remark on this possibility to extend results. 





## Perturbation and dimensionality
> Regarding the perturbation ...
We have now changed the statement in Sec.3.1 to be $l\\neq 0$ for clarity.


## Discretization of SDE
We appreciate the suggestion to better explain the discretization procedure, i.e., the steps for going from Eq.(7) to Eq.(6).


### Steps
In the supplementary material we will include a note on the discretization, which goes as follows.

**Discretize the time variable:** Let $t_n = n \\Delta t$.

The Euler-Maruyama method for a stochastic differential equation $$\\mathrm{d}{\\mathbf{x}} = \\left(-\\mathbf{x} + f(\\mathbf{W}_{\\text{in}} \\mathbf{I}(t) + \\mathbf{W} \\mathbf{x} + \\mathbf{b})\\right)\\mathrm{d}{t} + \\sigma\\mathrm{d}{W}_t$$ is given by :
$$\\mathbf{x}_{n+1} = \\mathbf{x}_n + \\left( -\\mathbf{x}_n + f(\\mathbf{W}_{\\text{in}} \\mathbf{I}_n + \\mathbf{W} \\mathbf{x}_n + \\mathbf{b}) \\right) \\Delta t + \\sigma \\Delta W_n,$$
with $\\Delta W_{n}=W_{(n+1)\\Delta t}-W_{n\\Delta t}\\sim \\mathcal{N}(0,\\Delta t).$

**Subsitute $\\Delta t = 1$:**
$$
\\begin{aligned}
 \\mathbf{x}_{t+1} &= \\mathbf{x}_t + \\left( -\\mathbf{x}_t + f(\\mathbf{W}_{\\text{in}} \\mathbf{I}_t + \\mathbf{W} \\mathbf{x}_t + \\mathbf{b}) \\right) + \\sigma \\Delta W_t, \\\\
 &= f(\\mathbf{W}_{\\text{in}} \\mathbf{I}_t + \\mathbf{W} \\mathbf{x}_t + \\mathbf{b}) + \\sigma \\Delta W_t.
 \\end{aligned}
$$

**Introduce the noise term** $\\zeta_t = \\sigma \\Delta W_t$, which represents the discrete-time noise term.
Thus, we have derived the discrete-time equation:

$$\\mathbf{x}_t = f(\\mathbf{W}_{\\text{in}} \\mathbf{I}_t + \\mathbf{W} \\mathbf{x}_{t-1} + \\mathbf{b}) + \\zeta_t.$$


<<<<<<< HEAD
We thought that $\\Delta t=1$ would simplify the presentation, however, it seems to be misleading the readers.
In our numerical experiments, we actually used a $\\Delta t<1$. 
We will update our manuscript to include $\\Delta t$ for clarity.
=======
We thought that $\Delta t=1$ would simplify the presentation, however, it seems to be misleading the readers.
In our numerical experiments, we actually used a $\Delta t<1$.
We will update our manuscript to include $\Delta t$ for clarity.
>>>>>>> 42d65274e9935cf89c131ee42807c92a179edd47



### Integration scheme
Numerical integration of a stochastic differential equation is an extensive field by itself [3a].
We chose to use the simplest Euler-Maruyama discretization form, because this leads to the standard RNN form, even though it is inferior to other methods.
Computational neuroscientists often train RNNs as models of neural computation [4a,5a]
and interpret them as dynamical systems.
Our experiments connects to existing literature.
In future studies, it would be interesting to perform experiments with Neural SDEs [6a].







[1a] Fenichel, N. (1979). Geometric singular perturbation theory for ordinary differential equations. Journal of differential equations, 31(1), 53-98.

[2a] Kirchgraber, U., & Palmer, K. J. (1990). Geometry in the neighborhood of invariant manifolds of maps and flows and linearization. (No Title).

[3a] https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/

[4a] Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. nature, 503(7474), 78-84.

[5a] Sussillo, D., & Barak, O. (2013). Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. Neural computation, 25(3), 626-649.

[6a] Tzen, B., & Raginsky, M. (2019). Neural stochastic differential equations: Deep latent gaussian models in the diffusion limit. arXiv preprint arXiv:1905.09883.



