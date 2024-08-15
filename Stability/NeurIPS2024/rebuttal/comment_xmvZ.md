We thank the reviewer for the additional comments.


## Bifurcation and stability analysis
> Bifurcation and stability analysis in Appendix (S2) are not only restricted to a low-dimensional system but also to a system with specific parameters

### Analytically tractable examples
We would first like to clarify that in Supplementary Sections 1 and 2 the provided examples are illustrative rather than exhaustive.
Recognizing that the supplementary material previously lacked clear structure and signposting, we will revise the text to better convey the motivation behind the examples and to provide a clearer explanation of the analysis.
The analysis can furthermore be easily extended to a more general form $W = [w\\_{11} w\\_{12}; w\\_{21} w\\_{22}]$ that has a bounded line attractor, through a coordinate transformation.


### Extension of bounded line attractor analysis to higher dimensional systems
We agree that this analysis in S2 can be extended to higher dimensions, and we will incorporate this remark into the supplementary text.
An extension of these results of a low-dimensional system can be easily achieved by the 'addition' of dimensions that have an attractive flow normal to the low-dimensional continuous attractor or invariant manifold.


More generally, the results from a low-dimensional system can indeed be extended to higher-dimensional systems through reduction methods from center manifold theory.
On the center manifold the singular perturbation problem (as is the case for continuous attractors) restricts to a regular perturbation problem [1a].
Furthermore, relying on the Reduction Principle [2a], one can always reduce all systems (independent of dimension) to the same canonical form, given that they have diffeomorphic invariant manifold. We thank the reviewer for pointing this out and will add a remark on this possibility to extend results.





## Perturbation and dimensionality
> Regarding the perturbation ...
We understand how that may have lead to confusion. We changed the statement in Sec.3.1 to be $l\\\neq 0$ for clarity.


## Discretization of SDE
We appreciate the suggestion to better explain the discretization procedure, i.e., the steps for going from Eq.(7) to Eq.(6).


### Steps
In the supplementary material, we will include a note on the discretization, which goes as follows.

**Discretize the time variable:** Let $t\\_n = n \\\Delta t$.

The Euler-Maruyama method for a stochastic differential equation

$$\\mathrm{d}{\\mathbf{x}} = (-\\mathbf{x} + f( \\mathbf{W}\\_{\\text{in}} \\mathbf{I}(t) + \\mathbf{W} \\mathbf{x} + \\mathbf{b} )) \\mathrm{d}{t} + \\sigma\\mathrm{d}{W}\\_t$$

is given by :

$$\\mathbf{x}\\_{n+1} = \\mathbf{x}\\_n + ( - \\mathbf{x}\\_n + f ( \\mathbf{W}\\_{\\text{in}} \\mathbf{I}\\_n + \\mathbf{W} \\mathbf{x}\\_{n} + \\mathbf{b} ) ) \\Delta t + \\sigma \\Delta W\\_{n}$$

with $\\Delta W\\_{n}=W\\_{(n+1)\\Delta t}-W\\_{n\\Delta t}\\sim \\mathcal{N}(0,\\Delta t).$

**Subsitute $\\Delta t = 1$:**

$$\\begin{aligned}
 \\mathbf{x}\\_{t+1} &= \\mathbf{x}\\_t + ( -\\mathbf{x}\\_t + f(\\mathbf{W}\\_{\\text{in}} \\mathbf{I}\\_t + \\mathbf{W} \\mathbf{x}\\_t + \\mathbf{b}) ) + \\sigma \\Delta W\\_t, \\\\
 &= f(\\mathbf{W}\\_{\\text{in}} \\mathbf{I}\\_t + \\mathbf{W} \\mathbf{x}\\_t + \\mathbf{b}) + \\sigma \\Delta W\\_t.
 \\end{aligned}
$$

**Introduce the noise term** $\\zeta\\_t = \\sigma \\Delta W\\_t$, which represents the discrete-time noise term.
Thus, we have derived the discrete-time equation:

$$\\\mathbf{x}\\_t = f(\\\mathbf{W}\\_{\\\text{in}} \\\mathbf{I}\\_t + \\\mathbf{W} \\\mathbf{x}\\_{t-1} + \\\mathbf{b}) + \\\zeta\\_t$$


We thought that $\\Delta t=1$ would simplify the presentation, however, it seems to be misleading the readers.
In our numerical experiments, we used a $\\Delta t<1$.
We will update our manuscript to include $\\Delta t$ for clarity.



### Integration scheme
Numerical integration of a stochastic differential equation is an extensive field by itself [3a].
We chose to use the simplest Euler-Maruyama discretization form because this leads to the standard RNN form.
Although it is generally inferior to other methods in terms of efficiency, systems with a fast-slow decomposition are stiff which presents additional challenges in their solution.

Computational neuroscientists often train RNNs as models of neural computation [4a,5a]
and interpret them as dynamical systems.
Our experiments connect to existing literature.
In future studies, it would be interesting to perform experiments with Neural SDEs [6a].







[1a] Fenichel, N. (1979). Geometric singular perturbation theory for ordinary differential equations. Journal of differential equations, 31(1), 53-98.

[2a] Kirchgraber, U., & Palmer, K. J. (1990). Geometry in the neighborhood of invariant manifolds of maps and flows and linearization. (No Title).

[3a] https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/

[4a] Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. nature, 503(7474), 78-84.

[5a] Sussillo, D., & Barak, O. (2013). Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. Neural computation, 25(3), 626-649.

[6a] Tzen, B., & Raginsky, M. (2019). Neural stochastic differential equations: Deep latent gaussian models in the diffusion limit. arXiv preprint arXiv:1905.09883.