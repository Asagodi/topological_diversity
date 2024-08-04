### Weaknesses
> 1. The experiments and associated analyses focus solely on networks that approximate 1D ring attractors.
> This is quite simplistic, and at least for the numerical expreiments, the authors could consider tasks like navigation where a planar 2D attractor is approximated by the networks.

We appreciate the remark, and shall include additional tasks where the approximate continuous attractor is of higher dimension.
That being said, we would like to reiterate that the primary contribution is theoretical and the numerical experiments are meant to illustrate the theory.
Furthermore, planar attractors are diffeomorphic to $R^2$ do not conform to the assumptions on normally hyperbolic invariant manifolds, since $R^2$ isn't compact.
There are suitable generalizations of this theory to noncompact manifolds, but we do not pursue them since they require more refined tools, which would only obscure the point that we are trying to make.
We would also like to point out that we assume that the brain is bounded and hence expect its dynamics to be describable by a compact manifold.
Similarly, approximations of planar attractors are diffeomorphic to $[0,1]\times[0,1]$ are also excluded since they’re an example of a **manifold with corners**.

> The authors have only qualitatively characterized the variations in the topologies of the networks. It is perhaps possible to quantitatively characterize this by using Dynamical Similarity Analysis [1] on various trained networks.

We thank the reviewer for pointing out the reference; we are currently applying DSA to our numerical results.
Our preliminary observations are that DSA reflects the fact that the geometry of the invariant manifold is preserved, but cannot resolve the emergence of fixed-points and saddles on the pertubed manifold.
This appears to be consistent with the results reported in the referenced paper, c.f. Figure 4 shows a gradual increase in DSA as $\alpha \to 1$ despite having a bifurcation at $\alpha = 1$.


> For the generalization analysis, the authors could evaluate generalization performance by the nature/type of the approximate attractor as well. Furthermore, although I may have missed this, could the authors comment on what networks hyperparameters lead to which approximations?

We looked at the generalization performance by the nature/type of the approximate attractor (Fig.5D MSE vs nfps).
The only networks hyperparameters that we varied were the nonlinearity and the size.
In Fig.5D we show to what fixed point topology the different nonlinearities and sizes converge.

> The figures and presentation could be improved:
> 1. On line 107 there is a comment that should be removed ("add link to details").
> 2. Fig. 4C, caption should indicate the nature of the solution found.
> 3. Fig. 5B, y-axis label is missing.
> 4. Fig. 5D, could also show the mean
> - std for classes of networks.
> - Fig. 5E, y-axis label is missing. Also, the authors could just use
> the normalized MSE on the axis could just follow the convention used in
> Fig. 5A instead of using dB.
We appreciate the comments, and will change the manuscript accordingly.


> Overall, the writing could be improved in several places to improve clarity. For example, the conclusions of the generalization analysis and their implications are not very clear, and how this connects to the various types of approximate attractors is not clear (related to W3).

We will revise the writing to improve clarity.


### Questions:

> How do the authors identify the various kinds of approximations of the attractors? Can this be automated, perhaps by using to DSA to cluster the various types?

See above.

> At what level of performance are all trained networks compared? Are they all trained until the same loss value and how close is this MSE to 0?
All networks are trained for 5000 gradient steps.
We exclude those networks from the analysis that are performing less than -20dB in terms of normalized mean squared error.