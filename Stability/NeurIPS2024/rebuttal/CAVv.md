### Weaknesses
> 1. The experiments and associated analyses focus solely on networks that approximate 1D ring attractors.
> This is quite simplistic, and at least for the numerical expreiments, the authors could consider tasks like navigation where a planar 2D attractor is approximated by the networks.

We appreciate the remark, and shall include additional tasks where the approximate continuous attractor is of higher dimension.
That being said, we would like to reiterate that the primary contribution is theoretical and the numerical experiments are meant to illustrate the theory.
Regarding navigation tasks, two points bear mentioning.
1. For planar attractors are diffeomorphic to $R^2$, note that they do not conform to the assumptions on normally hyperbolic invariant manifolds, since $R^2$ isn't compact.
There are suitable generalizations of this theory to noncompact manifolds [1], but we do not pursue them since they require more refined tools, which would only obscure the point that we are trying to make.
Tangentially, we would also like to point out that we assume that neural dynamics are naturally bounded (e.g. by energy constraints) and hence sufficiently well described by compact invariant manifolds.
1. Approximations of planar attractors that are diffeomorphic to $[0,1]\times[0,1]$ are also excluded since they’re an example of a **manifold with corners**.

In the revised version of the manuscript, we will include the above limitations and provide reference to [2].




> The authors have only qualitatively characterized the variations in the topologies of the networks. It is perhaps possible to quantitatively characterize this by using Dynamical Similarity Analysis [1] on various trained networks.

We thank the reviewer for pointing out the reference; we applied DSA to our numerical results.
Our preliminary observations are that DSA reflects the fact that the geometry of the invariant manifold is preserved, but it cannot detect the emergence of fixed-points and saddles on the pertubed manifold.
The DSA values clustered around two points regardless of the number of fixed points.
<!-- How to discuss DSA giving near zero ds score for networks trained on different tasks?-->
This appears to be consistent with the results reported in the referenced paper, c.f. Figure 4 shows a gradual increase in DSA as $\alpha \to 1$ despite having a bifurcation at $\alpha = 1$.

Lastly, we would like to note that the analysis using DSA cannot be trivially automated. As pointed out by the authors of DSA:
1. The DSA 'score' is relative; one needs to compare different dynamics.
1. DSA essentially requires 'learning' or fitting a separate model, which implicitly requires performing model selection with respect to the delay embedding, rank of the linear operator.

For these reasons, we would like to adhere to the spirit of our initial analyses.



> For the generalization analysis, the authors could evaluate generalization performance by the nature/type of the approximate attractor as well.

We looked at the generalization performance by the nature/type of the approximate attractor (Fig.5D MSE vs number of fixed points).
>Furthermore, although I may have missed this, could the authors comment on what networks hyperparameters lead to which approximations?
The only networks hyperparameters that we varied were the nonlinearity and the size.

> The figures and presentation could be improved [...]
We appreciate the comments, and changed the manuscript accordingly.


> Overall, the writing could be improved in several places to improve clarity.

We improved the writing, focusing on overall clarity.


>For example, the conclusions of the generalization analysis and their implications are not very clear, and how this connects to the various types of approximate attractors is not clear (related to W3).

In the revised version, we will make a stronger point that connects the inherent slowness of the invariant manifold to the generalizability of the approximate solutions.
We also added a longer description of the implications of our numerical experiments to the main text.

### Questions:

> How do the authors identify the various kinds of approximations of the attractors? Can this be automated, perhaps by using to DSA to cluster the various types?

We identify approximations by their (1) attractive invariant manifold (as motivated by the theory) and (2) asymptotic behavior (as motivated by our analysis of perturbations and approximations of ring attractors).
The invariant manifold in our examples typically take the structure of a ring with fixed points and trasient trajectories on it.
We find the fixed points and their stabilities by identifying where the flow reverses by sampling the direction of the local flow for 1024 sample points along the found invariant manifold.
The only example we found that is of another type is the attractive torus (Fig.4D).
For this network, instead of finding the fixed points, we identifies stable limit cycles where there was a recurrence of the simulated trajectories, i.e., where the flow returned back to an initial chosen number of time steps (up to a distance of 10^{-4}).



For the difficulties of using DSA, see above.

> At what level of performance are all trained networks compared? Are they all trained until the same loss value and how close is this MSE to 0?

All networks are trained for 5000 gradient steps.
We exclude those networks from the analysis that are performing less than -20dB in terms of normalized mean squared error.



[2] Eldering, J. (2013). Normally hyperbolic invariant manifolds: the noncompact case (Vol. 2). Atlantis Press.
