Thank you very much for your thoughtful and constructive review of our manuscript. We are grateful for your positive feedback.

### Weaknesses:
>their theoretical investigation and results are limited to a few very simple systems, low-dimensional systems [...] 
We respectfully disagree with the stated limitation-- the role of the analysis, and numerical experiments is not to prove the generality of the theory but to illustrate it. Hence, we focus on visualizable low-dimensional systems and are more helpful in developing intuition.
Nevertheless, we include additional results with RNNs trained on a 2D task.
In the updated manuscript, we emphasize that the theory holds under broad, practically relevant conditions.
1. We added statements that assure that our theory is applicable regardless of the dimensionality or the invariant manifold see shared rebuttal.
1. Furthermore, we will revise Theorem 1, to show that normal hyperbolicity is both **sufficient and necessary** for invariant manifolds to persist, see [1].

> RNNs with specific activation functions and restrictive settings
We appreciate the reviewer's insightful comment. In response, we have conducted additional experiments LSTMs and GRUs, for which the results are included in the supplementary document and discussed in the shared rebuttal.

> 1. Under what conditions the perturbation function p(x) induces a bifurcation?
Continuous attractors satisfy the first-order conditions for a local bifurcation; that is, they are equilibria, and their Jacobian linearization possesses a non-trivial subspace with eigenvalues having zero real parts. Consequently, any generic perturbation $p(x)$ will induce a bifurcation of the system.
For a more comprehensive discussion on this topic, see [2].

>2) What types of (generic) bifurcations can arise from the perturbation p(x)?
We are working to characterize codimension-1 bifurcations for a ring attractor and believe a simple polynomial normal form can be derived. 
Characterizing bifurcations with codimension $n > 2$ is an open problem. 
Perturbations in the neighbourhood of a ring attractor (in $C^1$ topology) will result in no bifurcation, a limit cycle, or a ring slow manifold with fixed points. 

>the functions h and g are also not clear 
The essence of Theorem 1 can be restated as follows: **if** the function $f$ has a normally hyperbolic invariant manifold (NHIM), **then** there exist vector fields $h$ and $g$ that satisfy the conditions for equivalence between the systems defined by Eq.2 and Eqs.3&4.
This means that the existence of these functions is guaranteed under the condition of having a NHIM, but their explicit forms is case specific.

>1. What does sufficiently smooth mean in Theorem 1?
We mean that a system needs to be at least continuous, but some extra conditions apply if a system is not differentiable (discontinuous systems are not considered in our theory). Since all theoretical models and activation functions for RNNs are at least continuous and piecewise smooth, our theory is broadly applicable.
Center manifold are unique, and generally are local in **both** space and time and therefore invariance under the flow generally cannot be analyzed using them.

>1. It is unclear under what conditions RNN dynamics can be decomposed into slow-fast form to which we can apply Theorem 1.
Theorem 1 holds for all dynamical systems that have a normally hyperbolic continuous attractor. For example, RNNs with a ReLU activation functions can only have such continuous attractors. The continuous attractors and their approximations that we discuss are all normally hyperbolic. In fact, there is a huge benefit from having normal hyperbolicity as it can counteract state noise.

> 1. line 213 [...]
The transformation from the continuous-time to the discrete-time is independent on the function $f$ anf the matrix $W$.
However, it is important to note that the discretization process can result in significantly different system behavior depending on the activation functions used. For instance, discretization can introduce dynamics that are not present in the continuous-time system.

>1. In S4, the sentence "All such [...]
We will reformulate it as "all such perturbations leave the geometry of the continuous attractor intact as an attractive invariant slow manifold, i.e. the parts where the fixed points disappear a slow flow appears."
The persistence of the invariant manifold under perturbations is a direct consequence of the normal hyperbolicity condition in Theorem 1.
Therefore, for a normally hyperbolic continuous attractor there will remain an attractive slow invariant manifold.

**Questions:**
>How does the selection of a specific threshold value...
[We can include a plot that shows Hausdorff distance between identifies invariant manifolds as a function of the perturbation parameter?]

>Could you elaborate [...] emergence of persistent manifolds?
It is unclear to which threshold the reviewer is referring. The emergence of persistent manifolds happens under the 4 conditions we discuss.
We demonstrate that all systems with a sufficiently good generalization property (in our case, defined as networks with NMSE lower than -20 dB) must have a NHIM. The persistence of these manifolds is a direct consequence of their normal hyperbolicity.

**Limitations:**
>The authors have discussed most limitations [...]
We appreciate the reviewer's suggestion to make the limitations of our analysis more explicit. In response, we have included a dedicated Limitations subsection in the discussion section of the manuscript. Please refer to the shared rebuttal for further details.

[1] Mané, R. (1978). Persistent manifolds are normally hyperbolic. Transactions of the American Mathematical Society, 246, 261-283.
[2] Kuznetsov, Y. A., Kuznetsov, I. A., & Kuznetsov, Y. (1998). Elements of applied bifurcation theory (Vol. 112, pp. xx+-591). New York: Springer.