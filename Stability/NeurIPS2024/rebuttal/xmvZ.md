Thank you very much for your thoughtful and constructive review of our manuscript. We are grateful for your positive feedback.
Now, we would like to address the specific weaknesses and questions that were raised.

### Weaknesses:
> their theoretical investigation and results are limited to a few very simple systems, low-dimensional systems [...] 
We respectfully disagree with the stated limitation-- the role of the analysis, and numerical experiments is not to prove the generality of the theory but to illustrate it.
Because of this, we focus on low-dimensional systems which are easier to visualize and arguably are more helpful in developing intuition.
In the updated manuscript, we will emphasize that the theory holds under broad, practically relevant conditions.
1. Specifically, we will add statements that assure that our theory is applicable regardless of the dimensionality of the system and the dimensionality of the invariant manifold.
1. Furthermore, we will revise Theorem 1, to show that normal hyperbolicity is both **sufficient and necessary** for invariant manifolds to persist, see [1].
Please also see the shared reply to all reviewers, where we discuss the scope and the limitations of Theorem 1.

> RNNs with specific activation functions and restrictive settings,
We appreciate the reviewer's insightful comment regarding the use of specific activation functions and restrictive settings in RNNs. In response, we have conducted additional experiments LSTMs and GRUs. The results of these experiments are discussed in the shared rebuttal, see also the supplementary document.

> 1. Under what conditions the perturbation function p(x) induces a bifurcation?
Continuous attractors satisfy the first-order conditions for a local bifurcation; that is, they are equilibria, and their Jacobian linearization possesses a non-trivial subspace with eigenvalues having zero real parts. Consequently, any generic perturbation ($p(x)$) will induce a bifurcation in the system.
For a more comprehensive discussion on this topic, we refer the reviewer to the following references:
Kuznetsov, Y. A., Kuznetsov, I. A., & Kuznetsov, Y. (1998). Elements of applied bifurcation theory (Vol. 112, pp. xx+-591). New York: Springer.
These references provide detailed insights into the conditions under which perturbations lead to bifurcations, supporting our assertion.

>2) What types of (generic) bifurcations can arise from the perturbation p(x)?
We are working to characterize codimension-1 bifurcations for a ring attractor and believe a simple polynomial normal form can be derived. 
Characterizing bifurcations with codimension $n > 2$ is an open problem. 
Perturbations in the neighbourhood of a ring attractor (in $C^1$ topology) will result in no bifurcation, a limit cycle, or a ring slow manifold with fixed points. We propose that in many cases the specific bifurcation in fact is irrelevant, instead, what is imporant is that the continuous attractor persists as an invariant manifold.

> the functions h and g are also not clear 
The essence of Theorem 1 can be restated as follows: **if** the function \( f \) has a normally hyperbolic invariant manifold, **then** there exist vector fields \( h \) and \( g \) that satisfy the conditions for equivalence between the systems defined by Eq.2 and Eqs.3&4.
This means that the existence of these functions is guaranteed under the condition of having a normally hyperbolic invariant manifold, but their explicit forms depend on the specific problem at hand.

> 1. What does **sufficiently smooth** mean in Theorem 1?
Please note, that a center manifold is not necessarily unique, and generally is local in **both** space and time ($J \subset \mathcal{X} \times T$).
Stability, or invariance under the flow therefore generally cannot be analyzed using these methods.


>However, discontinuity-induced bifurcations cannot be examined in the same way, as there is no slow manifold in these cases.
We will clarify in the text that discontinuous systems are not considered in our theory. Since all theoretical models and activation functions for RNNs are at least continuous and piecewise smooth, we believe this limitation still encompasses a significant portion of dynamical systems relevant for theoretical neuroscience, making our theory broadly applicable.



>1. It is unclear under what conditions RNN dynamics can be decomposed into slow-fast form to which we can apply Theorem 1.
Theorem 1 holds for all dynamical systems that have a normally hyperbolic continuous attractor. 
For example, RNNs with a ReLU activation functions can only have normally hyperbolic continuous attractors. This is because at any point the vector field that they define either is zero or has a linear terms (ReLU RNNs are piecewise linear systems).
The continuous attractors and their approximations that we discuss are all normally hyperbolic.
In fact, there is a huge benefit from having normal hyperbolicity as it can counteract state noise.



> 1. line 213 [...]
The transformation from the continuous-time RNN described by Eq. (6) to the discrete-time RNN in Eq. (7) is indeed independent on the function \( f \) and the matrix \( W \).
However, it is important to note that the discretization process can result in significantly different system behavior depending on the activation functions used. For instance, in the case of Filippov systems, discretization can introduce instabilities that are not present in the continuous-time system.




> 1. In S4, the sentence "All such [...]
We will reformulate it as "all such perturbations leave the geometry of the continuous attractor intact as an attractive invariant slow manifold, i.e. the parts where the fixed points disappear a slow flow appears."
The preservation of the invariant manifold under perturbations is a direct consequence of the normal hyperbolicity condition described in Theorem 1.
This means that for a normally hyperbolic continuous attractor, any small perturbation will bifurcate away the continuum of fixed points but there will remain an attractive invariant manifold.
In regions where fixed points disappear due to the perturbation, the dynamics will form a slow flow.



**Questions:**
%%%%%
>How does the selection of a specific threshold value influence the identification and characterization of slow manifolds in neural networks with continuous attractors as discussed in the first lines of section 4.2?
[We can include a plot that shows Hausdorff distance between identifies invariant manifolds as a function of the perturbation parameter?]


> Could you elaborate on how different threshold settings impact the dynamics of network states and the emergence of persistent manifolds?
It is unclear to which threshold the reviewer is referring. However, we can clarify the general impact of thresholds on the dynamics of network states and the emergence of persistent manifolds.
We demonstrate that all systems with a sufficiently good generalization property (in our case, defined as networks with NMSE lower than -20 dB) must have a normally hyperbolic invariant manifold that approximates a continuous attractor. According to the theory, these manifolds are necessarily persistent.
The persistence of these manifolds is a direct consequence of their normal hyperbolicity, which ensures that small perturbations do not destroy the manifold but may alter its structure. This property guarantees that the overall geometry of the continuous attractor persists.



**Limitations:**

> The authors have discussed most limitations [...]
We appreciate the reviewer's suggestion to make the limitations of our analysis more explicit. In response, we have included a dedicated Limitations subsection in the discussion section of the manuscript. Please refer to the shared rebuttal for further details.



[1] Mané, R. (1978). Persistent manifolds are normally hyperbolic. Transactions of the American Mathematical Society, 246, 261-283.