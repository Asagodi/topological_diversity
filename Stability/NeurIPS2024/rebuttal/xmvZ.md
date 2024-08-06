Thank you very much for your thoughtful and constructive review of our manuscript. We are grateful for your positive feedback, particularly your recognition of the interesting main idea of our work and its potential significance. We are pleased that you found the connection to Fenichel’s theorem to be noteworthy and that the visualizations in Figure 1 helped convey our overall message. Your appreciation of these aspects is greatly encouraging.

Now, we would like to address the specific weaknesses and questions that were raised.
### Weaknesses:
> 1. They discussed some interesting theoretical techniques (e.g., Theorem 1, Proposition 1) in their study.
>However, their theoretical investigation and results are limited to a few very simple systems, low-dimensional systems either in Section S4 or low-dimensional RNNs with specific activation functions and restrictive settings, i.e., specific parameter values (e.g., equations (1) and (10)).
> The bifurcation analysis of the line attractors and fast-slow decomposition in Section S2 are also studied for very simple systems. Therefore, it is difficult to determine how general their theoretical discussion is and whether it can be applied to investigate and obtain results for more general and high-dimensional cases.

We respectfully disagree with the stated limitation-- the role of the analysis, and numerical experiments is not to prove the generality of the theory but to illustrate it.
Because of this, we focus on low-dimensional systems which are easier to visualize and arguably are more helpful in developing intuition.
In the updated manuscript, we will emphasize that the theory holds under broad, practically relevant conditions.
1. Specifically, we will add statements that assure that our theory is applicable regardless of the dimensionality of the system and the dimensionality of the invariant manifold.
1. Furthermore, we will revise Theorem 1, to show that normal hyperbolicity is both **sufficient and necessary** for invariant manifolds to persist. See [1]

<!-- We believe that the above modifications will convey the very general applicability of the results we're presenting. -->
Please also see the shared reply to all reviewers, where we discuss the scope and limitations of Theorem 1.

> RNNs with specific activation functions and restrictive settings,

We appreciate the reviewer's insightful comment regarding the use of specific activation functions and restrictive settings in RNNs. In response, we have conducted additional experiments with gated RNN architectures, specifically LSTM and GRU models. The results of these experiments have been thoroughly documented and uploaded to OpenReview in a supplementary document for your review.

Our findings indicate that the solutions identified by the trained LSTM and GRU models exhibit similar patterns to those observed in Vanilla RNNs. Specifically, we observed the presence of a normally hyperbolic attractive invariant ring characterized by slow dynamics that eventually converge to fixed points. This consistency across different RNN architectures reinforces the robustness of our initial observations and provides further validation of our theoretical framework.


> 1. In Sect. 3.1, the perturbation p(x) is not clear enough. Specifically, it is unclear:
> 1. Under what conditions the perturbation function p(x) induces a bifurcation?

Continuous attractors satisfy the first-order conditions for a local bifurcation; that is, they are equilibria, and their Jacobian linearization possesses a non-trivial subspace with eigenvalues having zero real parts. Consequently, any generic perturbation ($p(x)$) will induce a bifurcation in the system.

For a more comprehensive discussion on this topic, we refer the reviewer to the following references:
1. Kuznetsov, 2004
1. Robinson, 1999
1. Abraham and Marsden, 2008
These references provide detailed insights into the conditions under which perturbations lead to bifurcations, supporting our assertion.

<!-- 1) For most CANs p(x) almost always induces a bifurcation. (Is it true that the only way to modify the dynamics such that there is not a bifurcation is via changing the level of attractiveness of the the continuous manifold?. In this sense “almost always” = measure zero of parameter space (?) For exact determination maybe Piotr’s ideas) -->
<!-- Piotr: I took a stab at writing it but I get too annoyed to phrase it well. The almost always is a bit different though--- for parametric systems it happens on a dense set of parameters. For the vector fields, I'm guessing a similar statement can be made but I can't recall the precise phrasing of it. -->

>2) What types of (generic) bifurcations can arise from the perturbation p(x)?

We are actively working to characterize codimension-1 bifurcations for a ring attractor and believe that a simple, polynomial normal form can be derived for this purpose.

However, the general problem remains an open one. For a continuous attractor of dimension $n$, the bifurcation is potentially of codimension $n$.
The characterization of bifurcations with codimension $n > 2$ is an open problem.
Our approach aims to balance the specificity of a detailed bifurcation analysis with generality.

We have however characterized all possible perturbations for a ring attractor.
Any perturbations that are at most an epsilon distance from the original ring attractor (in ( C^1 ) topology) will either result in no bifurcation, a limit cycle, or a slow manifold with fixed points.

Instead of specifying the topology of the dynamics within the slow manifold, we provide a geometric statement. We demonstrate that the persistence of an invariant manifold can lead to fruitful analyses, even without explicit knowledge of the bifurcation. This approach allows us to derive generalization bounds and other insights.





>Likewise, the functions h and g are also not clear enough. It is unclear how one can obtain/choose the functions h and g such that the two systems defined by Eq. (2) and Eqs. (3) & (4) are equivalent.

Obtaining $h,\,g$ is problem specific, and cannot be given in closed for general invariant manifolds.
Let us state the essence of Theorem 1 differently: **if** $f$ has a normally hyperbolic invariant manifold, then there exist vector fields $h,\,g$.





> 1. What does **sufficiently smooth** mean in Theorem 1? As mentioned by the authors after this theorem, it applies to continuous piecewise linear systems. However, it cannot be applied to all piecewise smooth (PWS) systems , such as Filippov systems.
>In particular, for these systems, bifurcations involving non-hyperbolic fixed points can be analyzed using similar slow (center) manifold approaches, but only for part of the phase space.

Please note, that a center manifold is not necessarily unique, and generally is local in **both** space and time ($J \subset \mathcal{X} \times T$).
Stability, or invariance under the flow therefore generally cannot be analyzed using these methods.

>However, discontinuity-induced bifurcations cannot be examined in the same way, as there is no slow manifold in these cases.

We will make more clear in the text that discontinuous systems are not being considered.
Because all both theoretical models and activation functions for RNNs are at least continuous and piecewise smooth, we believe that this limitation still includes an essential part of relevant dynamical systems and that therefore our theory is very general.

Furthermore, the smoothness of the system determines how close and how smoothly the invariant manifold will change w.r.t. perturbation parameter.
If there is no assumption about even continuity, then the system will not

Finally, we rely on the minimal requirement of continuity of the vector field to be able to


>1. It is unclear under what conditions RNN dynamics can be decomposed into slow-fast form to which we can apply Theorem 1.

Theorem 1 holds for all RNNs that have a normally hyperbolic continuous attractor.
All continuous attractors are normally hyperbolic, the zero flow leaves an infinite gap.
So they can all be decomposed
We do not have a general expression for the decomposition, but show that it necessarily needs to exist.

> 1. In Sect. 4.1, line 213, it is vague how assuming an Euler integration with unit time step, the discrete-time RNN of (6) transforms to eq. (7). Is this transformation independent of the function f and matrix W in eq. (6)?

> 1. In S4, the sentence "All such perturbations leave at least a part of the continuous attractor intact and preserve the invariant manifold, i.e. the parts where the fixed points disappear a slow flow appears." needs more clarification. Could you explain the mathematical reasoning behind this assertion?

This just follows from Theorem 1……………………

**Questions:**

>How does the selection of a specific threshold value influence the identification and characterization of slow manifolds in neural networks with continuous attractors as discussed in the first lines of section 4.2?

> Could you elaborate on how different threshold settings impact the dynamics of network states and the emergence of persistent manifolds?

[We can include a plot that shows Hausdorff distance between identifies invariant manifolds as a function of the perturbation parameter?]
It’s not about emergence by identification.

**Limitations:**

> The authors have discussed most limitations of the analysis in the discussion section, but I suggest making them more explicit. This could be done by either incorporating a dedicated (sub)section on limitations or adding a bold title "**Limitations**" at the beginning of the relevant paragraph within the discussion section.
> As mentioned above, another important limitation is that it is difficult to determine how general their theoretical discussion is and whether it can be applied to investigate and obtain results for more general and high-dimensional cases.

We included a **Limitations** subsection.
@Abel, do you mean the part of the checklist?
If so I'll just tell all of the reviewers we'll add a sub-section like that.