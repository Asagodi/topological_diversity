### Weaknesses:
1. They discussed some interesting theoretical techniques (e.g., Theorem 1, Proposition 1) in their study. However, their theoretical investigation and results are limited to a few very simple systems, low-dimensional systems either in Section S4 or low-dimensional RNNs with specific activation functions and restrictive settings, i.e., specific parameter values (e.g., equations (1) and (10)). The bifurcation analysis of the line attractors and fast-slow decomposition in Section S2 are also studied for very simple systems. Therefore, it is difficult to determine how general their theoretical discussion is and whether it can be applied to investigate and obtain results for more general and high-dimensional cases.

The theory holds for any dimension.

The analytical results …

The numerical experiments …

1. In Sect. 3.1, the perturbation p(x) is not clear enough. Specifically, it is unclear:1) Under what conditions the perturbation function p(x) induces a bifurcation? 2) What types of (generic) bifurcations can arise from the perturbation p(x)? Likewise, the functions h and g are also not clear enough. It is unclear how one can obtain/choose the functions h and g such that the two systems defined by Eq. (2) and Eqs. (3) & (4) are equivalent.

1) Under what conditions the perturbation function p(x) induces a bifurcation?

2) What types of (generic) bifurcations can arise from the perturbation p(x)?

3) the functions h and g (how can one obtain/choose the functions h and g such that the two systems defined by Eq. (2) and Eqs. (3) & (4) are equivalent?)

1) For most CANs p(x) almost always induces a bifurcation. (Is it true that the only way to modify the dynamics such that there is not a bifurcation is via changing the level of attractiveness of the the continuous manifold?. In this sense “almost always” = measure zero of parameter space (?) For exact determination maybe Piotr’s ideas)

2) Any that are at most at epsilon distance from original system (C^1 topology) (Could you say / is it true that it will always be either no bifurcation, or a limit cycle or a slow manifold with fixed points?  I have that intuition but do not know if it is fair)

3)  h(x,y,\epsilon)

1. What does **sufficiently smooth** mean in Theorem 1? As mentioned by the authors after this theorem, it applies to continuous piecewise linear systems. However, it cannot be applied to all piecewise smooth (PWS) systems , such as Filippov systems. In particular, for these systems, bifurcations involving non-hyperbolic fixed points can be analyzed using similar slow (center) manifold approaches, but only for part of the phase space. However, discontinuity-induced bifurcations cannot be examined in the same way, as there is no slow manifold in these cases.

We will make more clear in the text that discontinuous systems are not being considered.

Because all typical RNNs are at least piecewise smooth, we believe that this limitation still includes a really big part of relevant dynamical systems and is therefore a very general theoretical result.

Further, the smoothness of the system determines how close and how smoothly the invariant manifold will change w.r.t. perturbation parameter.

1. It is unclear under what conditions RNN dynamics can be decomposed into slow-fast form to which we can apply Theorem 1.

Theorem 1 holds for all RNNs that have a normally hyperbolic continuous attractor. (Can you say something on whether most CAs are normally hyperbolic? I have that intuition but do not know if it is fair)

So they can all be decomposed

We do not have a general expression for the decomposition though.

1. In Sect. 4.1, line 213, it is vague how assuming an Euler integration with unit time step, the discrete-time RNN of (6) transforms to eq. (7). Is this transformation independent of the function f and matrix W in eq. (6)?

1. In S4, the sentence "All such perturbations leave at least a part of the continuous attractor intact and preserve the invariant manifold, i.e. the parts where the fixed points disappear a slow flow appears." needs more clarification. Could you explain the mathematical reasoning behind this assertion?

This just follows from Theorem 1……………………

**Questions:**

How does the selection of a specific threshold value influence the identification and characterization of slow manifolds in neural networks with continuous attractors as discussed in the first lines of section 4.2?

Could you elaborate on how different threshold settings impact the dynamics of network states and the emergence of persistent manifolds?

[We can include a plot that shows Hausdorff distance between identifies invariant manifolds as a function of the perturbation parameter?]
It’s not about emergence by identification.

**Limitations:**

- The authors have discussed most limitations of the analysis in the discussion section, but I suggest making them more explicit. This could be done by either incorporating a dedicated (sub)section on limitations or adding a bold title "**Limitations**" at the beginning of the relevant paragraph within the discussion section.
- As mentioned above, another important limitation is that it is difficult to determine how general their theoretical discussion is and whether it can be applied to investigate and obtain results for more general and high-dimensional cases.

We included a **Limitations** subsection