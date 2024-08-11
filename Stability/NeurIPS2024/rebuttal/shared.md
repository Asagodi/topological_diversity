We are grateful and encouraged that the reviewers found our work novel and interesting. Reviewers remarked that it is "a novel contribution and an important result to bolster the continuous attractor hypothesis", "fresh look, novel, original, and interesting [and] superb theoretical motivation", that "the main thrust of the paper was very interesting and very novel" and "it should be applauded."
We feel many of your suggestions have led us to changes and additions that better position the paper.

To respond to the reviewer's comments, we have performed the following analysis:

 * quantify the fast-slow time scale separation on the manifold found in task-trained RNNs (Fig R1)

 * trained LSTM and GRU networks (Fig R2)

 * trained RNNs on a 2D task where the continuous attractor manifold is a torus (Fig R3)

# Generality of the theory
While most bifurcation analyses in theoretical neuroscience and machine learning are based on a particular parameterization (e.g., pairwise weight matrix), our theory applies to any differentiable dynamical system and to continuous piecewise smooth systems (with a global continuous attractor). Hence, the robustness of most continuous attractors are covered. The only necessary condition is the *normal hyperbolicity* as demonstrated via separation of eigenvalues (Fig R1).
## Architecture
We tested our theory with LSTMs and GRUs to support our claim about the universality on trained RNNs.
These networks forms the same normally hyperbolic invariant slow ring manifold just like Vanilla RNNs (Fig R2C,D) and on this manifold we find fixed points (Fig R2A,B). This consistency of structure across different RNN architectures provides further validation of our theoretical framework.
## Simple systems
The analysis of theoretical models and numerical experiments is intended to illustrate the theory's practical applicability rather than to prove its generality.
We focused on low-dimensional systems because they are easier to visualize and are a better guide to developing intuition.
We include results on RNNs trained on a 2D task (a double angular velocity integration task) to further demonstrate our theory's relevance. In the trained RNNs (Fig R3A,B), we find a slow, attractive, invariant manifold in the shape of a torus with a point topology. Additionally, we find evidence supporting the relevance of the error bound in these trained RNNs (Fig R3C,D).
## Broader impact
Approximate continuous attractor theory applies to dynamical systems inferred from data, from task-trained neuralODE and RNNs, finite size effects on theoretical models, and due to lack of expressive power parametrized dynamics.
We believe our theory opens up a new way of grouping similar dynamical systems for understanding the essential computation.

# Clarity
We acknowledge the reviewers' concerns regarding clarity and add details on the main topics highlighted in the reviews. If accepted, our final manuscript will be updated.
## Section 5.2
Section 5.2 outlines the conditions under which approximations to an analog working memory problem are near a continuous attractor. This section is crucial for clarifying when a situation like Proposition 1 would occur. These conditions are met for RNNs:

 * C1: This translates to the existence of a manifold in the neural activity space with the same topology as the memory content. We formalize the dependence as the output mapping being a locally trivial fibration over the output manifold.
 * C2: Persistence, as per the reverse of the Persistent Manifold Theorem, requires the flow on the manifold to be slow and bounded.
 * C3+C4: Non-positive Lyapunov exponents correspond to negative eigenvalues of $\nabla_zh$. Along with dynamics robustness (corresponding to the persistence of the manifold), this implies normal hyperbolicity. We have expanded on this correspondence by building on [1].

## Parameter dependence for the analysis
The threshold parameter for identifying invariant slow manifolds was chosen to reflect the bimodal distributions of speeds along the integrated trajectories.
The supplementary document (Fig R1) shows that the identified invariant manifold accurately reflects the fast-slow separation expected for a normally hyperbolic system, thereby validating our method's legitimacy.
The number of fixed points (NFPS) identified depends on the number of points sampled for the angular flow on the invariant ring, but converges to the true NFPS as the grid of initial points is refined.
# Limitations
We will add a separate **Limitations** subsection:

Although our theory is general, since the magnitude of perturbation is measured in uniform norm, for specific parameterizations, further analysis is needed. If the parameters are not flexible enough, the theory may not apply, for example, RNNs with "built-in" continuous attractors such as LSTM without a forget gate cannot be destroyed. However, in biological systems, this is highly unlikely at least at the network level.

Our empirical analysis requires the fast-slow decomposition around the manifold. Not all dynamical system solutions to the tasks that require analog memory have this property (hence sec 5.2). Solutions such as the quasi-periodic toroidal attractors or constant spirals represent challenges to the current framework of analysis in understanding the general phenomena of analog memory without continuous attractors.

Our numerical analysis relies on identifying a time scale separation from simulated trajectories. If the separation of time scales is too small, it may inadvertently identify parts of the state space that are only forward invariant (i.e., transient). However, this did not pose a problem in our analysis of the trained RNNs, which is unsurprising, as the separation is guaranteed by state noise robustness (due to injected state noise during traning).

[1] Mané, R. (1978). Persistent manifolds are normally hyperbolic. Transactions of the American Mathematical Society, 246, 261-283.
