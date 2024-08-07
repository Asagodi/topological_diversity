We are grateful that the reviewers found our work novel and interesting. Reviewers remarked that it is "a novel contribution and an important result to bolster the continuous attractor hypothesis", "fresh look, novel, original, and interesting [and] superb theoretical motivation", that "the main thrust of the paper was very interesting and very novel" and that "it should be applauded."

To respond to the reviewer's comments, we have performed the following analysis:
 - quantify the fast-slow time scale separation on the manifold found in task-trained RNNs [Fig R1]
 - trained LSTM and GRU networks [Fig R2]
 - trained RNNs on a 2D task where the continuous attractor manifold is a torus [Fig R3]
# Generality of the theory
While most bifurcation analyses in theoretical neuroscience and machine learning are based on a particular parameterization (e.g., pairwise weight matrix), our theory applies to any differentiable dynamical system and to continuous piecewise smooth systems (with a global continuous attractor). Hence, the robustness of most continuous attractors are covered. The only necessary condition is the normal hyperbolicity.
## Architecture
We tested our theory with LSTMs and GRUs to support our claim about the universality of the dynamics of trained RNNs.
These networks have the same normally hyperbolic invariant slow ring manifold just like Vanilla RNNs (FigR.2C,D) and on this manifold we find fixed points (FigR.2A,B). This consistency of structure across different RNN architectures provides further validation of our theoretical framework.
## Simple systems
The analysis of theoretical models and numerical experiments is intended to illustrate the theory's practical applicability rather than to prove its generality.
We focused on low-dimensional systems because they are easier to visualize and are a better guide to developing intuition.
We include results on RNNs trained on a 2D task (a double angular velocity integration task) to further demonstrate our theory's relevance. In the trained RNNs (FigR.3A,B), we find a slow, attractive, invariant manifold in the shape of a torus with a point topology. Additionally, we find evidence supporting the relevance of the error bound in these trained RNNs (FigR.3C,D).
## Broader impact
Our theory is relevant to all (theoretical) neuroscientists trying to understand analog working memory and its robustness.
We demonstrate the robustness of the implemented analog memory in recurrent systems to perturbations of their connection matrix.
The theory has implications for theoretical models more generally: for example, we can be wrong about the activation function of the neurons; the neural dynamics is functionally robust to changes to it.

# Clarity
We acknowledge the reviewers' concerns regarding clarity and add details on the main topics highlighted in the reviews. 
## Section 5.2
Section 5.2 (now in a separate section) outlines the conditions under which approximations to an analog working memory problem are near a continuous attractor. This section is crucial for clarifying when a situation like Proposition 1 would occur. These conditions are met for RNNs:
C1: This translates to the existence of a manifold in the neural activity space with the same topology as the memory content. We formalize the dependence as the output mapping being a locally trivial fibration over the output manifold.
C2: Persistence, as per the reverse of the Persistent Manifold Theorem, requires the flow on the manifold to be slow and bounded.
C3+C4: Non-positive Lyapunov exponents correspond to negative eigenvalues of $\nabla_zh$. Along with dynamics robustness (corresponding to the persistence of the manifold), this implies normal hyperbolicity. We have expanded on this correspondence by building on [1].
## Parameter dependence for the analysis
The threshold parameter for identifying invariant slow manifolds was chosen to reflect the bimodal distributions of speeds along the integrated trajectories.
The supplementary document (FigR.1) shows that the identified invariant manifold accurately reflects the fast-slow separation expected for a normally hyperbolic system, thereby validating our method's legitimacy.
The number of fixed points (NFPS) identified depends on the number of points sampled for the angular flow on the invariant ring, but converges to the true NFPS as the grid of initial points is refined.
# Limitations
We have added a separate **Limitations** subsection:
While we explicitly describe the topology and dimensionality of the identified invariant manifolds for a representative set, our results indicate that most solutions exhibit a ring invariant manifold with a slow flow. This separation of timescales necessarily exists for well-trained networks; however, the analysis is not guaranteed to work for systems without a fast-slow decomposition.

To identify solutions with a fast-slow decomposition, we rely solely on the generalization property of the network, measured in terms of the normalized mean square error over ten times longer trials. The possible solutions the networks can find are restricted by having a linear output mapping. For a nonlinear output mapping, a possible solution for analog memory is the quasi-periodic toroidal attractor, but this is not a possible solution with a linear output mapping. While our analysis methods can identify these limit sets, we do not have a straightforward way to parametrize the invariant manifold.

Our analysis relies on identifying a time scale separation from simulated trajectories. If the separation of time scales is too small, our method may inadvertently identify parts of the state space that are only forward invariant (i.e., transient). However, this did not pose a problem in our analysis of the trained RNNs, which is unsurprising, as the separation is guaranteed by state noise robustness.

[1] Mané, R. (1978). Persistent manifolds are normally hyperbolic. Transactions of the American Mathematical Society, 246, 261-283.