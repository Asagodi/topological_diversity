We are grateful to hear that the reviewers found our work novel and interesting. Reviewers remarked that it is "a novel contribution and an important result to bolster the continuous attractor hypothesis", "fresh look, novel, original, and interesting [and] superb theoretical motivation", that "the main thrust of the paper was very interesting and very novel" and that "it should be applauded."

To respond to the reviewer's comments, we have performed the following analysis:
 - quantify the fast-slow time scale seperation on the manifold found in task-trained RNN [Fig R1]
 - trained LSTM and GRU networks [Fig R2]
 - trained RNNs on 2D task where the continuous attractor manifold is a torus [Fig R3]

## Generality of the Theory

While most bifurcation analyses in theoretical neuroscience and machine learning is based on a particular parameterization (e.g., pairwise weight matrix), our theory is applies to any differentiable dynamical system and to continuous piecewise smooth systems (for which the continuous attractor is a global attractor). Hence, the behavior of many different ring attractors and RNNs discussed can be explained in this framework. Note that the only important condition is the normal hyperbolicity.

Another point about the generality of the theory involves our claim of the universality of the found RNNs through training.
We tested our theory in other architectures as well, namely LSTMs and GRUs.
Our findings indicate that the solutions identified by the trained LSTM and GRU models exhibit similar patterns to those observed in Vanilla RNNs. Specifically, we observed the presence of a normally hyperbolic attractive invariant ring characterized by slow dynamics that eventually converge to fixed points. This consistency across different RNN architectures reinforces the robustness of our initial observations and provides further validation of our theoretical framework.

## Focus on the ring attractor implementations

MPComment: I'm not happy with the writing below. Maybe process through LLM if you are stuck.

The role of the analysis of theoretical models, and that of the numerical experiments involving trained RNNs is not to prove the generality of the theory, but to illustrate its practical applicability.
We focused on low-dimensional systems because they easier to visualize and are a better guide to develop intuition.
We agree however that applying our theory to higher dimensional problems would provide a convincing argument for its practical relevance.
Therefore, we include results on RNNs trained on a 2D task: a double angular velocity integration task.
We find a slow attractive invariant manifold with a point topology in the trained RNNs.
Furthermore, we find evidence of the relevance of the bound on the error in these trained RNNs as well.

## Broader impact within compuational neuroscience


## Clarity

We agree with the reviewers on the lack of clarity in places.

MPComment: I don't think we should promise what we will change...that doesn't work well

We have now added clarifications about topics, experiments, figures and concepts that were pointed out in the reviews and improved the overall flow of the text and the main message of the different subsections.

However, we are disappointed by the corresponding scores.
Perhaps these scores can be explained by the lack of clarity in the previous version of the paper.


We would like to summarize the important big picture clarifications to our paper.



2. For whom is this relevant? <!--# Discuss contributions and impacts-->
We believe that our theory is relevant to all (theoretical) neuroscientists who are trying to understand analog working memory, the ways how it might be implemented in the brain and its robustness
We demonstrate robustness of the implemented analog memory in recurrent systems to perturbations of their connection matrix.
However, the theory also has implications robustness of theoretical models more generally than just : for small changes to the functional form of the activation function of the neurons the network will behave functionally similar to the original model.


3. Section 5.2

Section 5.2 (which we moved to its own section now) is showing under what conditions approximations to a analog working memory problem are near a continuous attractor. 
This section is important to clarify under what conditions a situation such as in Proposition 1 would arise. 
In fact, these conditions are met for RNNs.
C1 translates to the existence of a manifold in the neural activity space with the same topology as the memory content (we have elaborated on this by formalizing this dependence as that the output mapping is a locally trivial fibration over the output manifold.)
(C2) The persistence (see the verserse of the Persistent Manifold Theorem) requires that the flow on the manifold is slow and bounded.
(C3+C4) The S-type robustness requires non-positive Lyapunov exponents (i.e. the negative eigenvalues of \(\nabla_\vz \vh\)).
Along with the D-type robustness (corresponding to persistence of the manifold) implies and normally hyperbolic (we have expanded on this correspondence through building on the work of Mane [1]).
The Persistent Manifold Theorem says that systems in the neighbourhood of a (normally hyperbolic) continous attractor will behave similarily to it.

As a side note, we would like to mention that we have added a more general theoretical result that proves that the invariant slow manifold might have some additional dynamics (which is not slow) as long as the flow of the system is in the kernel of the output mapping  (which covers the torus solution type approximation in Fig.4D).





4. Hyperparameters / Parameter choices and parameter dependence for the analysis
There are very few limitations we place on training the RNNs.


The threshold parameter for identifying invariant slow manifolds was chosen such that it reflects the two distributions of speeds along the integrated trajectories.
There is a bit of dependence of how many fixed points are identified on how many points on the invariant are sampled for the angular flow.
However, this will converge to a maximal number if the grid of initial points is increased.
Becuase all networks have less than 50 fixed points, we believe that 1024 initial points are sufficient.






# Limitations

We have now extended a serparate **Limitations** subsection in the paper.


Although, we only explicitly describe the topology and dimensionality of the identified invariant manifolds for a representative set, the results indicate that most solutions have a ring invariant manifold with a slow flow on it.
This separation of timescale necessarily exists for well-trained networks, however, the analysis is not guaranteed to work for systems without a fast-slow decomposition.

To identify solutions with a fast-slow decomposition only rely on the generalization property of the network (in terms of the normalized mean square error for ten times longer trials).
The possible solutions that the networks can find are restricted by having a linear output mapping.
For a nonlinear output mapping, a possible solution for analog memory is the quasi-periodic toroidal attractor, but this is not possible for a linear output mapping.
Our analysis methods can identify these limit sets, but we do not have a simple way to parametrize the invariant manifold.

Our analysis is dependent on a time scale separation that we identify from simulated trajectories.
If the separation of time-scales is small, our method can accidentally identify parts of state space that are only forward invariant (i.e. transient).
Nevertheless, this seems to have not caused a problem with our analysis of the trained RNNs.
This is not surprising, the separation is reflected in the distance of the approximation to the continous attractor; a system without a large separation will be either not robust to state (S-type) noise or will be performing poorly for longer trial times than it was trained on.



[1] Mané, R. (1978). Persistent manifolds are normally hyperbolic. Transactions of the American Mathematical Society, 246, 261-283.