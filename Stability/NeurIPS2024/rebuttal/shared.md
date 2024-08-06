We would like to thank the reviewers 

# List of changes
1.  
1. 
1. 


# 
We are very happy that the reviewers thought that
> the main idea of the paper is interesting
> this is a novel contribution and an important result to bolster the continuous attractor hypothesis
> study of continuous attractors in the finite time limit [...] is a fresh look, novel, original, and interesting [and] superb theoretical motivation
> the main thrust of the paper was very interesting and very novel
and that one of the reviewers even thinks 
> it should be applauded.


However, we are disappointed by the corresponding scores.
Perhaps these scores can be explained by the lack of clarity in the previous version of the paper.
We have now added clarifications about topics, experiments, figures and concepts that were pointed out in the reviews and improved the overall flow of the text and the main message of the different subsections.





We would like to summarize the important big picture clarifications to our paper.
1. We would like to emphasize that the theory applies very generally to any continuous attractor that is normally hyperbolic.
In fact, it applies to any differentiable dynamical system
and to continuous piecewise smooth systems (for which the continuous attractor is a global attractor).
This covers most of the theoretical models involving continuous attractors as we tried to point out by discussing the main classes of implementations of a ring attractor.

The role of the analysis of theoretical models, and that of the numerical experiments involving trained RNNs is not to prove the generality of the theory, but to illustrate its practical applicability.
We focused on low-dimensional systems because they easier to visualize and are a better guide to develop intuition.
We agree however that testing the theory in higher dimensional settings provides proof of the relevance of the theory in practice.
Therefore, we include results on RNNs trained on 2D tasks. <!-- Double angular velocity integrations and  -->
We find a slow attractive invariant manifold with a point topology in the trained RNNs.
Furthermore, we find evidence of the relevance of the bound on the error in these trained RNNs as well.

Another point about the generality of the theory involves our claim of the universality of the found RNNs through training.
We tested our theory in other architectures as well, namely LSTMs and GRUs.
We found the same normally hyperbolic attractive invariant manifolds of fixed point type.


1. For whom is this relevant?
We believe that our theory is relevant to all (theoretical) neuroscientists who are trying to understand analog working memory, the ways how it might be implemented in the brain and its robustness 
We demonstrate robustness of the implemented analog memory in recurrent systems to perturbations of their connection matrix. 
However, the theory also has implications robustness of theoretical models more generally than just : for small changes to the functional form of the activation function of the neurons the network will behave functionally similar to the original model.




1. Hyperparameters / Parameter choices and parameter dependence for the analysis




# Discuss contributions and impacts





# Limitations

We have now extended a serparate **Limitations** subsection in the paper.


Although, we only explicitly describe the topology and dimensionality of the identified invariant manifolds for a representative set, the results indicate that most solutions have a ring invariant manifold with a slow flow on it.
This separation of timescale necessarily exists for well-trained networks, however, the analysis is not guaranteed to work for systems without a fast-slow decomposition.

To identify solutions with a fast-slow decomposition only rely on the generalization property of the network (in terms of the normalized mean square error for ten times longer trials).
The possible solutions that the networks can find are restricted by having a linear output mapping.
For a nonlinear output mapping, a possible solution for analog memory is the quasi-periodic toroidal attractor, but this is not possible for a linear output mapping.
Our analysis methods can identify these limit sets, but we do not have a simple way to parametrize the invariant manifold.

Our analysis is dependent on a time scale separation that we identify from simulated trajectories.
If the separation of time-scales is small, our method can 
This separation is reflected however in the distance of the approximation to the continous attractor; a system without a large separation will be either not robust to state (S-type) noise or will be performing poorly for longer trial times than it was trained on.



