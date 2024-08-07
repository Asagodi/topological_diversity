We would like to thank the reviewer for their valuable comments and suggestions. They have significantly enhanced the quality of our manuscript.

> signposting
We thank the reviewer for and agree with this feedback. We improved the text by:
- Changing the last paragraph of the introduction, summerizing each section
- Highlighting the main question arising from the discussed examples
- Adding introductory senteces at the start of sections and subsections to describe the main message of the section

> Fig 1) A was a great figure...
We now added the needed reference to Fig.1B.

> Fig 2) Why were there hexagons everywhere?...
This is a quality of the ring attractor in [1]. The attractor implemented by 6 neurons is made up of 6 line attractors, fused at their ends. At these fusion points there is a lack of smoothness (similar to the function abs(x)). Therefore, this piecewise straight "ring" attractor, when projected onto two dimensions looks like a hexagon. 

> Further, in B and C[...]
We find the connecting orbits between the fixed points by identifying the slow part of simulated trajectories. From these slow trajectories, we choose the trajectory that is closest to two fixed points. These trajectories go from a saddle node to a stable fixed point. We do this for every pair of neighbouring fixed points.

> [...] Should I be using it as evidence for your big theoretical claim? 
In this section, we aim to motivate the relevance of our theory by demonstrating two key points: 1) that continuous attractors are inherently fragile to noise in the parameters (a well-known fact), and 2) that all bifurcations from and approximations of continuous attractors share a common feature—a slow invariant attractive manifold. 
In the subsequent sections, we provide an explanation of the universal features that all discussed models share. 

> What am I looking at in figure 4A2...
We appreciate the remarks and have revised the text accordingly.
1. ou are correct; this figure shows multiple example trajectories. We have corrected this in the text and caption.
1. We have corrected the mistake of referring to Fig. 4B and C as a limit cycle. 
1. We did not include the slow repulsive points that we found as it decreased the interpretability of the figure and this structure is not relevant for how the network solves the task.
1. The grey lines on the torus represent simulated trajectories, indicating the network dynamics after a saccade.
1. Fig. 4A1 shows the output of an integrated angular velocity, illustrating the task in addition to the solution type in Fig. 4C.
Figure 4A1 and 4A2 are different in that they show how the networks behave (in output space).
The other subfigures in Fig.4 illustrate the stability structures of the networks, i.e., to which part of state space they tend over time in the absence of input.

> On that last point...
We included these details to support reproducibility.

> Figure 5A) Why did one of the finite time results go above the line?
This is indeed an exact theoretical result; however, our numerical methods are not exact.
Because we numerically approximate the invariant manifold of each trained RNN, on which we calculate the uniform norm of the vector field, we cannot guarantee the vector field simulated trajectories follow to be exact.
Additionally, the network initialization along the invariant manifold is approximate due to our parametrization (using a cubic spline).
Nevertheless, it is important to note that this method has only a single violation of our theory among approximately 250 networks tested.

> Seemed obtuse...
We appreciate the reviewer pointing this out. We have corrected the mistake and now observe the log-linear relationship, which aligns with the theoretical expectation. The angular error should asymptotically approach zero as the number of fixed points increases, making the log-linear plot the appropriate representation for this relationship.

> Did you define the memory capacity?
We determine the location of the fixed points through points of reversal of the flow direction (the system evolves along a 1D subspace).
We calculate the probability of a point converging to a stable fixed point by assessing the local flow direction (which allows us to characterize the basin of attraction).
The memory capacity is the entropy of this probability distribution.


> Did you need to introduce omega-limit set...
We believe that this definition is supporting the definition of memory capacity.
As we explained above, the memory capacity is calculated from the omega-limit set of each of the points on the invariant ring.
This idea can be more generally applied to systems with other omega-limit sets, like limit cycles or chaotic orbits and therefore included this definition.

> You should definitely cite...
[2] analyzes an Ising network perturbed with a specially structured noise at the thermodynamic limit.
Although their analysis elegantly shows that the population activity of the perturbed system does not destroy the Fisher information about the input to study instantaneous encoding, they do not consider a scenario where the ring attractor is used as a working memory mechanism. In contrast, our analysis involves understanding how the working memory content degrades over time due to the dynamics. We are not aware of any mean field analysis that covers this aspect.
We include this work to discuss continuous attractors in mean field approaches.

> What was section 5.2 trying to show? 
See the shared rebuttal for our clarification of Section 5.2.

### References
[1] Noorman, M., Hulse, B. K., Jayaraman, V., Romani, S., & Hermundstad, A. M. (2022). Accurate angular integration with only a handful of neurons. bioRxiv, 2022-05.
[2] Kühn, T., & Monasson, R. (2023). Information content in continuous attractor neural networks is preserved in the presence of moderate disordered background connectivity. Physical Review E, 108(6), 064301.
