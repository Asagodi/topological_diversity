We would like to thank the reviewer for their valuable comments and suggestions. They have significantly enhanced the quality of our manuscript.


> signposting

We thank the reviewer for and agree with this feedback. We tried to make the manuscript clearer by:
- Changing the last paragraph of the intruduction, making a reference to which points are made in which section.
- Adding which are the main questions that arise and in which section they are answered after the exploration in the end of Section 2.
- Adding extra senteces at the start of sections and subsections to better introduce the main messages of the section.


> Fig 1) A was a great figure, B was never mentioned in the text, despite being very pretty.

We now added an essential reference to Fig.1B.

> Fig 2) Why were there hexagons everywhere? I never could find any
reason for there to be hexagonal figures, did you just make the ring a hexagon for fun? If so, tell the reader!

This is simply a quality of the ring attractor as proposed in [1]. The ring attractor we discuss in the text is implemented by 6 neurons. This results in 6 line attractors, fused at their ends. 
At these fusion points there is a lack of smoothness (similar to the function abs(x)).
The ring is embedded in 6 dimensional space and we visualize it by projecting it to two dimensions.
So this piecewise straigh "ring" attractor, when projected onto two dimensions looks like a hexagon.
We added this clarification to the text.


> Further, in B and C, are the blue lines numerical? How did you choose what to colour blue?

We find the connecting orbits between the fixed points by identifying the slow part of simulated trajectories.
From these slow trajectories, we choose the trajectory that is closest to two fixed points.
These trajectories go from a saddle node to a stable fixed point.
We do this for every pair of neighbouring fixed points.



> [...] Should I be using it as evidence for your big theoretical claim? Or is it just illustrative?

In this section, we aim to motivate the relevance of our theory by demonstrating two key points: first, that continuous attractors are inherently fragile to noise in the parameters (a well-known fact), and second, that all bifurcations from and approximations of continuous attractors share a common feature—a slow invariant attractive manifold. In our examples, we focus on ring attractors to set the stage for our theoretical work, where the attractive manifold forms a ring.

In the subsequent sections, we provide a detailed explanation of this seemingly universal phenomenon using our theory. With the revised writing and improved signposting, we hope this purpose is now clearer.


> What am I looking at in figure 4A2...

We did not include the slow repulsive points that we found as they do not seem to be relevant for how the network solves the task.
We found this structure both through the Fixed Point Finder and through the Newton-Raphson method that we describe in the paper.
The grey lines on the torus show simulated trajectories. 
Fig. 4A1. is indeed showing the output of an integrated angular velocity.
This figure is just to illustrate the task. This is in addition to Fig. 4C. which shows the stability structures of the network for which the trajectories are shown as an example.

We appreciate your remarks and have revised the text accordingly.
1. Fig. 4A2: You are correct; this figure shows multiple example trajectories. We have corrected this in the text and caption.

1. Limit Cycle Mislabeling: We have corrected the mistake of referring to Fig. 4B and C as a limit cycle. 

1. Repulsive Ring Invariant Manifold: We have clarified that Fig. 4D describes the ring inside the torus. The torus contains repulsive ring invariant manifold, not just the attractive limit cycles marked on the figure.
We did not include the slow repulsive points that we found as they do not seem to be relevant for how the network solves the task and was decreasing the interpretability of the figure.

1. Coloring of the Torus: The grey lines on the torus represent simulated trajectories, indicating the paths that the network makes after making the saccade.

1. Fig. 4A1 Task Description: We acknowledge the confusion. Fig. 4A1 shows the output of an integrated angular velocity, illustrating the task. This figure is intended to provide context for the task in addition to Fig. 4C, which shows the stability structures of the network and the corresponding example trajectories.
Figure 4A1 and 4A2 are different in this sense, they show how the networks behave (in output space).
The other subfigures in Fig.4 illustrate the stability structures of the networks, i.e., to which part of state space they tend over time in the absence of input.



> On that last point [...]

We included these details to support reproducibility. We have now added the extra details to explain what is shown in Fig. 4.



> Figure 5A) Why did one of the finite time results go above the line?
Shouldn't this be an exact theoretical result, yet it appears not to be true?

This is indeed an exact theoretical result; however, our numerical methods are not exact.
Because we only numerically approximate the invariant manifold of each trained RNN, on which we calculate the uniform norm of the vector field, we cannot guarantee that this is exactly the vector field that the simulated trajectories follow.
Additionally, the network initialization along the invariant manifold is only approximate due to the limitations of our parametrization. As a result, there may be discrepancies between the theoretical expectations and the simulated trajectories.
Nevertheless, it is important to note that this method has only a single violation of our theory among approximately 250 networks tested.



> Seemed obtuse to claim a linear relationship then show a log-linear plot, fig 5C? How should I see this?

We appreciate the reviewer pointing this out. We have corrected the mistake and now observe the log-linear relationship, which aligns with the theoretical expectation. The angular error should asymptotically approach zero as the number of fixed points increases, making the log-linear plot the appropriate representation for this relationship.



> Did you define the memory capacity?

We characterize the memory capacity of the network by calculating the entropy of this probability distribution (described in Supp.Sec. 3.2, we have now moved the description to the general analysis methods description in Supp.Sec.6).
We determine the location of the fixed points through the local flow direction criterion.
Because the system evolves along a 1D subspace, we can simply look at the local flow direction to assess to which stable fixed point each part of the invariant ring converges to. We do this for 1024 sample points in the found invariant manifold.




> Did you need to introduce omega-limit set, especially given the likely audience for this paper?

We believe that this definition is supporting the definition of memory capacity.
As we explained above, the memory capacity is calculated from the omega-limit set of each of the points on the invariant ring.
This idea can be more generally applied to systems with other omega-limit sets, like limit cycles or chaotic orbits and therefore included this definition.

> You should definitely cite and discuss the relationship to this
paper: Information content in continuous attractor neural networks is
preserved in the presence of moderate disordered background
connectivity, Kuhn & Monasson, 2023.
>

[Kuhn and Monasson 2023][2] analyzes an Ising network perturbed with a specially structured noise at the thermodynamic limit.
Although their analysis elegantly shows that the population activity of the perturbed system does not destroy the Fisher information about the input, they do not consider a scenario where the ring attractor is used as a working memory mechanism, it is rather used to encode instantaneous representation. In contrast, our analysis involves understanding how the working memory content degrades over time due to the dynamics. We are not aware of any mean field analysis that covers this aspect.

We have now included a discussion of this work to a section on continuous attractors in mean field approaches (as an extension of the low-rank approximations).



> What was section 5.2 trying to show? 


See the shared rebuttal for our clarification of Section 5.2.



> Smaller things:

We thank the reviewer for pointing out these mistakes which we have now corrected.

### References

[1] Noorman, M., Hulse, B. K., Jayaraman, V., Romani, S., & Hermundstad, A. M. (2022). Accurate angular integration with only a handful of neurons. bioRxiv, 2022-05.

[2] Kühn, T., & Monasson, R. (2023). Information content in continuous attractor neural networks is preserved in the presence of moderate disordered background connectivity. Physical Review E, 108(6), 064301.
