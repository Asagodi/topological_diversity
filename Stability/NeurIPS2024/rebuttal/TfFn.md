### Weaknesses:

> I got very confused by what most of the paper was trying to show. The main result, section 3, is, almost draw-droppingly, nice. (Though I'm partly trusting that the authors are applying the results correctly; they seem to be, because the step from the stated theorem to their claims is not large) The rest is, ..., a bit more meh, especially in how it was presented.
>

> I would have liked a lot more signposting: why are these sections the right ones  to care about? What are you showing, and why?
>

We added extra senteces as the start of sections and subsections to better introduce the main message of the section.


> For example, section 2 shows that noise degrades CANs - a nice
literature review - and then shows a categorisation of perturbed fixed
point dynamics. I guess the idea was to show that noise-instability was a
 problem for all CAN types in the literature? If so you could do with
signposting that. If not, why did you include all of that? Why was it
important that some patterns appear and not others in the perturbed
dynamics?
>

> Then there were a lot of quite confusing things, especially in the figures:
>

> Fig 1) A was a great figure, B was never mentioned in the text, despite being very pretty.

We now added a reference to Fig.1B.

> Fig 2) Why were there hexagons everywhere? I never could find any
reason for there to be hexagonal figures, did you just make the ring a hexagon for fun? If so, tell the reader!

This is simply a quality of the ring attractor as proposed in [1]. The ring attractor we discuss in the text is implemented by 6 neurons. This results in 6 line attractors, fused at their ends. 
At these fusion points there is a lack of smoothness (similar to the function abs(x)).
The ring is embedded in 6 dimensional space and we visualize it by projecting it to two dimensions.
So this piecewise straigh "ring" attractor, when projected onto two dimensions looks like a hexagon.


> Further, in B and C, are the blue lines numerical? How did you choose what to colour blue?
We find the connecting orbits between the fixed points by identifying slow part of simulated trajectories and choosing the trajectory that is closest to two fixed points (which goes from a saddle node to a stable fixed point).

> Should I be using it as evidence for your big theoretical claim? Or is it just illustrative?
We believe that all these approximations share a common feature: a slow invariant attractive manifold.
In our examples, because we focus on ring attractors to motivate our theoretical work, the attractive manifold is a ring.
We set out to explain this seemingly universal phenomenon with our theory.


> Fig 3) What am I looking at in figure 4A2: it says example
trajectory? But there are many trajectories, no? The dots in B and C are
 presumably fixed points, why then is it called a limit cycle (line
246)? It doesn't look like that? Why is 4D described as around a
repulsive ring invariant manifold? Are you describing the ring inside
the torus? (rather than the presumably attractive limit cycles that are
marked on the figure) What does the colouring of the torus (the greys)
denote? I didn't get told what the task was for figure A1 so had to go
to the appendix to see that this is apparently the output of the
integration? Why are you plotting the output space, and not the
recurrent space as in all the other figures?

We appreciate the remarks, and will revise the text.
Fig.4A2 is indeed showing example trajectories, we have corrected this in the text and caption.
We have further corrected the mistake of referring to Fig.4B and C as a limit cycle.
We did not include the slow repulsive points that we found as they do not seem to be relevant for how the network solves the task.
We found this structure both through the Fixed Point Finder and through the Newton-Raphson method that we describe in the paper.
The grey lines on the torus show simulated trajectories. 
Fig.4A1 is indeed showing the output of an integrated angular velocity.
This figure is just to illustrate the task. This is in addition to Fig.4C which shows the stability structures of the network for which the trajectories are shown as an example.


> On that last point, why include details of the (standard)
discretisation scheme, MSE loss, and network details, when key steps to
understand what I am looking at (e.g. figure A1 = output space) are
missing?
We included these details to support reproducability.
We have now added the extra details to explain what is shown in Fig.4.

> Figure 5A) Why did one of the finite time results go above the line?
Shouldn't this be an exact theoretical result, yet it appears not to be true?

This is indeed an exact theoretical result, however, our numerical methods are not exact.
Becuase we only approximate the invariant manifold, on which we calculate the uniform norm of the vector field, we cannot guarantee that this is exactly the vector field that the simulated trajectories follow.


> Seemed obtuse to claim a linear relationship then show a log-linear plot, fig 5C? How should I see this?

We have corrected this mistake to indeed observe the log-linear relationship.


> Did you define the memory capacity?

We characterize the memory capacity of the network by calculating the entropy of this probability distribution.
We determine the location of the fixed points through the local flow direction criterion as described in Sec.~\ref{sec:fastslowmethod}
and determine the basin of attraction
\begin{equation}
\basin(x^*) \coloneqq \{x\in \manifold \ | \lim_{t\rightarrow\infty}\varphi(t,x)=\{x^*\}\}.
\end{equation}
through assesing the local flow direction for 1024 sample points in the found invariant manifold.
This invariant manifold was found to be consistently close the the original invariant ring attractor.
We have updated the manuscript to better explain this term.

> Did you need to introduce omega-limit set, especially given the likely audience for this paper?

We believe that this definition is supporting the definition of memory capacity.
As we explained above, the memory capacity is calculated from the omega-limit set of each of the points on the invariant ring.

> Finall, some other points:
>

> You should definitely cite and discuss the relationship to this
paper: Information content in continuous attractor neural networks is
preserved in the presence of moderate disordered background
connectivity, Kuhn & Monasson, 2023.
>

presence of disorder in the interactions could impact the information stored in attractor networks in the form of patterns. In the case of CANNs this translates into a breaking of the translational symmetry of bump-like solutions

disorder does not wipe out all information in the attractor network. In particular, the Fisher information is robust to
the introduction of disorder



> What was section 5.2 trying to show? First it claims that 2.2
presents a theory of approximate solutions in the neighbourhood of
continuous attractors (news to me, as far as I could tell, that section
showed me that all CAN models were unstable to noise and turn into a
variety of different fixed points under noise, that doesn't sound like a
 theory at all? Section 3 seems to be the theory?) Then you list four
conditions on what sounds like exactly the same problem? What is the
difference between dynamical systems having robust manifolds, and the
working memory task being solved, isn't the whole point of the model
that these two are the same? (i.e. you can solve a working memory task
with a CAN). Is this supposed to be a concluding section that says when
working memory can be solved? Then why have you suddenly defined state
and dynamical noise that haven't been used before, I thought we had a
perfectly nice definition of perturbations on the network (equation 2)?
This section seemed... strange, in my eyes the paper would be improved
by removing it
>

> Smaller things:
>

> line 107 - comment left in doc
>

> Figure 2 caption line 1, missing 'of' and pluralisation of 'implementation'
>

We thank the reviewer for pointing out these mistakes.


> So all in all, I think the exposition, despite, as I said, often
being very clear when describing a single idea, is, on a macro scale, a
mess. I had a very hard time following, and the figures were quite
tough-going. I think this paper should probably get in, but I think it
should be given a good clean up first, at least for a humble
neuro-theorist, rather than a dynamical systems guru, to understand.







[1] Noorman, M., Hulse, B. K., Jayaraman, V., Romani, S., & Hermundstad, A. M. (2022). Accurate angular integration with only a handful of neurons. bioRxiv, 2022-05.