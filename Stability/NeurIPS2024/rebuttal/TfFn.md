### Weaknesses:

> I got very confused by what most of the paper was trying to show. The main result, section 3, is, almost draw-droppingly, nice. (Though I'm partly trusting that the authors are applying the results correctly; they seem to be, because the step from the stated theorem to their claims is not large) The rest is, ..., a bit more meh, especially in how it was presented.
>

> I would have liked a lot more signposting: why are these sections the right ones  to care about? What are you showing, and why?
>

Fix this with better section titles?

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
>

> Fig 2) Why were there hexagons everywhere? I never could find any
reason for there to be hexagonal figures, did you just make the ring a hexagon for fun? If so, tell the reader!
>

> Further, in B and C, are the blue lines numerical? How did you choose what to colour blue?
>

> Should I be using it as evidence for your big theoretical claim? Or is it just illustrative?
>

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
>

> On that last point, why include details of the (standard)
discretisation scheme, MSE loss, and network details, when key steps to
understand what I am looking at (e.g. figure A1 = output space) are
missing?
>

> Figure 5A) Why did one of the finite time results go above the line?
Shouldn't this be an exact theoretical result, yet it appears not to be true?
>

> Seemed obtuse to claim a linear relationship then show a log-linear plot, fig 5C? How should I see this?
>

> Did you define the memory capacity?
>

> Did you need to introduce omega-limit set, especially given the likely audience for this paper?
>

> Finall, some other points:
>

> You should definitely cite and discuss the relationship to this
paper: Information content in continuous attractor neural networks is
preserved in the presence of moderate disordered background
connectivity, Kuhn & Monasson, 2023.
>

Cited a whole 1 time. Guess that’s the author 🙃

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

> So all in all, I think the exposition, despite, as I said, often
being very clear when describing a single idea, is, on a macro scale, a
mess. I had a very hard time following, and the figures were quite
tough-going. I think this paper should probably get in, but I think it
should be given a good clean up first, at least for a humble
neuro-theorist, rather than a dynamical systems guru, to understand.
>

### Questions:

> My, many, questions and confusions ended up being in the weaknesses section.
>