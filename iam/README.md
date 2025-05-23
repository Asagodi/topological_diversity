# Dynamical Archetype Analysis

These are the instructions for running the code for the paper **Dynamical Archetype Analysis**.

## Notebooks for main figures

- `nra_pert_exp.ipynb`: Fits the archetypes on the deformed and perturbed ring attractor for all deformation/perturbation sizes.
- `ra_pert_analysis.ipynb`: Generates Fig. 2 from these fits.
- `target_trajectories.ipynb`: Generates the trajectories of all the target systems (except for the RNN internal compass).
- `run_on_targets.ipynb`: Fits all the archetypes to all the target systems (in the code, archetype is still referred to as motif).
- `avi_run_train.ipynb`: Trains the models on the RNN internal compass target (N=64) for all the archetypes, and for N=128 and 256 for the ring attractor archetype.
- `avi_analysis.ipynb`: Generates Fig. 16 (referred to as Fig. 6 in the main paper due to a labeling error).
- `dis_mat.ipynb`: Plots the similarity-simplicity figure (Fig. 3). Also generates the plots in Supp.Sec.K3.
- `dsa_comparison.ipynb`: Generates the DSA plots for Fig. 4.
- `spe_comparison.ipynb`: Generates the SPE plots for Fig. 4.


## Supplementary Material

- `source_motifs.ipynb`: Displays the behavior of the different archetypes in our archetype library (Supp. Sec. E).