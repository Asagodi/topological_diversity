#!/bin/sh

cp ../figures/FenichelThm.pdf .
cp ../figures/inv_man/N100_si2_rho1.9_g0_fp4.8.12.pdf .
cp ../figures/inv_man/angular_losses2.pdf .
cp ../figures/inv_man/bio_rings.pdf .
cp ../figures/bla_parameter_space.pdf .
cp ../figures/inv_man/empj_onoff_perturbation.pdf .
cp ../figures/inv_man/fastslow_decomposition_m_normhyp.pdf .
cp ../figures/inv_man/fastslow_decomposition_outputs2.pdf .
cp ../figures/inv_man/im_all_3x3.pdf .
cp ../figures/inv_man/im_all_last2.pdf .
cp ../figures/inv_man/lara_bifurcations_simple.pdf .
cp ../figures/noorman_ring_N6_pert_allfxdpnts_allnorms.pdf .
cp ../figures/inv_man/performance2.pdf .
cp ../figures/ring_n6_perturbations_schematic.pdf .
cp ../figures/task_fig.pdf .
cp ../figures/inv_man/training_losses.pdf .
cp ../figures/inv_man/vf_on_ring.pdf .
latexpand neurips_2024.tex > arxiv.flt
sed "/graphicspath/d" arxiv.flt > arxiv.tex
latexmk -c arxiv
latexmk arxiv

# after this, append the bbl file to the main tex
# compress it and upload it!
