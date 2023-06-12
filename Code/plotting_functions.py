import os, sys
import glob
current_dir = os.path.dirname(os.path.realpath('__file__'))
        
import numpy as np
import numpy.ma as ma
from itertools import chain, combinations, permutations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import proplot as pplt
import seaborn as sns

from utils import * 

#1D version
def plot_trial_1D(trial_i, x, y, yhat, training_kwargs, file_name, eps = 0.04, padding = 5, maxT="full", subplots=111, trial_params=[]):
    pplt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    eps = 0.01
    padding = 5
    if maxT=="full":
        maxT = y[trial_i,:,0].shape[0]
    else:
        maxT = np.max([np.argmax(y[trial_i,:,0]), np.argmax(y[trial_i,:,1])])+padding
    times = np.linspace(0, maxT*training_kwargs['dt'], maxT)

    ax.plot(times, -1+eps+np.where(x[trial_i,:maxT,0]>0., x[trial_i,:maxT,0], np.nan), '|', markersize=14, color='b', label = "Clicks Left")
    ax.plot(times, eps+np.where(y[trial_i,:maxT,0]>0., 0, np.nan), 'b-o', markersize=14, markerfacecolor='none') #target
    ax.plot(times, eps+np.where(yhat[trial_i,:maxT,0]>eps, 0, np.nan), 'b-x', markersize=10, markerfacecolor='none') #output
    ax.fill_between(times, 0, 1, where=np.where(x[trial_i,:maxT,2], True, False),
                    facecolor='green', alpha=0.5, transform=trans)

    ax.plot(times, -1-eps+np.where(x[trial_i,:maxT,1]>0., x[trial_i,:maxT,1], np.nan), '|', markersize=14, color='r', label = "Clicks Right")
    ax.plot(times, -eps+np.where(y[trial_i,:maxT,1]>0., 0, np.nan), 'r-o', markersize=14, markerfacecolor='none')
    ax.plot(times, -eps+np.where(yhat[trial_i,:maxT,1]>eps, 0, np.nan), 'r-x', markersize=10, markerfacecolor='none')
    ax.fill_between(times, 0, 1, where=np.where(x[trial_i,:maxT,3], True, False),
                    facecolor='green', alpha=0.5, transform=trans)

    ax.set_xlabel("Time (ms)")
    ax.set_yticks([eps, -eps])
    ax.set_yticklabels(["Left", "Right"])
    ax.set_ylim([-2*eps, eps*2])
    ax.set_ylabel("Side")

    r_patch = mlines.Line2D([], [], color='blue', marker='|', linestyle='None',
                              markersize=10, label='Clicks Right')
    l_patch = mlines.Line2D([], [], color='red', marker='|', linestyle='None',
                              markersize=10, label='Clicks Left')

    r_target = mlines.Line2D([], [], color='blue', marker='o', markerfacecolor='none', linestyle='None',
                              markersize=10, label='Target Left')
    l_target = mlines.Line2D([], [], color='red', marker='o', markerfacecolor='none', linestyle='None',
                              markersize=10, label='Target Left')

    r_output = mlines.Line2D([], [], color='blue', marker='x', markerfacecolor='none', linestyle='None',
                              markersize=8, label='Output Left')
    l_output = mlines.Line2D([], [], color='red', marker='x', markerfacecolor='none', linestyle='None',
                              markersize=8, label='Output Left')

    ax.legend(handles=[r_patch, l_patch,
                       r_target, l_target,
                      r_output, l_output],
              loc='upper center', bbox_to_anchor=(0.5, -0.24),
              fancybox=True, shadow=True, ncol=2)

def plot_trial(trial_i, x, y, yhat, training_kwargs, file_name, eps = 0.04, padding = 5, maxT="full", subplots=111):
    # label_size = 12
    # pplt.rc['tick.labelsize'] = label_size 
    # pplt.rc['axes.labelsize'] = label_size + 3
    # sns.set_context("poster", font_scale = 1, rc={"grid.linewidth": 5})
    fig = pplt.figure(figsize=(6,4))
    ax = plt.subplot(subplots)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    if maxT=="full":
        maxT = y[trial_i,:,0].shape[0]
    else:
        maxT = np.max([np.argmax(y[trial_i,:,0]), np.argmax(y[trial_i,:,1])])+padding
    times = np.linspace(0, maxT*training_kwargs['dt'], maxT)

    ax.plot(times, eps+np.where(x[trial_i,:maxT,0]>0., x[trial_i,:maxT,0], np.nan), '|', color='b', label = "Clicks Left")
    ax.plot(times, y[trial_i,:maxT,0], 'bs', alpha=0.5, markersize=5, label='Target Left')
    ax.plot(times, yhat[trial_i,:maxT,0], 'b+', alpha=0.5, markersize=15, label='Output Left')
    ax.fill_between(times, 0, 1+eps, where=np.where(x[trial_i,:maxT,2], True, False),
                    facecolor='green', alpha=0.5, transform=trans)
    if training_kwargs['fixed_cue_onsetandduration'][1]==1:
        fill = np.zeros((maxT))
        onset_time = np.argwhere(x[trial_i, :, 2] == 1)[0][0]
        fill[onset_time-2:onset_time+2] = 1.
        ax.fill_between(times, 0, 1+eps, where=fill,
                        facecolor='green', alpha=0.5, transform=trans)

    ax.plot(times, -eps+np.where(x[trial_i,:maxT,1]>0., x[trial_i,:maxT,1], np.nan), '|', color='r', label = "Clicks Right")
    ax.plot(times, y[trial_i,:maxT,1], 'rs', alpha=0.5, markersize=5, label='Target Right')
    ax.plot(times, yhat[trial_i,:maxT,1], 'r+', alpha=0.5, markersize=15, label='Output Right')
    ax.fill_between(times, 0, 1+2*eps, where=np.where(x[trial_i,:maxT,3], True, False),
                    facecolor='green', alpha=0.5, transform=trans)
    if training_kwargs['fixed_cue_onsetandduration'][1]==1:
        fill = np.zeros((maxT))
        onset_time = np.argwhere(x[trial_i, :, 3] == 1)[0][0]
        fill[onset_time-2:onset_time+2] = 1.
        ax.fill_between(times, 0, 1+2*eps, where=fill,
                        facecolor='green', alpha=0.5, transform=trans)

    ax.set_xlabel("Time (ms)")
    ax.set_yticks([0, 1])
    ax.set_ylim([0, 1+3*eps])

    target = mlines.Line2D([], [], color='k', marker='s', linestyle='None',
                              markersize=5, label='Target')
    output = mlines.Line2D([], [], color='k', marker='+', linestyle='None',
                              markersize=8, label='Output')
    click = mlines.Line2D([], [], color='k', marker='|', linestyle='None',
                              markersize=8, label='Click')
    leg = plt.legend(handles = [target, output, click], 
              loc='upper center', bbox_to_anchor=(0.25, -0.25),
              fancybox=True, shadow=True, ncol=3)

    left = mlines.Line2D([], [], color='b', linewidth='6',
                              markersize=5, label='Left')
    right = mlines.Line2D([], [], color='r', linewidth='6',
                              markersize=5, label='Right')
    ax.add_artist(leg)
    plt.legend(handles = [left, right], 
              loc='upper center', bbox_to_anchor=(0.75, -0.25),
              fancybox=True, shadow=True, ncol=3)


    plt.savefig(training_kwargs['figures_path'] + "/" + file_name + ".pdf")
    plt.savefig(training_kwargs['figures_path'] + "/" + file_name + ".png")
    
    return fig
    

    
def plot_losses(epoch_losses, epoch_val_losses, training_kwargs):
    fig = plt.figure(figsize=(6,6))
    plt.plot(epoch_losses, 'x', label="Training loss")
    plt.plot(epoch_val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(training_kwargs['figures_path'] + '/losses.pdf')
    plt.close()
    

    
def plot_rcs(RCs, expnbin, maxval, save_folder=None):
    nbins = 2**expnbin
    delta = 1/nbins
    x_max = 2**expnbin
    y_max = 2**expnbin
    xs = np.arange(1, x_max+1)
    ys = np.arange(1, y_max+1)

    size = [400, 400] # in pixels
    dpi = 3.5*(2**(10-expnbin))
    figsize = [i / dpi for i in size]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim([0, x_max + 1])
    ax.set_ylim([0, y_max + 1]) 

    plt.xlabel("Hidden 0", size=5)
    plt.ylabel("Hidden 1", size=5)
    ax.set_xticks([0,2**(expnbin-1),2**expnbin])
    ax.set_yticks([0,2**(expnbin-1),2**expnbin])
    ax.set_xticklabels([0.,maxval/2,maxval], size=5)
    ax.set_yticklabels([0.,maxval/2,maxval], size=5)

    marker = markers.MarkerStyle(marker='s')
    for i in range(len(RCs)):
        for x in np.array(RCs[i]):
            x, y = x[:2]
            plt.scatter([x-delta], [y-delta], color='green', marker=marker)
    if save_folder != None:
        plt.savefig(save_folder + "/wc_inv_cubical_%s.pdf"%expnbin)
        
        
        
def plot_cohdir_trajectories_2(trajectories, unstable_fixedpoint, trial_params, 
                               coherence_list, training_kwargs, N_trial_plot=0, point_size = 2, label_text="Neuron",
                               cmaps=[plt.get_cmap('blues') , plt.get_cmap('reds')],
                               coherence_labels=None,
                               save_name='trajectories_fixedduronset_cohdir_condmean'):
    """
    Args: 
    N_trial_plot: to add N_trial_plot random trajectories
    
    """
    flat_trajectories = trajectories.reshape((-1, trajectories.shape[-1]))
    x_min, x_max = np.min(flat_trajectories[:,0]), np.max(flat_trajectories[:,0])
    y_min, y_max = np.min(flat_trajectories[:,1]), np.max(flat_trajectories[:,1])
    x_padding = (x_max-x_min)/10
    y_padding = (y_max-y_min)/10
    
    if coherence_labels==None:
        coherence_labels = coherence_list
    
    directions = np.array([trial_params[i]['direction'] for i in range(trial_params.shape[0])])
    coherences = np.array([trial_params[i]['coherence'] for i in range(trial_params.shape[0])])
    
    fig, axes = plt.subplots(2, len(coherence_list), figsize=(10, 5), sharey=True, sharex=True)
    cmap1, cmap2 = cmaps 

    trials_per_cohanddir = [[[], []]]*len(coherence_list) #list of lists: per coherence, then perdirection
    for coh_i,coherence in enumerate(coherence_list):
        for direction in [0,1]:
            ax_idx = coh_i*2 + direction
            arr1 = np.where(coherences==coherence)[0]
            arr2 = np.where(directions==direction)[0]
            intersection = [idx for idx in arr1 if idx in arr2]
            for trial_i in intersection:
                    trials_per_cohanddir[coh_i][direction].append(trajectories[trial_i,:,:])
            # trials_per_cohanddir[coh_i][direction] = trajectories[intersection,:,:]

            cond_mean = np.mean(trials_per_cohanddir[coh_i][direction], axis=0)
            # cond_mean = np.mean(np.array(trials_per_cohanddir[coh_i][direction]), axis=0)
            sc = axes[direction, coh_i].scatter(cond_mean[:,0], cond_mean[:,1], s=1, marker='o',
                                           vmin=0, vmax=training_kwargs['T'], cmap=cmap2,
                                          c=np.linspace(0, training_kwargs['T'], trajectories.shape[1]))

            if N_trial_plot>0:
                random_trials = np.random.choice(intersection, N_trial_plot)
                for trial_i in random_trials:
                    axes[direction, coh_i].plot(trajectories[trial_i,:,:][:,0], trajectories[trial_i,:,:][:,1],
                                                '-', color='0.8', linewidth=.5, zorder=1)
                    sc2 = axes[direction, coh_i].scatter(trajectories[trial_i,:,:][:,0],
                                     trajectories[trial_i,:,:][:,1],
                                     s=point_size, vmin=0, vmax=training_kwargs['T'], cmap=cmap1,
                                     c=np.linspace(0, training_kwargs['T'], trajectories.shape[1]), zorder=2)

            if np.all(unstable_fixedpoint) != None:
                axes[direction, coh_i].scatter(unstable_fixedpoint[0], unstable_fixedpoint[1], s=50, c='m', marker='x', zorder=2)
            if direction == 0:
                axes[direction, coh_i].set_title("%s"%coherence_labels[coh_i])

            axes[direction, coh_i].set_xlim([x_min-x_padding, x_max+x_padding])
            axes[direction, coh_i].set_ylim([y_min-y_padding, y_max+y_padding])
            axes[direction, coh_i].set_xlabel(label_text + " 1")
            if direction==0:
                axes[direction, coh_i].set_ylabel("Right \n \n "+label_text+" 2")
            else:
                axes[direction, coh_i].set_ylabel("Left \n \n "+label_text+" 2")

    cb_ax = fig.add_axes([.91,.124,.02,.754])
    cb_ax2 = fig.add_axes([.93,.124,.02,.754])
    fig.colorbar(sc, cax=cb_ax, fraction=0.5, pad=0.04).set_ticks([])
    fig.colorbar(sc2, cax=cb_ax2, fraction=0.5, pad=0.04).set_label('Time (ms)',rotation=90)
    fig.supylabel('Direction')
    fig.suptitle("    Coherence")
    for ax in plt.gcf().axes:
        try:
            ax.label_outer()
        except:
            pass
    
    if N_trial_plot>0:
        plt.savefig(training_kwargs['figures_path'] + '\\'+save_name+'_examples.pdf', bbox_inches="tight")
    else:
        plt.savefig(training_kwargs['figures_path'] + '\\'+save_name+'.pdf', bbox_inches="tight")
        
        
def plot_cohdir_trajectories_pc(trajectories, unstable_fixedpoint, trial_params, 
                               coherence_list, training_kwargs, N_trial_plot=0, point_size = 2, label_text="Neuron",
                               cmaps=[plt.get_cmap('blues') , plt.get_cmap('reds')],
                            coherence_labels=['26:14', '31:9', '37:3', '39:1'],
                               save_name='trajectories_fixedduronset_cohdir_condmean'):
    """
    Args: 
    N_trial_plot: to add N_trial_plot random trajectories
    
    """
    flat_trajectories = trajectories.reshape((-1, trajectories.shape[-1]))
    x_min, x_max = np.min(flat_trajectories[:,0]), np.max(flat_trajectories[:,0])
    y_min, y_max = np.min(flat_trajectories[:,1]), np.max(flat_trajectories[:,1])
    x_padding = (x_max-x_min)/10
    y_padding = (y_max-y_min)/10
    
    directions = np.array([np.where(trial_params[i]['ratio']>1.,1,0) for i in range(trial_params.shape[0])])
    ratios = np.array([np.abs(trial_params[i]['ratio']) for i in range(trial_params.shape[0])])

    fig, axes = plt.subplots(2, len(coherence_list), figsize=(10, 5), sharey=True, sharex=True)
    cmap1, cmap2 = cmaps 

    trials_per_cohanddir = [[[], []]]*len(coherence_list) #list of lists: per coherence, then perdirection
    for coh_i,coherence in enumerate(coherence_list):
        for direction in [0,1]:
            ax_idx = coh_i*2 + direction
            arr1 = np.where(ratios==coherence)[0]
            arr2 = np.where(directions==direction)[0]
            intersection = [idx for idx in arr1 if idx in arr2]
            for trial_i in intersection:
                    trials_per_cohanddir[coh_i][direction].append(trajectories[trial_i,:,:])
            # trials_per_cohanddir[coh_i][direction] = trajectories[intersection,:,:]

            # cond_mean = np.mean(trials_per_cohanddir[coh_i][direction], axis=0)
            cond_mean = np.mean(np.array(trials_per_cohanddir[coh_i][direction]), axis=0)
            sc = axes[direction, coh_i].scatter(cond_mean[:,0], cond_mean[:,1], s=1, marker='o',
                                           vmin=0, vmax=trajectories.shape[1], cmap=cmap2,
                                          c=np.linspace(0, trajectories.shape[1], trajectories.shape[1]), zorder=3)

            if N_trial_plot>0:
                random_trials = np.random.choice(intersection, N_trial_plot)
                for trial_i in random_trials:
                    axes[direction, coh_i].plot(trajectories[trial_i,:,:][:,0], trajectories[trial_i,:,:][:,1],
                                                '-', color='0.8', linewidth=.5, zorder=1)
                    sc2 = axes[direction, coh_i].scatter(trajectories[trial_i,:,:][:,0],
                                     trajectories[trial_i,:,:][:,1],
                                     s=point_size, vmin=0, vmax=trajectories.shape[1], cmap=cmap1,
                                     c=np.linspace(0, trajectories.shape[1], trajectories.shape[1]), zorder=2)

            if np.all(unstable_fixedpoint) != None:
                axes[direction, coh_i].scatter(unstable_fixedpoint[0], unstable_fixedpoint[1], s=50, c='m', marker='x', zorder=2)
            if direction == 0:
                axes[direction, coh_i].set_title("%s"%coherence_labels[coh_i])

            axes[direction, coh_i].set_xlim([x_min-x_padding, x_max+x_padding])
            axes[direction, coh_i].set_ylim([y_min-y_padding, y_max+y_padding])
            axes[direction, coh_i].set_xlabel(label_text + " 1")
            if direction==0:
                axes[direction, coh_i].set_ylabel("Right \n \n "+label_text+" 2")
            else:
                axes[direction, coh_i].set_ylabel("Left \n \n "+label_text+" 2")

    cb_ax = fig.add_axes([.91,.124,.02,.754])
    cb_ax2 = fig.add_axes([.93,.124,.02,.754])
    fig.colorbar(sc, cax=cb_ax, fraction=0.5, pad=0.04).set_ticks([])
    fig.colorbar(sc2, cax=cb_ax2, fraction=0.5, pad=0.04).set_label('Time (ms)',rotation=90)
    fig.supylabel('Direction')
    fig.suptitle("    Coherence")
    for ax in plt.gcf().axes:
        try:
            ax.label_outer()
        except:
            pass
    
    if N_trial_plot>0:
        plt.savefig(training_kwargs['figures_path'] + '\\'+save_name+'_examples.pdf', bbox_inches="tight")
    else:
        plt.savefig(training_kwargs['figures_path'] + '\\'+save_name+'.pdf', bbox_inches="tight")