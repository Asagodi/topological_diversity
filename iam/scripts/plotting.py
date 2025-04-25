import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set(style="darkgrid", palette="muted", font="serif")
plt.rcParams.update(plt.rcParamsDefault)


def plot_sampled_trajectories(trajectories, initial_conditions, bounds):
    fig, ax = plt.subplots(figsize=(4, 4))
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], color='firebrick', alpha=0.5)

    ax.scatter(initial_conditions[:, 0], initial_conditions[:, 1], color='lightcoral', label='Start', zorder=5)

    # Ensure equal scaling for both axes
    ax.set_xlim([bounds[0], bounds[1]])
    ax.set_ylim([bounds[0], bounds[1]])

    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.legend()
    plt.show()




# Function to plot the action of the homeomorphism with optional grid points
def plot_homeomorphism_action(original_samples, transformed_samples, bounds):
    # Create a subplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].set_xlim([bounds[0], bounds[1]])
    axes[0].set_ylim([bounds[0], bounds[1]])
    axes[1].set_xlim([bounds[0], bounds[1]])
    axes[1].set_ylim([bounds[0], bounds[1]])

    # Original space (before homeomorphism)
    axes[0].scatter(original_samples[:, 0].detach().numpy(), original_samples[:, 1].detach().numpy(), color='blue', label='Original Points')
    axes[0].quiver(original_samples[:, 0].detach().numpy(), original_samples[:, 1].detach().numpy(),
                   transformed_samples[:, 0].detach().numpy() - original_samples[:, 0].detach().numpy(),
                   transformed_samples[:, 1].detach().numpy() - original_samples[:, 1].detach().numpy(),
                   angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5)
    axes[0].set_title('Action of Homeomorphism (Original Space)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True)

    # Transformed space (after homeomorphism)
    axes[1].scatter(transformed_samples[:, 0].detach().numpy(), transformed_samples[:, 1].detach().numpy(), color='red', label='Transformed Points')
    axes[1].quiver(transformed_samples[:, 0].detach().numpy(), transformed_samples[:, 1].detach().numpy(),
                   original_samples[:, 0].detach().numpy() - transformed_samples[:, 0].detach().numpy(),
                   original_samples[:, 1].detach().numpy() - transformed_samples[:, 1].detach().numpy(),
                   angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5)
    axes[1].set_title('Inverse Action of Homeomorphism (Transformed Space)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_trajectories_comparison(original_trajectories, transformed_trajectories, bounds=(2.0, 2.0)):
    """
    Plot the original and transformed trajectories to compare the effect of the homeomorphism.

    :param original_trajectories: Original trajectories before applying the homeomorphism.
    :param transformed_trajectories: Transformed trajectories after applying the homeomorphism.
    :param num_points: Number of trajectories to plot.
    :param bounds: Tuple specifying the limits for x and y axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Set limits for the axes based on the bounds
    axes[0].set_xlim([-bounds[0], bounds[0]])
    axes[0].set_ylim([-bounds[1], bounds[1]])
    axes[1].set_xlim([-bounds[0], bounds[0]])
    axes[1].set_ylim([-bounds[1], bounds[1]])

    # Plot original trajectories (before homeomorphism)
    for i in range(len(original_trajectories)):
        axes[0].plot(original_trajectories[i][:, 0].detach().numpy(), original_trajectories[i][:, 1].detach().numpy(), color='blue')
    axes[0].set_title('Original LimitCycle Trajectories')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True)

    # Plot transformed trajectories (after homeomorphism)
    for i in range(len(transformed_trajectories)):
        axes[1].plot(transformed_trajectories[i][:, 0].detach().numpy(), transformed_trajectories[i][:, 1].detach().numpy(), color='red')
    axes[1].set_title('Transformed LimitCycle Trajectories (Under Homeomorphism)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True)

    # Add quiver plots showing the difference (transformed - original)
    # for i in range(min(num_points, len(original_trajectories))):
    #     # Compute differences for quiver plot
    #     dx = transformed_trajectories[i][:, 0] - original_trajectories[i][:, 0]
    #     dy = transformed_trajectories[i][:, 1] - original_trajectories[i][:, 1]
        # axes[0].quiver(original_trajectories[i][:, 0].detach().numpy(), original_trajectories[i][:, 1].detach().numpy(),
        #                dx.detach().numpy(), dy.detach().numpy(), angles='xy', scale_units='xy', scale=0.1, color='blue', alpha=0.5)
        # axes[1].quiver(transformed_trajectories[i][:, 0].detach().numpy(), transformed_trajectories[i][:, 1].detach().numpy(),
        #                -dx.detach().numpy(), -dy.detach().numpy(), angles='xy', scale_units='xy', scale=0.1, color='red', alpha=0.5)

    plt.tight_layout()
    plt.show()


import torch.optim as optim

def plot_trajectories_stt(trajectories_source, transformed_trajectories, trajectories_target, num_points=10, bounds=(2.0, 2.0),
                          color_source='royalblue', color_transformed='darkorange', color_target='crimson'):
    """
    Plot the original and transformed trajectories to compare the effect of the diffeomorphism.

    :param trajectories_source: Original trajectories before applying the diffeomorphism.
    :param transformed_trajectories: Transformed trajectories after applying the diffeomorphism.
    :param trajectories_target: Target trajectories.
    :param num_points: Number of trajectories to plot.
    :param bounds: Tuple specifying the limits for x and y axes.
    :param color_source: Color for source trajectories.
    :param color_transformed: Color for transformed trajectories.
    :param color_target: Color for target trajectories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Set axis limits
    for ax in axes:
        ax.set_xlim([-bounds[0], bounds[0]])
        ax.set_ylim([-bounds[1], bounds[1]])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)

    # Plot original and transformed trajectories (before transformation)
    for i in range(min(num_points, len(trajectories_source))):
        if i == 0:
            axes[0].plot(trajectories_source[i][:, 0],
                         trajectories_source[i][:, 1],
                         color=color_source, label='Source Trajectories')

            axes[0].plot(transformed_trajectories[i][:, 0],
                         transformed_trajectories[i][:, 1],
                         color=color_transformed, label='Transformed Trajectories')
        else:
            axes[0].plot(trajectories_source[i][:, 0],
                         trajectories_source[i][:, 1],
                         color=color_source)

            axes[0].plot(transformed_trajectories[i][:, 0],
                         transformed_trajectories[i][:, 1],
                         color=color_transformed)

    axes[0].set_title('Original & Transformed Trajectories')
    axes[0].legend()

    # Plot transformed and target trajectories (after transformation)
    for i in range(min(num_points, len(transformed_trajectories))):
        if i == 0:
            axes[1].plot(transformed_trajectories[i][:, 0],
                         transformed_trajectories[i][:, 1],
                         color=color_transformed, label='Transformed Trajectories')

            axes[1].plot(trajectories_target[i][:, 0],
                         trajectories_target[i][:, 1],
                         color=color_target, label='Target Trajectories')
        else:
            axes[1].plot(transformed_trajectories[i][:, 0],
                         transformed_trajectories[i][:, 1],
                         color=color_transformed)

            axes[1].plot(trajectories_target[i][:, 0],
                         trajectories_target[i][:, 1],
                         color=color_target)

    axes[1].set_title('Transformed & Target Trajectories')
    axes[1].legend()

    plt.tight_layout()
    plt.show()



def plot_trajectories_allmotifs(
    trajectories_source_list, transformed_trajectories_list, trajectories_target, asymptotic_trajectories_list=None,
    num_points=10, bounds=(2.0, 2.0), alpha=0.7,
    target_color='firebrick', 
    base_colors = ["mediumseagreen", "#00BFFF", "#8B008B"],  # Deep green → Bright cyan → Vibrant magenta
    asymptotic_colors=['#d3869b', 'seagreen', 'navy', 'purple'],  # target, 3 motifs 
    source_names = ["Source 1", "Source 2", "Source 3"],
    save_name=None, show_fig=True
):
    """
    Plot multiple sets of source and transformed trajectories with matched colors.

    - Left subplot: Different sources vs. their transformed counterparts (same color, solid vs. dashed).
    - Right subplot: Transformed vs. single target trajectories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)

    # Set axis limits and labels
    for ax in axes:
        ax.set_xlim([-bounds[0], bounds[0]])
        ax.set_ylim([-bounds[1], bounds[1]])
        ax.set_xlabel(r'$x$', fontsize=14)
        ax.set_ylabel(r'$y$', fontsize=14)
        ax.grid(True)

    # Use a color map for distinct source-transformed pairs
    num_sources = len(trajectories_source_list)
    color_map = mcolors.LinearSegmentedColormap.from_list("custom_colormap", base_colors, N=num_sources)

    # --- Left subplot: Sources vs. Transformed Trajectories (Matched Colors) ---
    for idx, (trajectories_source, transformed_trajectories) in enumerate(zip(trajectories_source_list, transformed_trajectories_list)):
        color = color_map(idx)  # Assign a unique color per source-transformed pair
        K, T, N = trajectories_source.shape  # Get shape

        for i in range(min(num_points, K)):
            source_traj = trajectories_source[i]  # Shape: (T, N)
            transformed_traj = transformed_trajectories[i]  # Shape: (T, N)

            if i == 0:
                axes[0].plot(source_traj[:, 0], source_traj[:, 1], color=color, alpha=alpha, label=source_names[idx])
                axes[0].plot(transformed_traj[:, 0], transformed_traj[:, 1], color=color, linestyle="--", alpha=alpha, label='Transformed '+source_names[idx])
            else:
                axes[0].plot(source_traj[:, 0], source_traj[:, 1], color=color, alpha=alpha)
                axes[0].plot(transformed_traj[:, 0], transformed_traj[:, 1], color=color, linestyle="--", alpha=alpha)
        

    #axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    #axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    xlims = axes[0].get_xlim()
    ticks = [xlims[0], 0, xlims[1]]  # Set ticks to include 0
    #axes[0].set_xticks(ticks)                        
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f"{x:g}" if x in ticks else ""  ))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f"{x:g}" if x in ticks else ""  ))

    axes[0].set_title('Source & Transformed Trajectories')

    # --- Right subplot: Transformed vs. Target trajectories ---
    K, T, N = trajectories_target.shape  # Ensure shape consistency

    for i in range(min(num_points, K)):
        target_traj = trajectories_target[i]  # Shape: (T, N)

        for idx, transformed_trajectories in enumerate(transformed_trajectories_list):
            transformed_traj = transformed_trajectories[i]  # Shape: (T, N)
            color = color_map(idx)  # Use same color as source

            if  i == 0:
                if idx == 0:
                    axes[1].plot(target_traj[:, 0], target_traj[:, 1], color=target_color, alpha=0.9, label='Target')
                else:
                    axes[1].plot(target_traj[:, 0], target_traj[:, 1], color=target_color, alpha=0.9)
                axes[1].plot(transformed_traj[:, 0], transformed_traj[:, 1], color=color, linestyle="--", alpha=alpha, label='Transformed '+source_names[idx])

            else:
                axes[1].plot(transformed_traj[:, 0], transformed_traj[:, 1], color=color, linestyle="--", alpha=alpha)
                axes[1].plot(target_traj[:, 0], target_traj[:, 1], color=target_color, alpha=0.9)

    # Plot asymptotic trajectories
            if i==0 and asymptotic_trajectories_list is not None:
                axes[1].plot(asymptotic_trajectories_list[idx+1][:, 0], asymptotic_trajectories_list[idx+1][:, 1], color=asymptotic_colors[idx+1], alpha=1, label='Asymptotic '+source_names[idx], zorder=100)
                if asymptotic_trajectories_list[idx+1].shape[0] == 1:
                    axes[1].plot(asymptotic_trajectories_list[idx+1][:, 0], asymptotic_trajectories_list[idx+1][:, 1], 'o', color=asymptotic_colors[idx+1], alpha=1, label='Asymptotic '+source_names[idx])
    if asymptotic_trajectories_list is not None:
        axes[1].plot(asymptotic_trajectories_list[0][:, 0], asymptotic_trajectories_list[0][:, 1], color=asymptotic_colors[0], alpha=1, label='Asymptotic target')

    #axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    #axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f"{x:g}" if x in ticks else "" ))
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f"{x:g}" if x in ticks else "" ))
    axes[1].set_title('Transformed & Target Trajectories')

    # Combine legends from both subplots into a single legend under the figure
    handles, labels = axes[0].get_legend_handles_labels()
    target_handle, target_label = axes[1].get_legend_handles_labels()
    # Add only the target handle and label to the legend (first label in right subplot)
    handles.append(target_handle[0])
    labels.append(target_label[0])
    #handles.extend(target_handle[-4:])  # Add the rest of the handles from the right subplot
    #labels.extend(target_label[-4:])  # Add the rest of the labels from the right subplot
    print(target_label)
    fig.legend(
        handles, labels, loc='center', ncol=4,  # ncol columns
        bbox_to_anchor=(0.5, -0.05),  # Position the legend below the subplots
        frameon=False, columnspacing=1,  # Adjust spacing between columns
        handlelength=2, fontsize=12  # Adjust handle size and font size
    )

    if asymptotic_trajectories_list is not None:
        fig.legend(
            handles=target_handle[2::2], 
            labels=target_label[2::2], 
            loc='center', ncol=4,
            bbox_to_anchor=(0.5, -0.13),  # lower than the first legend
            frameon=False, columnspacing=1,
            handlelength=2, fontsize=12
        )
    
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)

    if show_fig:
        plt.show()

    return fig, axes
