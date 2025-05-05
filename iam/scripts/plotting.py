import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Optional


sns.set(style="darkgrid", palette="muted", font="serif")
plt.rcParams.update(plt.rcParamsDefault)


def clean_tick(value: float) -> float:
    if abs(value) >= 1:
        return round(value)  # Whole number
    else:
        return round(value, 1)  # One decimal place


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



def plot_single_motif_trajectories(
    trajectories_source: np.ndarray,
    transformed_trajectories: np.ndarray,
    trajectories_target: np.ndarray,
    source_name: str = "Source",
    trajectory_colors = {"target": "firebrick", "source": "mediumseagreen", "asymptotic_target": "#d3869b", "asymptotic_source": "seagreen"},
    asymptotic_target: Optional[np.ndarray] = None,
    asymptotic_source_transformed: Optional[np.ndarray] = None,
    num_points: int = 10,
    bounds: tuple | str = (2.0, 2.0),
    alpha: float = 0.7,
    which_axis: str = "both",
    ax_source: Optional[plt.Axes] = None,
    ax_target: Optional[plt.Axes] = None,
    save_name: Optional[str] = None,
    show_fig: bool = True,
):
    assert which_axis in {"first", "second", "both"}

    if isinstance(bounds, str) and bounds == 'from_targ_traj':
        # Assume trajectories_target shape: (batch_size, time_steps, dim)
        min_val = np.amin(trajectories_target)  # shape: (dim,)
        max_val = np.amax(trajectories_target)  # shape: (dim,)
        max_abs = np.maximum(np.abs(min_val), np.abs(max_val))  # symmetric bound
        bounds = (max_abs, max_abs)

    target_color = trajectory_colors["target"]
    source_color = trajectory_colors["source"]
    asymptotic_target_color = trajectory_colors["asymptotic_target"]
    asymptotic_source_color = trajectory_colors["asymptotic_source"]

    if not ax_source and not ax_target:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax_target = ax

    K, T, N = transformed_trajectories.shape
    num_points = min(num_points, K)

    if which_axis in ("first", "both") and ax_source is not None:
        ax_source.set_xlim([-bounds[0], bounds[0]])
        ax_source.set_ylim([-bounds[1], bounds[1]])
        ax_source.set_xlabel(r'$x$', fontsize=14)
        ax_source.set_ylabel(r'$y$', fontsize=14)
        ax_source.grid(True)
        for i in range(num_points):
            ax_source.plot(trajectories_source[i, :, 0], trajectories_source[i, :, 1], color=source_color, alpha=alpha, linestyle="--",)
            ax_source.plot(transformed_trajectories[i, :, 0], transformed_trajectories[i, :, 1], color=source_color,  alpha=alpha)
        ax_source.set_title(f'{source_name}: Source & Transformed')

    if which_axis in ("second", "both") and ax_target is not None:
        ax_target.set_xlim([-bounds[0], bounds[0]])
        ax_target.set_ylim([-bounds[1], bounds[1]])
        ax_target.set_xlabel(r'$x$', fontsize=14)
        ax_target.set_ylabel(r'$y$', fontsize=14)
        ax_target.grid(True)
        for i in range(num_points):
            if trajectories_target.shape[0] >= num_points:
                ax_target.plot(trajectories_target[i, :, 0], trajectories_target[i, :, 1], color=target_color, alpha=alpha)
            ax_target.plot(transformed_trajectories[i, :, 0], transformed_trajectories[i, :, 1], color=source_color, alpha=alpha)
        if asymptotic_target is not None:
            ax_target.plot(asymptotic_target[:, 0], asymptotic_target[:, 1], color=asymptotic_target_color, linewidth=2, alpha=1.0)
        if asymptotic_source_transformed is not None:
            ax_target.plot(asymptotic_source_transformed[:, 0], asymptotic_source_transformed[:, 1], color=asymptotic_source_color, linewidth=2, alpha=1.0)
        ax_target.set_title(f'{source_name}')
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    if show_fig:
        plt.show()


def plot_trajectories_allmotifs(
    trajectories_source_list,
    transformed_trajectories_list,
    trajectories_target,
    asymptotic_trajectories_list=None,
    num_points=10,
    bounds=(2.0, 2.0),
    alpha=0.7,
    target_color='firebrick',
    base_colors=["mediumseagreen", "#00BFFF", "#8B008B"],
    asymptotic_colors=['#d3869b', 'seagreen', 'navy', 'purple'],
    source_names=["Source 1", "Source 2", "Source 3"],
    save_name=None,
    show_fig=True,
    which_axis: str = "both"
):
    num_motifs = len(trajectories_source_list)
    num_subplots = 2 if which_axis == "both" else 1

    fig, axes = plt.subplots(
        num_motifs, num_subplots,
        figsize=(6 * num_subplots, 4 * num_motifs),
        squeeze=False,
        sharex=True, sharey=True
    )

    for i in range(num_motifs):
        color = base_colors[i % len(base_colors)]
        asym_color = asymptotic_colors[i % len(asymptotic_colors)]
        source_name = source_names[i] if i < len(source_names) else f"Source {i+1}"
        asym = asymptotic_trajectories_list[i + 1] if asymptotic_trajectories_list else None

        ax_source = axes[i, 0] if which_axis in ("first", "both") else None
        ax_target = axes[i, 1] if which_axis == "both" else axes[i, 0]

        plot_single_motif_trajectories(
            trajectories_source=trajectories_source_list[i],
            transformed_trajectories=transformed_trajectories_list[i],
            trajectories_target=trajectories_target,
            source_name=source_name,
            target_color=target_color,
            source_color=color,
            asymptotic_trajectory=asym,
            asymptotic_color=asym_color,
            num_points=num_points,
            bounds=bounds,
            alpha=alpha,
            which_axis=which_axis,
            ax_source=ax_source,
            ax_target=ax_target,
        )

    if asymptotic_trajectories_list is not None:
        ax_target = axes[0, 1] if which_axis == "both" else axes[0, 0]
        ax_target.plot(
            asymptotic_trajectories_list[0][:, 0],
            asymptotic_trajectories_list[0][:, 1],
            color=asymptotic_colors[0],
            linewidth=2,
            alpha=1.0,
            label='Asymptotic target'
        )

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    if show_fig:
        plt.show()

    return fig, axes





def plot_trajectories_3d(
    trajectories_list,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    elev: float = 20,
    azim: float = 30,
) -> None:
    """
    Visualizes multiple sets of 3D trajectories in a 3D plot.

    :param trajectories_list: A list of tensors, each of shape (num_trajectories x num_time_points x 3).
    :param colors: Optional list of colors for each set of trajectories.
    :param labels: Optional list of legend labels for each trajectory set.
    :param elev: Elevation angle for the 3D view. Default is 20.
    :param azim: Azimuth angle for the 3D view. Default is 30.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        colors = ['C' + str(i) for i in range(len(trajectories_list))]

    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(trajectories_list))]

    legend_elements = []

    for idx, trajectories in enumerate(trajectories_list):
        color = colors[idx % len(colors)]
        label = labels[idx]
        initial_conditions = trajectories[:, 0, :]

        for i in range(trajectories.shape[0]):
            ax.plot(
                trajectories[i, :, 0],
                trajectories[i, :, 1],
                trajectories[i, :, 2],
                color=color,
                alpha=0.8,
            )
            ax.scatter(
                initial_conditions[i, 0],
                initial_conditions[i, 1],
                initial_conditions[i, 2],
                color='black',
                marker='o',
                s=50,
                alpha=0.8,
            )

        # One legend entry per trajectory set
        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))

    # Labeling
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Only show min and max ticks (rounded)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        locs = axis.get_data_interval()
        ticks = [clean_tick(locs[0]), clean_tick(locs[1])]
        axis.set_ticks(ticks)

    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()


def plot_transformed_vector_field(transformed_points, transformed_vector_field):
    """
    Plot the transformed vector field using the homeomorphism.
    """

    # Plot the transformed vector field
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(transformed_points[:, 0].detach().numpy(), transformed_points[:, 1].detach().numpy(),
               transformed_vector_field[:, 0].detach().numpy(), transformed_vector_field[:, 1].detach().numpy(),
               angles='xy', scale_units='xy', scale=1)
    #plt.xlabel("x")
    #plt.ylabel("y")

    #TODO: add target vector field
    # XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    # with torch.no_grad():
    #     dXY = vdp_system.forward(t=torch.tensor(0.0), x=XY).numpy()
    # U = dXY[:, 0].reshape(X.shape)
    # V = dXY[:, 1].reshape(Y.shape)
    # ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=5, color='red')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid()
    plt.show()