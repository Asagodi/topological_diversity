import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
import seaborn as sns
from typing import List, Optional
import pandas as pd
plt.rcParams['xtick.labelsize'] = 14  # font size
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams["font.family"] = "serif"
mpl.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts (editable in Illustrator)
mpl.rcParams['ps.fonttype'] = 42   # Same for EPS
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
# plt.rcParams.update(plt.rcParamsDefault) # Reset to default settings
plt.rcParams.update({'font.size': 12, 'text.usetex': True,'text.latex.preamble': r'\usepackage{amsfonts}'})


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
    plot_start: bool = True, start_color: str = 'blue', start_size: int = 50,
    plot_end: bool = True, end_color: str = 'red', end_size: int = 50,
    save_name: Optional[str] = None,
    save_dir: Optional[str] = None,
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
            if plot_start:
                ax.scatter(
                    initial_conditions[i, 0],
                    initial_conditions[i, 1],
                    initial_conditions[i, 2],
                    color=start_color,
                    marker='o',
                    s=start_size,
                    alpha=0.8,
                )
            if plot_end:
                ax.scatter(
                    trajectories[i, -1, 0],
                    trajectories[i, -1, 1],
                    trajectories[i, -1, 2],
                    color=end_color,
                    marker='o',
                    s=end_size,
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
    if save_name:
        save_path = f"{save_dir}/{save_name}" if save_dir else save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def plot_trajectories_with_surface(
    trajectories ,  # (B, T, 3)
    surface_points = None,  # (P, 3)
    surface_color: str = 'lightgray',
    surface_alpha: float = 0.5,
    traj_color: str = 'blue',
    label: Optional[str] = None,
    elev: float = 20,
    azim: float = 30,
    plot_start: bool = True, start_color: str = 'blue', start_size: int = 50,
    plot_end: bool = True, end_color: str = 'red', end_size: int = 50,
    save_name: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """
    Plots a batch of 3D trajectories and optionally a surface defined by 3D points.

    :param trajectories: Tensor of shape (B, T, 3)
    :param surface_points: Optional tensor of shape (P, 3) for surface triangulation.
    """
    assert trajectories.ndim == 3 and trajectories.shape[2] == 3, "Trajectories must be of shape (B, T, 3)"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    B, T, _ = trajectories.shape
    traj_np = trajectories.detach().cpu().numpy()

    for i in range(B):
        ax.plot(traj_np[i, :, 0], traj_np[i, :, 1], traj_np[i, :, 2], color=traj_color, alpha=0.8)

        if plot_start:
            ax.scatter(*traj_np[i, 0], color=start_color, s=start_size, alpha=0.8)

        if plot_end:
            ax.scatter(*traj_np[i, -1], color=end_color, s=end_size, alpha=0.8)

    if label:
        legend_elements = [Line2D([0], [0], color=traj_color, lw=2, label=label)]
        ax.legend(handles=legend_elements)

    if surface_points is not None:
        assert surface_points.shape[1] == 3, "Surface points must be (P, 3)"
        points_np = surface_points.detach().cpu().numpy()
        tri = Delaunay(points_np[:, :2])  # Triangulate in XY, use Z for height
        verts = [points_np[simplex] for simplex in tri.simplices]
        poly = Poly3DCollection(verts, color=surface_color, alpha=surface_alpha)
        ax.add_collection3d(poly)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        locs = axis.get_data_interval()
        ticks = [clean_tick(locs[0]), clean_tick(locs[1])]
        axis.set_ticks(ticks)

    plt.tight_layout()
    if save_name:
        save_path = f"{save_dir}/{save_name}" if save_dir else save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
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

def plot_deformed_cylinder_surface(ax, surface_points, color='lightblue', alpha=0.5):
    xyz = surface_points
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Step 1: Estimate angular coordinate θ
    theta = np.arctan2(y, x)  # still valid even for deformed cylinders

    # Step 2: Build 2D parameterization (θ, z)
    param_2d = np.stack([theta, z], axis=1)

    # Step 3: Triangulate in parameter space
    tri = Delaunay(param_2d)

    # Step 4: Map triangles back to 3D space
    verts = [xyz[simplex] for simplex in tri.simplices]
    poly = Poly3DCollection(verts, facecolor=color, alpha=alpha)
    ax.add_collection3d(poly)

    ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])



def radial_scale(r: np.ndarray, r_threshold: float = 1.1) -> np.ndarray:
    """Scale factor that increases up to r_threshold and flattens after."""
    scale = np.ones_like(r)
    mask = r <= r_threshold
    scale[mask] = r_threshold / (r[mask] + 1e-8)  # Avoid division by zero
    scale[~mask] = 1.0
    return scale
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrow


def plot_vector_field_fixedquivernorm_speedcontour(
    X, Y, U, V, trajectories_pertring,
    title=None,
    background_color='white',
    scale=1.0,
    color='teal',
    cmap='plasma',
    traj_color='k',
    figsize=(6, 6),
    alpha=0.5,
    min_val_plot=1.25,
    vmin_log=-6,
    vmax_log=4,
    level_step=1,
    smoothing_sigma=1.0,
    upsample_factor=8,
    draw_arrows=True,
    save_name=None
):
    """
    Plot a vector field with scaled arrows and log-speed contours at fixed increments.

    Parameters:
        X, Y          : 2D meshgrid arrays for coordinates
        U, V          : 2D arrays for vector components at (X, Y)
        trajectories_pertring : array of shape (N_traj, T, 2) for plotting trajectories
        title         : plot title
        background_color : 'white' or 'black'
        scale         : global scaling factor for arrow sizes (currently unused)
        color         : not used (reserved for future vector coloring)
        cmap          : colormap for log-speed contours
        traj_color    : color for trajectory lines
        figsize       : figure size tuple
        alpha         : transparency for trajectory lines
        min_val_plot  : plot limits (xlim and ylim)
        vmin_log, vmax_log : bounds for log-speed color levels
        level_step    : interval between contour levels (e.g., 1 for integer steps)
        smoothing_sigma : standard deviation for optional Gaussian smoothing
        upsample_factor : factor to interpolate data grid for smoother contour plot
    """
    # Compute speed and log-speed
    speed = np.sqrt(U**2 + V**2)
    log_speed = speed
    log_speed = np.log10(speed + 1e-8)

    # Scale vector magnitude radially
    R = np.sqrt(X**2 + Y**2)
    scale_factor = radial_scale(R)
    # U_scaled = U * scale_factor
    # V_scaled = V * scale_factor

    # Normalize vectors for uniform arrow lengths
    U_unit = U / (speed + 1e-8)
    V_unit = V / (speed + 1e-8)

    # Interpolate log speed to finer grid
    x_fine = np.linspace(X.min(), X.max(), X.shape[1] * upsample_factor)
    y_fine = np.linspace(Y.min(), Y.max(), Y.shape[0] * upsample_factor)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    log_speed_fine = griddata(
        (X.flatten(), Y.flatten()),
        log_speed.flatten(),
        (X_fine, Y_fine),
        method='cubic',
        fill_value=np.nan
    )
    log_speed_fine = gaussian_filter(log_speed_fine, sigma=smoothing_sigma)
    log_speed_fine = np.clip(log_speed_fine, vmin_log, vmax_log)

    # Define discrete contour levels
    levels = np.arange(vmin_log, vmax_log + level_step, level_step)

    # Set up figure and axis
    font_color = 'white' if background_color == 'black' else 'black'
    plt.rcParams['axes.facecolor'] = background_color
    plt.rcParams['figure.facecolor'] = background_color
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_facecolor(background_color)
    if title:
        ax.set_title(title, color=font_color)

    # Filled contour plot of log-speed
    contour = ax.contourf(
        X_fine, Y_fine, log_speed_fine,
        levels=levels,
        cmap=cmap,
        alpha=0.8
    )

    # Colorbar formatting
    cbar = plt.colorbar(contour, ax=ax, shrink=0.75)
    cbar.set_label('log speed', color=font_color)
    cbar.ax.yaxis.set_tick_params(color=font_color)
    plt.setp(cbar.ax.get_yticklabels(), color=font_color)

    # Quiver plot of normalized vectors
    ax.quiver(X, Y, U_unit, V_unit, color=font_color, scale=30)

    # Plot trajectories
    for traj in trajectories_pertring:
        ax.plot(traj[:, 0], traj[:, 1], color=traj_color, alpha=alpha)
        if draw_arrows:
                    # Add arrow in the middle
            mid_idx = int(len(traj) * 0.1)
            #print(mid_idx)
            if mid_idx + 1 < len(traj):
                x_start, y_start = traj[mid_idx]
                x_end, y_end = traj[mid_idx + 1]
                dx = x_end - x_start
                dy = y_end - y_start
            #     ax.quiver(
            #     x_start, y_start, dx, dy,
            #     angles='xy',
            #     scale_units='xy',
            #     scale=1,
            #     color=traj_color,
            #     width=0.025,
            #     headwidth=3,
            #     headlength=4,
            #     headaxislength=3.5,
            #     alpha=alpha
            # )
                ax.annotate(
                    '',
                    xy=(x_end, y_end),
                    xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='-|>', color=traj_color, lw=1.5),
                )
        #         arrow = FancyArrow(
        #     x_start, y_start, dx, dy,
        #     #arrowstyle='-|>',  # triangle-style
        #     width=0.01,
        #     length_includes_head=True,
        #     head_width=0.1,
        #     head_length=0.1,
        #     color=traj_color,
        #     alpha=alpha
        # )
        #ax.add_patch(arrow)
    # Axis limits and formatting
    ax.set_xlim(-min_val_plot, min_val_plot)
    ax.set_ylim(-min_val_plot, min_val_plot)
    ax.set_xticks([])
    ax.set_yticks([])

    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.show()

def plot_vector_field_coloredquivernorm(X, Y, U, V, trajectories_pertring, title=None, normalize_quivers=False,
                                        cmap='plasma', traj_color='orange', alpha=0.5, save_name=None, figsize=(6, 6),
                                        min_val_plot=1.25, vmin_log=None, vmax_log=None, background_color='white'):
    """Plot vector field with fixed-length quivers colored by log speed."""
    speed = np.sqrt(U**2 + V**2)
    log_speed = np.log(speed + 1e-8)
    if vmin_log is None:
        vmin_log = np.min(log_speed)
    if vmax_log is None:
        vmax_log = np.max(log_speed)
    log_speed = np.clip(log_speed, vmin_log, vmax_log)

    R = np.sqrt(X**2 + Y**2)
    scale_factor = radial_scale(R)
    U_scaled = U * scale_factor
    V_scaled = V * scale_factor

    # Normalize vectors for uniform arrow length
    if normalize_quivers:
        U = U / (speed + 1e-8)
        V = V / (speed + 1e-8)

    if background_color == 'black':
        font_color = 'white'
    else:
        font_color = 'black'
    # Plot setup
    plt.rcParams['axes.facecolor'] = background_color
    plt.rcParams['figure.facecolor'] = background_color
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_facecolor(background_color)
    ax.set_title(title, color=font_color)

    # Quiver plot colored by log speed
    quiv = ax.quiver(X, Y, U, V, log_speed,
                     cmap=cmap, clim=(vmin_log, vmax_log), scale=30)

    # Colorbar for log speed
    cbar = plt.colorbar(quiv, ax=ax, shrink=0.75)
    cbar.set_label('log speed', color=font_color)
    cbar.ax.yaxis.set_tick_params(color=font_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=font_color)

    # Plot trajectories
    for i in range(trajectories_pertring.shape[0]):
        ax.plot(trajectories_pertring[i, :, 0], trajectories_pertring[i, :, 1],
                color=traj_color, alpha=alpha)

    ax.set_xlim(-min_val_plot, min_val_plot)
    ax.set_ylim(-min_val_plot, min_val_plot)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)

    plt.show()










###plotting results
#jacobians
# Define consistent colors
JACOBIAN_NORM_COLOR = '#D2691E'   # Chocolate
TRAIN_LOSS_COLOR = '#6B8E23'      # Olive Green
TEST_LOSS_COLOR = '#4682B4'       # Steel Blue
POINT_ALPHA = 0.5

def plot_jacobian_norms(df: pd.DataFrame, save_dir: str, save_name: str = "jacobian_norms.pdf") -> None:
    sns.set(style="darkgrid", palette="muted", font="serif")

    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter points
    for _, row in df.iterrows():
        target_jacobian_norm = row['target_jacobian_norm']
        jacobian_norm = row['jacobian_norm']
        ax.plot(target_jacobian_norm, jacobian_norm, 'o', color=JACOBIAN_NORM_COLOR, alpha=POINT_ALPHA)

    # Identity line y = x
    all_vals = pd.concat([df['target_jacobian_norm'], df['jacobian_norm']])
    min_val, max_val = all_vals.min(), all_vals.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='$y = x$')

    ax.set_xlabel("Target Jacobian norm", fontsize=14)
    ax.set_ylabel("Learned Jacobian norm", fontsize=14)
    ax.legend()

    plt.savefig(f"{save_dir}/{save_name}", bbox_inches='tight')
    plt.show()

    sns.set(style="white", palette="deep", font="sans-serif")  # Reset seaborn


def plot_losses_and_jacobian_norms(
    df: pd.DataFrame,
    save_dir: str,
    plot_error_scale: str = "linear",
    save_name: str = "losses_and_jacobian_norms.pdf",
    value_name: str = "value",
    figsize: tuple = (5, 5)
) -> None:
    sns.set(style="white", palette="muted", font="serif")

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Plot losses
    for i, row in df.iterrows():
        value = row[value_name]
        train_loss = row['train_loss']
        test_loss = row['test_loss']
        ax1.plot(value, train_loss, 'o', color=TRAIN_LOSS_COLOR, alpha=POINT_ALPHA, label='Train' if i == 0 else "")
        ax1.plot(value, test_loss, 'o', color=TEST_LOSS_COLOR, alpha=POINT_ALPHA, label='Test' if i == 0 else "")

    # Plot Jacobian norm
    for i, row in df.iterrows():
        value = row[value_name]
        jacobian_norm = row['jacobian_norm']
        ax2.plot(value, jacobian_norm, 'o', color=JACOBIAN_NORM_COLOR, alpha=POINT_ALPHA,
                 label=r'Complexity ($\|\partial\Phi/\partial x - \mathbb{I}\|$)' if i == 0 else "")

    ax1.set_yscale(plot_error_scale)
    ax1.set_ylabel('Distance (MSE)', color=TRAIN_LOSS_COLOR)
    ax1.tick_params(axis='y', labelcolor=TRAIN_LOSS_COLOR)

    ax2.set_ylabel('Complexity (Jacobian norm)', color=JACOBIAN_NORM_COLOR)
    ax2.tick_params(axis='y', labelcolor=JACOBIAN_NORM_COLOR)

    ax1.legend(loc='upper left')
    ax2.legend(loc='right')

    plt.savefig(f"{save_dir}/{save_name}", dpi=300, bbox_inches='tight')
    plt.show()

    sns.set(style="white", palette="deep", font="sans-serif")  # Reset seaborn