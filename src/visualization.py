import matplotlib.pyplot as plt
import os
import time
import numpy as np
from world import World
from world_objects import Location, Velocity
import imageio.v2 as imageio
from PIL import Image
import glob
from configs import VisualizationConfig
from particle_filter import TrackedTargetPF

# Global trajectory storage - reset using reset_trajectories()
_agent_trajectory = []
_landmark_trajectory = []


def visualize_world(world, vis_config: VisualizationConfig, filename=None, show_trajectories=True, collect_for_gif=True):
    """
    Visualize the world state in 2D (top-down view) and save it to a file.
    The filename will be saved inside vis_config.save_dir.
    If vis_config.save_dir is relative, it's assumed to be relative to CWD or an experiment dir.

    Args:
        world (World): The World object to visualize.
        vis_config (VisualizationConfig): Configuration for visualization settings.
        filename (str, optional): Optional filename for saving the plot (without directory).
        show_trajectories (bool): Whether to show the agent and landmark trajectories.
        collect_for_gif (bool): (Legacy) Indicates if the caller intends to use the frame for a GIF.

    Returns:
        str: Full path to the saved image file, or None if saving failed.
    """
    global _agent_trajectory, _landmark_trajectory
    max_trajectory_points = vis_config.max_trajectory_points

    if world.agent and world.agent.location:
        _agent_trajectory.append(
            (world.agent.location.x, world.agent.location.y))
    if world.true_landmark and world.true_landmark.location:
        _landmark_trajectory.append(
            (world.true_landmark.location.x, world.true_landmark.location.y))

    if len(_agent_trajectory) > max_trajectory_points:
        _agent_trajectory = _agent_trajectory[-max_trajectory_points:]
    if len(_landmark_trajectory) > max_trajectory_points:
        _landmark_trajectory = _landmark_trajectory[-max_trajectory_points:]

    # Use vis_config.save_dir directly. If it's relative, os.makedirs and os.path.join will handle it.
    # If it needs to be relative to an experiment path, the caller (e.g., evaluate_*) should modify
    # vis_config.save_dir to be an absolute path or a path relative to the experiment dir *before* calling this.
    save_dir = vis_config.save_dir 
    try:
        if not os.path.exists(save_dir): # save_dir is now potentially absolute or specific relative
            os.makedirs(save_dir)
    except OSError as e:
        print(f"Error creating visualization directory {save_dir}: {e}")
        return None

    fig, ax = plt.subplots(figsize=vis_config.figure_size)

    if show_trajectories and len(_agent_trajectory) > 1:
        agent_traj_x, agent_traj_y = zip(*_agent_trajectory)
        ax.plot(agent_traj_x, agent_traj_y, 'b-',
                linewidth=1.5, alpha=0.6, label='Agent Traj.')

    if show_trajectories and len(_landmark_trajectory) > 1:
        landmark_traj_x, landmark_traj_y = zip(*_landmark_trajectory)
        ax.plot(landmark_traj_x, landmark_traj_y, 'r-',
                linewidth=1.5, alpha=0.6, label='Landmark Traj.')

    if world.agent and world.agent.location:
        ax.scatter(world.agent.location.x, world.agent.location.y,
                   color='blue', marker='o', s=100, label=f'Agent (Z:{world.agent.location.depth:.1f})')

    if world.true_landmark and world.true_landmark.location:
        ax.scatter(world.true_landmark.location.x, world.true_landmark.location.y,
                   color='red', marker='^', s=100, label=f'True Lmk (Z:{world.true_landmark.location.depth:.1f})')
        if world.agent and world.agent.location:
            ax.plot([world.agent.location.x, world.true_landmark.location.x],
                    [world.agent.location.y, world.true_landmark.location.y],
                    'r--', alpha=0.5, label=f'True Range ({world.current_range:.1f})')

    if world.estimated_landmark and world.estimated_landmark.estimated_location is not None:
        est_loc = world.estimated_landmark.estimated_location
        ax.scatter(est_loc.x, est_loc.y,
                   color='green', marker='x', s=100, label=f'Est. Lmk (Z:{est_loc.depth:.1f})')

        if world.agent and world.agent.location:
            ax.plot([world.agent.location.x, est_loc.x],
                    [world.agent.location.y, est_loc.y],
                    'g--', alpha=0.5, label='Est. Range')

        if isinstance(world.estimated_landmark, TrackedTargetPF):        
            pf_core = world.estimated_landmark.pf_core
            if pf_core and pf_core.particles_state is not None and pf_core.num_particles < 500:
                particles = pf_core.particles_state
                ax.scatter(particles[:, 0], particles[:, 2], color='gray',
                        marker='.', s=1, alpha=0.3, label='Particles')

            if hasattr(pf_core, 'position_covariance_matrix') and pf_core.position_covariance_eigenvalues is not None:
                try:
                    from matplotlib.patches import Ellipse
                    eigvals = pf_core.position_covariance_eigenvalues
                    angle = np.degrees(pf_core.position_covariance_orientation)
                    safe_eigvals = np.maximum(eigvals, 1e-9)
                    width = safe_eigvals[0]**0.5 * 2 * 1.96
                    height = safe_eigvals[1]**0.5 * 2 * 1.96

                    ellipse = Ellipse(xy=(est_loc.x, est_loc.y), width=width, height=height, angle=angle,
                                    edgecolor='purple', fc='None', lw=1, ls='--', label='95% Conf.')
                    ax.add_patch(ellipse)
                except Exception as e:
                    print(f"Warning: Could not plot covariance ellipse - {e}")
    else:
        if world.agent and world.agent.location:
            ax.text(0.5, 0.02, "Estimator not initialized", ha='center',
                    transform=ax.transAxes, color='orange', fontsize=10)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    title_info = f"Reward: {world.reward:.4f}, Err: {world.error_dist:.2f}"
    if filename:
        try:
            step_part = filename.split('_')[-1].split('.')[0] # More robust split
            if "initial" in filename: # Check before isdigit for filenames like "frame_000_initial.png"
                title_info = "Step: 0, " + title_info
            elif step_part.isdigit():
                 title_info = f"Step: {int(step_part)}, " + title_info
        except (IndexError, ValueError): # Catch if split fails or not a digit
            pass


    ax.set_title(f'World State\n{title_info}')

    points_x = []
    points_y = []
    if world.agent and world.agent.location:
        points_x.append(world.agent.location.x)
        points_y.append(world.agent.location.y)
    if world.true_landmark and world.true_landmark.location:
        points_x.append(world.true_landmark.location.x)
        points_y.append(world.true_landmark.location.y)
    if world.estimated_landmark and world.estimated_landmark.estimated_location:
        points_x.append(world.estimated_landmark.estimated_location.x)
        points_y.append(world.estimated_landmark.estimated_location.y)

    if show_trajectories:
        if _agent_trajectory:
            traj_x, traj_y = zip(*_agent_trajectory)
            points_x.extend(traj_x)
            points_y.extend(traj_y)
        if _landmark_trajectory:
            traj_x, traj_y = zip(*_landmark_trajectory)
            points_x.extend(traj_x)
            points_y.extend(traj_y)

    if not points_x or not points_y:
        min_x, max_x, min_y, max_y = -10, 10, -10, 10
    else:
        min_x, max_x = min(points_x), max(points_x)
        min_y, max_y = min(points_y), max(points_y)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)
    max_range = max(range_x, range_y, 20.0) # Ensure a minimum viewport size
    padding = max_range * 0.2

    ax.set_xlim(center_x - (max_range / 2 + padding),
                center_x + (max_range / 2 + padding))
    ax.set_ylim(center_y - (max_range / 2 + padding),
                center_y + (max_range / 2 + padding))

    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1.0))

    if filename is None:
        timestamp = int(time.time())
        filename = f"world_state_{timestamp}.png"

    full_path = os.path.join(save_dir, filename) # save_dir is now from vis_config
    try:
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
        plt.savefig(full_path)
        plt.close(fig)
        return full_path
    except Exception as e:
        print(f"Error saving visualization to {full_path}: {e}")
        plt.close(fig)
        return None


def reset_trajectories():
    """Reset stored trajectory data."""
    global _agent_trajectory, _landmark_trajectory
    _agent_trajectory = []
    _landmark_trajectory = []


def save_gif(output_filename: str, vis_config: VisualizationConfig, frame_paths: list, delete_frames: bool = True):
    """
    Create a GIF from a list of frame image paths using visualization config.
    Assumes vis_config.save_dir is the correct directory for the output GIF.
    """
    if not frame_paths:
        print("No frame paths provided to create GIF.")
        return None

    save_dir = vis_config.save_dir # This should be the specific experiment's vis dir
    duration = vis_config.gif_frame_duration
    output_path = os.path.join(save_dir, output_filename)

    print(f"Creating GIF from {len(frame_paths)} frames: {output_path}")
    try:
        images = []
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                images.append(imageio.imread(frame_path))
            else:
                print(f"Warning: Frame file not found: {frame_path}")

        if not images:
            print("Error: No valid image frames found to create GIF.")
            return None

        imageio.mimsave(output_path, images, duration=duration)
        print(f"GIF saved successfully to {output_path}.")


        if delete_frames:
            deleted_count = 0
            for frame_path in frame_paths:
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                        deleted_count += 1
                except OSError as e:
                    print(
                        f"Warning: Could not delete frame file {frame_path}: {e}")
            if deleted_count > 0:
                print(f"Deleted {deleted_count} individual frame files.")

        return output_path

    except Exception as e:
        print(f"Error creating GIF {output_path}: {e}")
        return None


def create_gif_from_files(pattern: str, output_filename: str, vis_config: VisualizationConfig, delete_files: bool = False):
    """
    Create a GIF from existing image files matching a pattern in the visualization directory.
    Assumes vis_config.save_dir is the correct directory.
    """
    save_dir = vis_config.save_dir # This should be the specific experiment's vis dir
    search_pattern = os.path.join(save_dir, pattern)
    frame_paths = sorted(glob.glob(search_pattern))

    if not frame_paths:
        print(f"No files found matching pattern: {search_pattern}")
        return None

    return save_gif(
        output_filename=output_filename,
        vis_config=vis_config, # Pass the config which contains the correct save_dir
        frame_paths=frame_paths,
        delete_frames=delete_files
    )


if __name__ == "__main__":
    print("Running visualization example...")
    from configs import default_config
    
    # Create a temporary experiment-like structure for the example
    example_exp_base = "temp_example_experiments"
    example_exp_name = f"vis_example_{int(time.time())}"
    example_exp_path = os.path.join(example_exp_base, example_exp_name)
    example_vis_path = os.path.join(example_exp_path, "world_snapshots_example")
    os.makedirs(example_vis_path, exist_ok=True)
    
    print(f"Example visualizations will be saved in: {os.path.abspath(example_vis_path)}")

    # Get a copy of the default config to modify for this example
    current_config = default_config.model_copy(deep=True)
    current_config.visualization.save_dir = example_vis_path # Crucial: set the save_dir for this run
    
    world_cfg = current_config.world
    # Estimator config is resolved by DefaultConfig's post_init, so world_cfg.estimator_config is an instance
    
    world = World(world_config=world_cfg) # World uses its own config's estimator_config

    reset_trajectories()
    example_frames = []

    print("Visualizing initial state...")
    initial_file = visualize_world(
        world, vis_config=current_config.visualization, filename="example_frame_000.png")
    if initial_file:
        example_frames.append(initial_file)

    print("Simulating 10 steps...")
    for i in range(10):
        # Simple action: try to move towards (0,0) or a fixed point if landmark is far
        action_yaw_norm = np.random.uniform(-0.5, 0.5) # Random small yaw change

        world.step(action_yaw_norm, training=False)

        step_file = visualize_world(
            world, vis_config=current_config.visualization, filename=f"example_frame_{i+1:03d}.png")
        if step_file:
            example_frames.append(step_file)

    print("Creating example GIF...")
    gif_path = save_gif(
        output_filename="example_simulation.gif",
        vis_config=current_config.visualization, # Pass the config with the correct save_dir
        frame_paths=example_frames,
        delete_frames=current_config.visualization.delete_frames_after_gif
    )

    if gif_path:
        print(f"Visualization example complete. GIF saved to: {gif_path}")
    else:
        print("Visualization example failed to create GIF.")
    
    print(f"To clean up, remove the directory: {os.path.abspath(example_exp_base)}")