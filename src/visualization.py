import matplotlib.pyplot as plt
import os
import time
import numpy as np
from world import World
from world_objects import Location, Velocity
import imageio.v2 as imageio  # For creating GIFs
from PIL import Image  # For image processing
import glob

# Trajectory history storage
_agent_trajectory = []
_landmark_trajectory = []
# Frame storage for GIF creation
_frames = []

def visualize_world(world, filename=None, show_trajectories=True, max_trajectory_points=100, collect_for_gif=True):
    """
    Visualize the world state in 2D (top-down view) and save it to a file.

    Args:
        world (World): The World object to visualize
        filename (str, optional): Optional filename for saving the plot (without directory).
                                  If None, a timestamp will be used.
        show_trajectories (bool): Whether to show the agent and landmark trajectories
        max_trajectory_points (int): Maximum number of trajectory points to show
        collect_for_gif (bool): Whether to collect this frame for GIF creation

    Returns:
        str: Full path to the saved image file
    """
    global _agent_trajectory, _landmark_trajectory, _frames
    
    # Update trajectories
    _agent_trajectory.append((world.agent.location.x, world.agent.location.y))
    _landmark_trajectory.append((world.true_landmark.location.x, world.true_landmark.location.y))
    
    # Limit trajectory history to prevent memory issues
    _agent_trajectory = _agent_trajectory[-max_trajectory_points:]
    _landmark_trajectory = _landmark_trajectory[-max_trajectory_points:]
    
    # Create directory if it doesn't exist
    save_dir = "world_snapshots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create figure and 2D axes (top-down view)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot trajectories if enabled
    if show_trajectories and len(_agent_trajectory) > 1:
        agent_traj_x, agent_traj_y = zip(*_agent_trajectory)
        ax.plot(agent_traj_x, agent_traj_y, 'b-', linewidth=1.5, alpha=0.6, label='Agent Trajectory')

    if show_trajectories and len(_landmark_trajectory) > 1:
        landmark_traj_x, landmark_traj_y = zip(*_landmark_trajectory)
        ax.plot(landmark_traj_x, landmark_traj_y, 'r-', linewidth=1.5, alpha=0.6, label='Landmark Trajectory')

    # Plot agent
    ax.scatter(world.agent.location.x, world.agent.location.y,
              color='blue', marker='o', s=100, label=f'Agent (Z:{world.agent.location.depth:.1f})')

    # Plot true landmark
    ax.scatter(world.true_landmark.location.x, world.true_landmark.location.y,
              color='red', marker='^', s=100, label=f'True Landmark (Z:{world.true_landmark.location.depth:.1f})')

    # Check if estimated location exists before plotting
    if world.estimated_landmark.estimated_location is not None:
        est_loc = world.estimated_landmark.estimated_location
        # Plot estimated landmark
        ax.scatter(est_loc.x, est_loc.y,
                  color='green', marker='x', s=100, label=f'Est. Landmark (Z:{est_loc.depth:.1f})')

        # Add a line from agent to estimated landmark
        ax.plot([world.agent.location.x, est_loc.x],
                [world.agent.location.y, est_loc.y],
                'g--', alpha=0.5)

        # Optionally plot particles if available and not too many
        if world.estimated_landmark.current_particles_state is not None and world.estimated_landmark.pf_core.num_particles < 500:
             particles = world.estimated_landmark.current_particles_state
             ax.scatter(particles[:, 0], particles[:, 2], color='gray', marker='.', s=1, alpha=0.3, label='Particles')

        # Plot covariance ellipse if available
        if hasattr(world.estimated_landmark.pf_core, 'position_covariance_matrix'):
            from matplotlib.patches import Ellipse
            cov = world.estimated_landmark.pf_core.position_covariance_matrix
            eigvals = world.estimated_landmark.pf_core.position_covariance_eigenvalues
            angle = np.degrees(world.estimated_landmark.pf_core.position_covariance_orientation)
            # Scale eigenvalues for confidence interval
            safe_eigvals = np.maximum(eigvals, 1e-9) # Prevent issues with zero/negative values
            width = safe_eigvals[0]**0.5 * 2 * 1.96 # Major axis length (sqrt first)
            height = safe_eigvals[1]**0.5 * 2 * 1.96 # Minor axis length (sqrt first)

            ellipse = Ellipse(xy=(est_loc.x, est_loc.y), width=width, height=height, angle=angle,
                              edgecolor='purple', fc='None', lw=1, ls='--', label='95% Conf. Ellipse')
            ax.add_patch(ellipse)

    else:
        # Add placeholder text when estimate isn't available
        ax.text(0.5, 0.02, "Particle filter not yet initialized",
                ha='center', transform=ax.transAxes, color='orange', fontsize=10)

    # Add a line from agent to true landmark (representing true range)
    ax.plot([world.agent.location.x, world.true_landmark.location.x],
            [world.agent.location.y, world.true_landmark.location.y],
            'r--', alpha=0.5, label=f'True Range ({world.current_range:.1f})')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    title_step = "N/A"
    try:
        # Try to extract step number from filename
        title_step = filename.split('_')[1].split('.')[0]
    except (IndexError, AttributeError):
        pass # Keep "N/A" if format is different or filename is None

    ax.set_title(f'World State - Step: {title_step}\nReward: {world.reward:.4f}')

    # Dynamic Axis Limits
    points_x = [world.agent.location.x, world.true_landmark.location.x]
    points_y = [world.agent.location.y, world.true_landmark.location.y]
    
    if show_trajectories:
        for x, y in _agent_trajectory:
            points_x.append(x)
            points_y.append(y)
        for x, y in _landmark_trajectory:
            points_x.append(x)
            points_y.append(y)
            
    if world.estimated_landmark.estimated_location is not None:
        points_x.append(world.estimated_landmark.estimated_location.x)
        points_y.append(world.estimated_landmark.estimated_location.y)

    min_x, max_x = min(points_x), max(points_x)
    min_y, max_y = min(points_y), max(points_y)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    range_x = max(max_x - min_x, 1.0)  # Avoid zero range
    range_y = max(max_y - min_y, 1.0)  # Avoid zero range

    max_range = max(range_x, range_y, 20.0) # Ensure a minimum view size
    padding = max_range * 0.2 # Add 20% padding

    ax.set_xlim(center_x - (max_range / 2 + padding), center_x + (max_range / 2 + padding))
    ax.set_ylim(center_y - (max_range / 2 + padding), center_y + (max_range / 2 + padding))

    # Make the aspect ratio equal for more realistic visualization
    ax.set_aspect('equal', adjustable='box')

    # Add legend
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))

    # Generate filename if not provided
    if filename is None:
        timestamp = int(time.time())
        filename = f"world_state_{timestamp}.png"

    # Save the plot
    full_path = os.path.join(save_dir, filename)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(full_path)
    
    # Collect frame for GIF if needed
    if collect_for_gif:
        _frames.append(full_path)
    
    plt.close(fig) # Close the figure to free memory

    return full_path


def reset_trajectories():
    """Reset all stored trajectory data."""
    global _agent_trajectory, _landmark_trajectory, _frames
    _agent_trajectory = []
    _landmark_trajectory = []
    _frames = []


def save_gif(output_filename="simulation.gif", duration=0.2, delete_frames=False):
    """
    Create a GIF from the stored frames.
    
    Args:
        output_filename (str): Name of the output GIF file
        duration (float): Duration of each frame in seconds
        delete_frames (bool): Whether to delete individual frames after creating the GIF
        
    Returns:
        str: Path to the created GIF
    """
    global _frames
    
    if not _frames:
        print("No frames available to create GIF")
        return None
    
    save_dir = "world_snapshots"
    output_path = os.path.join(save_dir, output_filename)
    
    # Create GIF from frames
    images = []
    for frame_path in _frames:
        images.append(imageio.imread(frame_path))
    
    # Save the GIF
    imageio.mimsave(output_path, images, duration=duration)
    
    print(f"GIF saved to: {output_path}")
    
    # Optionally delete individual frame files
    if delete_frames:
        for frame_path in _frames:
            try:
                os.remove(frame_path)
            except OSError:
                pass
        print("Individual frame files deleted")
    
    # Reset frames list
    _frames = []
    
    return output_path


# Add functions to create GIF from existing frames
def create_gif_from_files(pattern="eval_frame_*.png", output_filename="simulation.gif", duration=0.2, delete_frames=False):
    """
    Create a GIF from existing image files matching a pattern.
    
    Args:
        pattern (str): Pattern to match files (e.g., "eval_frame_*.png")
        output_filename (str): Name of the output GIF file
        duration (float): Duration of each frame in seconds
        delete_frames (bool): Whether to delete individual frames after creating the GIF
        
    Returns:
        str: Path to the created GIF
    """
    save_dir = "world_snapshots"
    frame_paths = sorted(glob.glob(os.path.join(save_dir, pattern)))
    
    if not frame_paths:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    output_path = os.path.join(save_dir, output_filename)
    
    # Create GIF from frames
    images = []
    for frame_path in frame_paths:
        images.append(imageio.imread(frame_path))
    
    # Save the GIF
    imageio.mimsave(output_path, images, duration=duration)
    
    print(f"GIF saved to: {output_path}")
    
    # Optionally delete individual frame files
    if delete_frames:
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except OSError:
                pass
        print("Individual frame files deleted")
    
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Running visualization example...")
    world = World()
    
    # Reset trajectories at the beginning
    reset_trajectories()
    
    # Visualize initial state
    initial_file = visualize_world(world, filename="000_example_initial.png")
    print(f"Saved initial state to: {initial_file}")

    # Run some steps and visualize again
    print("Simulating 10 steps...")
    for i in range(10):
        # Simulate a simple movement (e.g., moving towards landmark initially)
        dx = world.true_landmark.location.x - world.agent.location.x
        dy = world.true_landmark.location.y - world.agent.location.y
        dist = max(1e-6, (dx**2 + dy**2)**0.5)
        step_speed = 1.0
        action = Velocity(dx/dist * step_speed, dy/dist * step_speed, 0)
        world.step(action)
        # Visualize every step for example
        step_file = visualize_world(world, filename=f"{i+1:03d}_example_step.png")
    
    # Create a GIF from the frames
    gif_path = save_gif("example_simulation.gif", duration=0.3)
    print(f"Visualization example complete. GIF saved to: {gif_path}")