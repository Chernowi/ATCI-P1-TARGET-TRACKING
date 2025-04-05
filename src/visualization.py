import matplotlib.pyplot as plt
import os
import time
import numpy as np # Make sure numpy is imported
from world import World # Assuming world.py is in the same directory
from world_objects import Location, Velocity # Assuming world_objects.py is in the same directory

def visualize_world(world, filename=None):
    """
    Visualize the world state in 2D (top-down view) and save it to a file.

    Args:
        world (World): The World object to visualize
        filename (str, optional): Optional filename for saving the plot (without directory).
                                  If None, a timestamp will be used.

    Returns:
        str: Full path to the saved image file
    """
    # Create directory if it doesn't exist
    save_dir = "world_snapshots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create figure and 2D axes (top-down view)
    fig, ax = plt.subplots(figsize=(10, 8))

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

        # --- CORRECTED PARTICLE VISUALIZATION ---
        # Optionally plot particles if available and not too many
        # Access current_particles_state from the TrackedTargetPF instance (world.estimated_landmark)
        # Access num_particles from the core filter instance (world.estimated_landmark.pf_core)
        if world.estimated_landmark.current_particles_state is not None and world.estimated_landmark.pf_core.num_particles < 500:
             particles = world.estimated_landmark.current_particles_state # Corrected access path
             ax.scatter(particles[:, 0], particles[:, 2], color='gray', marker='.', s=1, alpha=0.3, label='Particles')
        # --- END CORRECTION ---

        # Plot covariance ellipse if available
        if hasattr(world.estimated_landmark.pf_core, 'position_covariance_matrix'):
            from matplotlib.patches import Ellipse
            cov = world.estimated_landmark.pf_core.position_covariance_matrix
            eigvals = world.estimated_landmark.pf_core.position_covariance_eigenvalues
            angle = np.degrees(world.estimated_landmark.pf_core.position_covariance_orientation)
            # Scale eigenvalues for confidence interval (e.g., 2 std dev for ~95%)
            # Ensure eigenvalues are not negative before scaling
            safe_eigvals = np.maximum(eigvals, 1e-9) # Prevent issues with zero/negative values
            width = safe_eigvals[0]**0.5 * 2 * 1.96 # Major axis length (sqrt first)
            height = safe_eigvals[1]**0.5 * 2 * 1.96 # Minor axis length (sqrt first)

            ellipse = Ellipse(xy=(est_loc.x, est_loc.y), width=width, height=height, angle=angle,
                              edgecolor='purple', fc='None', lw=1, ls='--', label='95% Conf. Ellipse')
            ax.add_patch(ellipse)

    else:
        # Add placeholder text for when estimate isn't available
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
        # Try to extract step number from filename (e.g., "frame_001.png")
        title_step = filename.split('_')[1].split('.')[0]
    except (IndexError, AttributeError):
        pass # Keep "N/A" if format is different or filename is None

    ax.set_title(f'World State - Step: {title_step}\nReward: {world.reward:.4f}') # Add reward to title

    # --- Dynamic Axis Limits ---
    # Collect all relevant points
    points_x = [world.agent.location.x, world.true_landmark.location.x]
    points_y = [world.agent.location.y, world.true_landmark.location.y]
    if world.estimated_landmark.estimated_location is not None:
        points_x.append(world.estimated_landmark.estimated_location.x)
        points_y.append(world.estimated_landmark.estimated_location.y)

    min_x, max_x = min(points_x), max(points_x)
    min_y, max_y = min(points_y), max(points_y)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    range_x = (max_x - min_x)
    range_y = (max_y - min_y)

    # Determine the larger range and add padding
    max_range = max(range_x, range_y, 20.0) # Ensure a minimum view size
    padding = max_range * 0.2 # Add 20% padding

    ax.set_xlim(center_x - (max_range / 2 + padding), center_x + (max_range / 2 + padding))
    ax.set_ylim(center_y - (max_range / 2 + padding), center_y + (max_range / 2 + padding))
    # --- End Dynamic Axis Limits ---

    # Make the aspect ratio equal for more realistic visualization
    ax.set_aspect('equal', adjustable='box')

    # Add legend
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1)) # Place legend outside plot area

    # Generate filename if not provided
    if filename is None:
        timestamp = int(time.time())
        filename = f"world_state_{timestamp}.png"

    # Save the plot
    full_path = os.path.join(save_dir, filename)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(full_path)
    plt.close(fig) # Close the figure to free memory

    return full_path


if __name__ == "__main__":
    # Example usage
    # from world import World # Already imported
    # from world_objects import Velocity # Already imported

    print("Running visualization example...")
    world = World()
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
        # print(f"Saved step {i+1} to: {step_file}") # Less verbose output

    print("Visualization example complete.")
    print(f"Check the '{os.path.abspath('world_snapshots')}' directory for output images.")