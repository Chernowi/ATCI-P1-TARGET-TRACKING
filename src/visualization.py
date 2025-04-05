import matplotlib.pyplot as plt
import os
import time
from world import World

def visualize_world(world, filename=None):
    """
    Visualize the world state in 2D (top-down view) and save it to a file.
    
    Args:
        world (World): The World object to visualize
        filename (str, optional): Optional filename for saving the plot. If None, a timestamp will be used.
    
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
              color='blue', marker='o', s=100, label='Agent')
    
    # Plot true landmark
    ax.scatter(world.true_landmark.location.x, world.true_landmark.location.y,
              color='red', marker='^', s=100, label='True Landmark')
    
    # Plot estimated landmark
    ax.scatter(world.estimated_landmark.location.x, world.estimated_landmark.location.y,
              color='green', marker='x', s=100, label='Estimated Landmark')
    
    # Add a line from agent to true landmark
    ax.plot([world.agent.location.x, world.true_landmark.location.x],
            [world.agent.location.y, world.true_landmark.location.y],
            'r--', alpha=0.5)
    
    # Add a line from agent to estimated landmark
    ax.plot([world.agent.location.x, world.estimated_landmark.location.x],
            [world.agent.location.y, world.estimated_landmark.location.y],
            'g--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'World State (Top-down View) - Reward: {world.reward:.4f}')
    
    # Add annotations for depth values
    ax.annotate(f"Agent Z: {world.agent.location.depth:.1f}", 
                (world.agent.location.x, world.agent.location.y), 
                xytext=(10, 10), textcoords='offset points')
    ax.annotate(f"True Z: {world.true_landmark.location.depth:.1f}", 
                (world.true_landmark.location.x, world.true_landmark.location.y), 
                xytext=(10, 10), textcoords='offset points')
    ax.annotate(f"Est Z: {world.estimated_landmark.location.depth:.1f}", 
                (world.estimated_landmark.location.x, world.estimated_landmark.location.y), 
                xytext=(10, 10), textcoords='offset points')
    
    # Center the view on the true landmark
    true_landmark_x = world.true_landmark.location.x
    true_landmark_y = world.true_landmark.location.y
    
    # Calculate distances from the true landmark to determine appropriate view range
    distances = [
        abs(world.agent.location.x - true_landmark_x),
        abs(world.agent.location.y - true_landmark_y),
        abs(world.estimated_landmark.location.x - true_landmark_x),
        abs(world.estimated_landmark.location.y - true_landmark_y)
    ]
    
    max_distance = max(distances)
    padding = max_distance * 0.3  # Add 30% padding for better visualization
    view_range = max_distance + padding
    
    # Set the limits with true landmark at center
    ax.set_xlim(true_landmark_x - view_range, true_landmark_x + view_range)
    ax.set_ylim(true_landmark_y - view_range, true_landmark_y + view_range)
    
    # Make the aspect ratio equal for more realistic visualization
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend()
    
    # Generate filename if not provided
    if filename is None:
        timestamp = int(time.time())
        filename = f"world_state_{timestamp}.png"
    
    # Save the plot
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    plt.close(fig)
    
    return full_path


if __name__ == "__main__":
    # Example usage
    from world import World
    from world_objects import Velocity
    
    world = World()
    # Visualize initial state
    visualize_world(world, "initial_state.png")
    
    # Run some steps and visualize again
    for i in range(10):
        world.step(Velocity(1, 1, 0))
    
    visualize_world(world, "after_10_steps.png")