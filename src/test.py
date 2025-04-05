import math
import numpy as np
from world import World
from world_objects import Velocity, Location
from visualization import visualize_world

def generate_landmark_approach_and_circle(world, target_radius=10.0, approach_speed=2.0, circle_speed=1.0):
    """
    Generate a series of actions that:
    1. Move the agent toward the true landmark 
    2. Circle around it at a specified radius
    
    Args:
        world: The World object
        target_radius: Radius at which to circle the landmark
        approach_speed: Speed for approaching the landmark
        circle_speed: Speed for circling the landmark
    
    Returns:
        List of Velocity objects
    """
    actions = []
    
    # Clone the initial world state
    initial_agent_location = Location(world.agent.location.x, world.agent.location.y, world.agent.location.depth)
    landmark_location = Location(world.true_landmark.location.x, world.true_landmark.location.y, world.true_landmark.location.depth)
    
    # First phase: Generate actions to approach the landmark
    # Calculate vector from agent to landmark
    dx = landmark_location.x - initial_agent_location.x
    dy = landmark_location.y - initial_agent_location.y
    dz = landmark_location.depth - initial_agent_location.depth
    
    # Calculate initial distance
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Calculate how far we need to move (distance - target_radius)
    travel_distance = max(0, distance - target_radius)
    
    # Calculate unit direction vector
    if distance > 0:
        direction_x = dx / distance
        direction_y = dy / distance
        direction_z = dz / distance
    else:
        direction_x, direction_y, direction_z = 0, 0, 0
    
    # Number of steps needed to approach (with some margin)
    approach_steps = int(travel_distance / approach_speed) + 1
    
    # Generate approach actions
    for _ in range(approach_steps):
        actions.append(Velocity(direction_x * approach_speed, 
                               direction_y * approach_speed, 
                               direction_z * approach_speed))
    
    # Second phase: Generate actions to circle the landmark
    # Number of steps for a full circle (each step is roughly circle_speed units of distance)
    circle_circumference = 2 * math.pi * target_radius
    num_circle_steps = int(circle_circumference / circle_speed) + 1
    
    # Generate points on the circle
    for i in range(num_circle_steps):
        angle = 2 * math.pi * i / num_circle_steps
        
        # Position on the circle relative to landmark
        circle_x = target_radius * math.cos(angle)
        circle_y = target_radius * math.sin(angle)
        
        # Absolute target position
        target_x = landmark_location.x + circle_x
        target_y = landmark_location.y + circle_y
        target_z = landmark_location.depth
        
        # If this isn't the first circle step, calculate velocity to get there
        if i > 0:
            prev_x = landmark_location.x + target_radius * math.cos(2 * math.pi * (i-1) / num_circle_steps)
            prev_y = landmark_location.y + target_radius * math.sin(2 * math.pi * (i-1) / num_circle_steps)
            
            # Calculate velocity vector
            velocity_x = (target_x - prev_x)
            velocity_y = (target_y - prev_y)
            velocity_z = 0  # Maintain same depth
            
            # Scale to desired speed
            current_speed = math.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
            if current_speed > 0:
                scale_factor = circle_speed / current_speed
                velocity_x *= scale_factor
                velocity_y *= scale_factor
                velocity_z *= scale_factor
            
            actions.append(Velocity(velocity_x, velocity_y, velocity_z))
    
    return actions

if __name__ == "__main__":
    # Create world
    world = World()
    
    # Generate approach and circle actions
    actions = generate_landmark_approach_and_circle(world, target_radius=10.0)
    
    # Visualize initial state
    visualize_world(world, "00_initial_state.png")
    
    # Apply actions and visualize at key points
    for i, action in enumerate(actions):
        world.step(action)
        
        # Visualize every few steps
        if i % 5 == 0 or i == len(actions) - 1:
            visualize_world(world, f"{i+1:02d}_state.png")
            
            # Calculate distance to landmark for debugging
            dx = world.agent.location.x - world.true_landmark.location.x
            dy = world.agent.location.y - world.true_landmark.location.y
            dz = world.agent.location.depth - world.true_landmark.location.depth
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            print(f"Step {i+1}: Distance to landmark = {distance:.2f}")
    
    print(f"Generated {len(actions)} actions to approach and circle the landmark")