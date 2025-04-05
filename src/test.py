import math
import numpy as np
import os
# Import necessary classes from other files
from world import World
from world_objects import Velocity, Location
# Import the visualization function we just defined/provided
from visualization import visualize_world


def generate_landmark_approach_and_circle(world, target_radius=10.0, approach_speed=2.0, circle_speed=1.0, dt=1.0):
    """
    Generate a series of actions that:
    1. Move the agent toward the true landmark
    2. Circle around it at a specified radius

    Args:
        world: The World object
        target_radius: Radius at which to circle the landmark
        approach_speed: Speed for approaching the landmark (per step)
        circle_speed: Speed for circling the landmark (per step)
        dt: Timestep duration (used to determine movement per step)

    Returns:
        List of Velocity objects representing the desired velocity for each step.
    """
    actions = []

    # Use copies of locations to avoid modifying world state during planning
    agent_location = Location(world.agent.location.x, world.agent.location.y, world.agent.location.depth)
    landmark_location = Location(world.true_landmark.location.x, world.true_landmark.location.y, world.true_landmark.location.depth)

    # --- Phase 1: Approach ---
    current_agent_pos = agent_location
    step_count = 0
    max_approach_steps = 200 # Safety break

    print(f"Starting approach. Initial distance: {math.sqrt((landmark_location.x - current_agent_pos.x)**2 + (landmark_location.y - current_agent_pos.y)**2 + (landmark_location.depth - current_agent_pos.depth)**2):.2f}")

    while step_count < max_approach_steps:
        dx = landmark_location.x - current_agent_pos.x
        dy = landmark_location.y - current_agent_pos.y
        dz = landmark_location.depth - current_agent_pos.depth
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Stop if we are close enough or have overshot slightly
        # Use a small tolerance when comparing floating point distances
        if distance <= target_radius + 0.1: # Stop slightly *outside* or on the radius
            print(f"Approach complete after {step_count} steps. Distance: {distance:.2f}")
            break

        # Calculate unit direction vector
        if distance > 1e-6: # Avoid division by zero
            direction_x = dx / distance
            direction_y = dy / distance
            direction_z = dz / distance
        else: # Already at the landmark (shouldn't happen with check above)
            print("Agent is already at the landmark center.")
            break

        # Calculate velocity for this step
        # Move by approach_speed * dt in the target direction
        # Note: If dt != 1, speed is units/second. If dt=1, speed is units/step.
        vel_x = direction_x * approach_speed
        vel_y = direction_y * approach_speed
        vel_z = direction_z * approach_speed
        action = Velocity(vel_x, vel_y, vel_z)
        actions.append(action)

        # Simulate agent's position update for next iteration's check
        # We need to consider dt here for the simulation step
        current_agent_pos.x += vel_x * dt
        current_agent_pos.y += vel_y * dt
        current_agent_pos.depth += vel_z * dt # Renamed from depth to z for consistency
        step_count += 1

    if step_count == max_approach_steps:
        print("Warning: Max approach steps reached.")

    # --- Phase 2: Circle ---
    # Calculate total angle to cover for one full circle
    total_angle = 2 * math.pi
    # Calculate the angular speed needed to maintain linear speed 'circle_speed' at 'target_radius'
    # arc_length = speed * dt => radius * delta_angle = speed * dt => delta_angle = (speed * dt) / radius
    if target_radius < 1e-6:
        print("Warning: Target radius too small for circling.")
        return actions # Cannot circle with zero radius

    # Angular change per step
    angular_step = (circle_speed * dt) / target_radius if target_radius > 0 else 0

    # Calculate number of steps for a full circle
    # Ensure angular_step is positive before division
    num_circle_steps = max(20, int(total_angle / angular_step)) if angular_step > 1e-9 else 20 # Ensure reasonable number of steps, avoid division by zero
    print(f"Generating {num_circle_steps} steps for circling.")

    # Get the agent's position *relative* to the landmark after approach phase
    # Use the simulated 'current_agent_pos'
    relative_x = current_agent_pos.x - landmark_location.x
    relative_y = current_agent_pos.y - landmark_location.y
    # Calculate the starting angle in the XY plane
    start_angle = math.atan2(relative_y, relative_x)

    for i in range(num_circle_steps):
        # Calculate the target angle for this step (angle *after* this movement)
        # Add half step angle to calculate tangent at midpoint for better stability? No, original is fine.
        current_angle = start_angle + (i + 1) * angular_step

        # Calculate the direction tangent to the circle at this angle
        # Tangent vector is perpendicular to the radius vector (-sin(angle), cos(angle)) for CCW
        tangent_x = -math.sin(current_angle)
        tangent_y = math.cos(current_angle)
        tangent_z = 0 # Assume circling in the XY plane (constant depth)

        # Create the velocity vector with the desired circle_speed
        # Note: If dt != 1, speed is units/second. If dt=1, speed is units/step.
        vel_x = tangent_x * circle_speed
        vel_y = tangent_y * circle_speed
        vel_z = tangent_z * circle_speed # This will be 0

        action = Velocity(vel_x, vel_y, vel_z)
        actions.append(action)

        # --- Optional: Simulate agent pos update for next iteration if needed ---
        # current_agent_pos.x += vel_x * dt
        # current_agent_pos.y += vel_y * dt
        # current_agent_pos.z += vel_z * dt
        # start_angle = math.atan2(current_agent_pos.y - landmark_location.y, current_agent_pos.x - landmark_location.x) # Update start angle dynamically? Might cause drift. Sticking to original plan.


    print(f"Generated {len(actions)} total actions.")
    return actions

if __name__ == "__main__":
    # Create world
    world = World(dt=1.0) # Use dt=1.0 for simplicity, matches implicit assumption in generate_...

    # Define parameters
    approach_circle_radius = 15.0 # Increased radius slightly
    approach_move_speed = 1.5 # Speed units per second (or per step if dt=1)
    circling_move_speed = 1.0 # Speed units per second (or per step if dt=1)

    # Generate approach and circle actions
    actions = generate_landmark_approach_and_circle(
        world,
        target_radius=approach_circle_radius,
        approach_speed=approach_move_speed,
        circle_speed=circling_move_speed,
        dt=world.dt
    )

    # --- Visualization Setup ---
    # The visualize_world function handles directory creation ('world_snapshots')
    # We just need to provide the base filename for each frame.

    # Clear previous frames in the target directory (optional)
    snapshot_dir = "world_snapshots"
    if os.path.exists(snapshot_dir):
        for filename in os.listdir(snapshot_dir):
            if filename.startswith("frame_") and filename.endswith(".png"):
                 try:
                     os.remove(os.path.join(snapshot_dir, filename))
                 except OSError as e:
                     print(f"Error removing file {filename}: {e}")
    # --- End Visualization Setup ---

    # Visualize initial state
    initial_filename = "frame_000_initial.png"
    visualize_world(world, filename=initial_filename)
    print(f"Step 000: Initial State. Agent: {world.agent.location.x:.1f},{world.agent.location.y:.1f} Landmark: {world.true_landmark.location.x:.1f},{world.true_landmark.location.y:.1f}")


    # Apply actions and visualize each step
    for i, action in enumerate(actions):
        world.step(action) # World dt is already set within the world object

        # Visualize current state
        frame_num = i + 1
        current_filename = f"frame_{frame_num:03d}.png"
        visualize_world(world, filename=current_filename)

        # Calculate distance to landmark for debugging
        dx = world.agent.location.x - world.true_landmark.location.x
        dy = world.agent.location.y - world.true_landmark.location.y
        # Use 2D distance for checking radius in XY plane
        distance_2d = math.sqrt(dx*dx + dy*dy)
        # Also calculate 3D distance
        dz = world.agent.location.depth - world.true_landmark.location.depth
        distance_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

        phase = "Approach" if distance_3d > approach_circle_radius + 0.5 else "Circle" # Approximate phase detection
        est_info = "N/A"
        if world.estimated_landmark.estimated_location:
            est_loc = world.estimated_landmark.estimated_location
            est_err = ((est_loc.x - world.true_landmark.location.x)**2 + (est_loc.y - world.true_landmark.location.y)**2)**0.5
            est_info = f"Est=(x:{est_loc.x:.1f}, y:{est_loc.y:.1f}, z:{est_loc.depth:.1f}), Err={est_err:.2f}"

        print(f"Step {frame_num:03d}: Phase={phase}. Action=(vx:{action.x:.2f}, vy:{action.y:.2f}). Agent=(x:{world.agent.location.x:.1f}, y:{world.agent.location.y:.1f}). TrueDist2D={distance_2d:.2f}. {est_info}")


    print(f"\nFinished simulation. Generated {len(actions)} actions.")
    print(f"Find visualization frames in the '{os.path.abspath(snapshot_dir)}' directory.")
    print("You can use tools like ffmpeg or online converters to create a video/gif from these frames.")
    print("Example ffmpeg command (run in terminal in the project directory):")
    print(f"ffmpeg -r 10 -i {snapshot_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p trajectory.mp4")