# --- START OF FILE run_manual_policy.py ---

import argparse
import os
import math
import numpy as np
import time

from world import World
from world_objects import Velocity, Location
from configs import CONFIGS, DefaultConfig
from visualization import visualize_world, reset_trajectories, save_gif

def calculate_manual_action(agent_loc: Location, landmark_loc: Location, target_radius: float, speed: float) -> Velocity:
    """
    Calculates the next velocity action based on agent and landmark locations.

    Args:
        agent_loc: Current location of the agent.
        landmark_loc: Current location of the true landmark.
        target_radius: The desired circling radius.
        speed: The desired speed of the agent.

    Returns:
        Velocity object representing the calculated action.
    """
    # Vector from agent to landmark
    vector_to_landmark = np.array([landmark_loc.x - agent_loc.x,
                                   landmark_loc.y - agent_loc.y])

    distance = np.linalg.norm(vector_to_landmark)

    # Avoid division by zero if agent is exactly on the landmark
    if distance < 1e-6:
        # Move in an arbitrary direction if too close
        return Velocity(x=speed, y=0.0, z=0.0)

    # Normalize the direction vector
    direction_to_landmark = vector_to_landmark / distance

    # --- Control Logic ---
    approach_buffer = 2.0 # Start circling a bit before reaching the exact radius
    correction_factor = 0.2 # How strongly to correct towards the target radius

    if distance > target_radius + approach_buffer:
        # Phase 1: Approach the landmark directly
        desired_velocity = direction_to_landmark * speed
    else:
        # Phase 2: Circle the landmark
        # Calculate tangential direction (counter-clockwise)
        tangential_direction = np.array([-direction_to_landmark[1], direction_to_landmark[0]])

        # Calculate radial correction velocity (inward if too far, outward if too close)
        radius_error = distance - target_radius
        # Pointing towards the center (opposite of direction_to_landmark)
        radial_direction_inward = -direction_to_landmark
        correction_velocity = radial_direction_inward * radius_error * correction_factor

        # Combine tangential movement with radial correction
        combined_velocity = tangential_direction * speed + correction_velocity

        # Re-normalize the combined velocity to maintain the target speed
        combined_norm = np.linalg.norm(combined_velocity)
        if combined_norm > 1e-6:
            desired_velocity = (combined_velocity / combined_norm) * speed
        else:
            # If correction perfectly cancels tangential, just move tangentially
            desired_velocity = tangential_direction * speed

    return Velocity(x=desired_velocity[0], y=desired_velocity[1], z=0.0)


def run_manual_policy(config_name: str, num_steps: int, target_radius: float, speed: float, render: bool):
    """
    Run a manual policy (approach and circle) in the environment.

    Args:
        config_name: Name of the configuration profile.
        num_steps: Number of simulation steps to run.
        target_radius: Desired circling radius around the landmark.
        speed: Speed of the agent.
        render: Whether to render the environment and save a GIF.
    """
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}' for manual policy run")

    # Ensure visualization directory exists if rendering
    if render:
        os.makedirs(config.visualization.save_dir, exist_ok=True)
        reset_trajectories()

    # Instantiate the world
    # World uses its internal estimator config, but we control the agent manually
    world = World(world_config=config.world)

    total_reward = 0.0
    episode_frames = []

    print(f"Running manual policy for {num_steps} steps...")
    print(f"Target Radius: {target_radius}, Agent Speed: {speed}")
    print(f"Initial Agent Pos: {world.agent.location}")
    print(f"Initial Landmark Pos: {world.true_landmark.location}")


    # Visualize initial state
    if render:
        try:
            initial_frame_file = visualize_world(
                world=world,
                vis_config=config.visualization,
                filename=f"manual_policy_frame_000_initial.png",
                collect_for_gif=True
            )
            if initial_frame_file:
                episode_frames.append(initial_frame_file)
        except Exception as e:
            print(f"Warning: Visualization failed for initial state. Error: {e}")

    start_time = time.time()
    for step in range(num_steps):
        agent_loc = world.agent.location
        landmark_loc = world.true_landmark.location # Use true location for control

        # Calculate the action based on the manual policy
        action = calculate_manual_action(agent_loc, landmark_loc, target_radius, speed)

        # Step the environment - use training=False as we don't need agent rewards/termination
        # but the world will still calculate its internal reward based on its logic
        world.step(action, training=False)

        # Accumulate reward (the reward calculated by the world environment based on error/distance)
        total_reward += world.reward

        # Print status periodically
        if (step + 1) % 20 == 0 or step == num_steps - 1:
            print(f"Step: {step+1}/{num_steps}, "
                  f"Reward: {world.reward:.3f}, "
                  f"Total Reward: {total_reward:.3f}, "
                  f"Agent-Lmk Dist: {world._calculate_range_measurement(agent_loc, landmark_loc):.2f}, " # Use actual range for reporting
                  f"Est Error: {world.error_dist:.2f}, "
                  f"Agent Pos: ({agent_loc.x:.1f}, {agent_loc.y:.1f})")

        # Render frame
        if render:
            try:
                frame_file = visualize_world(
                    world=world,
                    vis_config=config.visualization,
                    filename=f"manual_policy_frame_{step+1:03d}.png",
                    collect_for_gif=True
                )
                if frame_file:
                    episode_frames.append(frame_file)
            except Exception as e:
                print(f"Warning: Visualization failed for step {step+1}. Error: {e}")

        # Optional: Add a small sleep to make visualization progress visible
        # if render: time.sleep(0.05)

    end_time = time.time()
    duration = end_time - start_time
    steps_per_sec = num_steps / duration if duration > 0 else float('inf')

    print("\n--- Manual Policy Run Summary ---")
    print(f"Total Steps: {num_steps}")
    print(f"Final Total Reward: {total_reward:.2f}")
    print(f"Final Agent Position: {world.agent.location}")
    print(f"Final True Landmark Position: {world.true_landmark.location}")
    print(f"Final Estimated Landmark Position: {world.estimated_landmark.estimated_location}")
    print(f"Final Error Distance (True vs. Est): {world.error_dist:.2f}")
    print(f"Simulation Duration: {duration:.2f} seconds ({steps_per_sec:.1f} steps/sec)")

    # Create GIF
    if render and episode_frames:
        gif_filename = f"manual_policy_{config_name}_r{target_radius}_s{speed}.gif"
        print(f"\nCreating GIF: {gif_filename}...")
        try:
            save_gif(
                output_filename=gif_filename,
                vis_config=config.visualization,
                frame_paths=episode_frames,
                delete_frames=config.visualization.delete_frames_after_gif
            )
            print(f"GIF saved to {os.path.join(config.visualization.save_dir, gif_filename)}")
        except Exception as e:
            print(f"Error creating GIF: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a manual approach-and-circle policy in the landmark tracking environment.")
    parser.add_argument(
        "--config", "-c", type=str, default="default",
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=200,
        help="Number of simulation steps to run."
    )
    parser.add_argument(
        "--radius", "-r", type=float, default=10.0,
        help="Target circling radius around the landmark."
    )
    parser.add_argument(
        "--speed", "-sp", type=float, default=1.5,
        help="Target speed of the agent."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable visualization and save a GIF of the run."
    )

    args = parser.parse_args()

    run_manual_policy(
        config_name=args.config,
        num_steps=args.steps,
        target_radius=args.radius,
        speed=args.speed,
        render=args.render
    )
# --- END OF FILE run_manual_policy.py ---