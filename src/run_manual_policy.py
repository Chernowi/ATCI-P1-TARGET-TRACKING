import argparse
import os
import math
import numpy as np
import time

from world import World
# from world_objects import Velocity # No longer needed here
from world_objects import Location # Still needed
from configs import CONFIGS, DefaultConfig

# Conditional import for visualization
vis_available = False
try:
    from visualization import visualize_world, reset_trajectories, save_gif
    import imageio.v2 as imageio # Needed by save_gif
    vis_available = True
except ImportError:
    print("Visualization libraries not found, rendering will be disabled for manual policy.")


def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle

def calculate_manual_action(agent_loc: Location, agent_vel_x: float, agent_vel_y: float,
                            landmark_loc: Location, target_radius: float, max_yaw_change: float) -> float:
    """
    Calculates the next normalized yaw angle change [-1, 1] based on agent state and landmark location.

    Args:
        agent_loc: Current location of the agent.
        agent_vel_x: Current X velocity of the agent.
        agent_vel_y: Current Y velocity of the agent.
        landmark_loc: Current location of the true landmark.
        target_radius: The desired circling radius.
        max_yaw_change: The maximum yaw change allowed per step (from config).

    Returns:
        float: Normalized yaw change action in [-1, 1].
    """
    # Vector from agent to landmark
    vector_to_landmark = np.array([landmark_loc.x - agent_loc.x,
                                   landmark_loc.y - agent_loc.y])
    distance = np.linalg.norm(vector_to_landmark)
    if distance < 1e-6: distance = 1e-6 # Avoid division by zero

    # Angle from agent to landmark
    angle_to_landmark = math.atan2(vector_to_landmark[1], vector_to_landmark[0])

    # Current agent heading
    current_heading = math.atan2(agent_vel_y, agent_vel_x)

    # --- Control Logic ---
    approach_buffer = 2.0 # Start circling a bit before reaching the exact radius
    correction_strength = 0.5 # How strongly to correct towards the target radius/tangent

    desired_heading = current_heading # Initialize with current heading

    if distance > target_radius + approach_buffer:
        # Phase 1: Approach the landmark directly
        desired_heading = angle_to_landmark
    else:
        # Phase 2: Circle the landmark
        # Calculate target tangential heading (counter-clockwise)
        # Tangent angle is angle_to_landmark + pi/2
        tangential_heading = angle_to_landmark + math.pi / 2
        tangential_heading = normalize_angle(tangential_heading)

        # Calculate radial correction towards target radius
        radius_error = distance - target_radius
        # If too far, steer slightly inwards (towards landmark); if too close, steer slightly outwards
        # Correction angle towards center is angle_to_landmark
        # Correction angle away from center is angle_to_landmark + pi
        correction_angle = normalize_angle(angle_to_landmark + math.pi) if radius_error < 0 else angle_to_landmark
        radial_correction_strength = abs(radius_error) * correction_strength

        # Blend tangential heading and radial correction heading
        # Weighted average of angles requires careful handling (use vectors or complex numbers)
        # Simpler approach: calculate target tangential velocity and target radial velocity, sum, get angle.
        tangent_vec = np.array([math.cos(tangential_heading), math.sin(tangential_heading)])
        # Vector pointing towards center = -vector_to_landmark / distance
        radial_vec_inward = -vector_to_landmark / distance
        # Adjust radial vector direction based on error sign
        radial_vec = radial_vec_inward * radius_error * correction_strength # Inward if error > 0, outward if error < 0

        combined_vec = tangent_vec + radial_vec
        desired_heading = math.atan2(combined_vec[1], combined_vec[0])


    # Calculate required yaw change
    yaw_change_required = normalize_angle(desired_heading - current_heading)

    # Clamp the yaw change to the maximum allowed
    clamped_yaw_change = np.clip(yaw_change_required, -max_yaw_change, max_yaw_change)

    # Normalize the clamped yaw change to [-1, 1] for the action output
    normalized_action = clamped_yaw_change / max_yaw_change if max_yaw_change > 1e-6 else 0.0

    return normalized_action


def run_manual_policy(config_name: str, num_steps: int, target_radius: float, render: bool):
    """
    Run a manual policy (approach and circle) in the environment.

    Args:
        config_name: Name of the configuration profile.
        num_steps: Number of simulation steps to run.
        target_radius: Desired circling radius around the landmark.
        render: Whether to render the environment and save a GIF.
    """
    global vis_available # Use the global flag determined by imports

    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}' for manual policy run")

    max_yaw_change = config.world.yaw_angle_range[1] # Get max change from config

    if render and not vis_available:
        print("Rendering requested but libraries not found. Disabling rendering.")
        render = False

    if render:
        os.makedirs(config.visualization.save_dir, exist_ok=True)
        reset_trajectories()

    world = World(world_config=config.world)
    total_reward = 0.0
    episode_frames = []

    print(f"Running manual policy for {num_steps} steps...")
    print(f"Target Radius: {target_radius}, Agent Speed: {world.agent_speed}")
    print(f"Initial Agent Pos: {world.agent.location}, Vel: {world.agent.velocity}")
    print(f"Initial Landmark Pos: {world.true_landmark.location}")

    if render:
        try:
            initial_frame_file = visualize_world(
                world=world,
                vis_config=config.visualization,
                filename=f"manual_policy_frame_000_initial.png",
                collect_for_gif=True
            )
            if initial_frame_file: episode_frames.append(initial_frame_file)
        except Exception as e: print(f"Warning: Visualization failed init state. E: {e}")

    start_time = time.time()
    for step in range(num_steps):
        agent_loc = world.agent.location
        agent_vx = world.agent.velocity.x
        agent_vy = world.agent.velocity.y
        landmark_loc = world.true_landmark.location

        action_normalized = calculate_manual_action(
            agent_loc, agent_vx, agent_vy, landmark_loc, target_radius, max_yaw_change
        )

        # Step the environment with the normalized action
        world.step(action_normalized, training=False) # Use training=False for manual run

        total_reward += world.reward

        if (step + 1) % 20 == 0 or step == num_steps - 1:
             agent_heading_deg = math.degrees(math.atan2(world.agent.velocity.y, world.agent.velocity.x))
             print(f"Step: {step+1}/{num_steps}, "
                   # f"Action: {action_normalized:.3f}, "
                   f"Reward: {world.reward:.3f}, "
                   f"Total Reward: {total_reward:.3f}, "
                   f"Agent-Lmk Dist: {world._calculate_range_measurement(agent_loc, landmark_loc):.2f}, "
                   f"Est Error: {world.error_dist:.2f}, "
                   # f"Agent Pos: ({agent_loc.x:.1f}, {agent_loc.y:.1f}), "
                   f"Agent Hdg: {agent_heading_deg:.1f} deg")


        if render:
            try:
                frame_file = visualize_world(
                    world=world,
                    vis_config=config.visualization,
                    filename=f"manual_policy_frame_{step+1:03d}.png",
                    collect_for_gif=True
                )
                if frame_file: episode_frames.append(frame_file)
            except Exception as e: print(f"Warning: Visualization failed step {step+1}. E: {e}")

    end_time = time.time()
    duration = end_time - start_time
    steps_per_sec = num_steps / duration if duration > 0 else float('inf')

    print("\n--- Manual Policy Run Summary ---")
    print(f"Total Steps: {num_steps}")
    print(f"Final Total Reward: {total_reward:.2f}")
    print(f"Final Agent Position: {world.agent.location}, Velocity: {world.agent.velocity}")
    print(f"Final True Landmark Position: {world.true_landmark.location}")
    est_loc_str = "None"
    if world.estimated_landmark.estimated_location:
         est_loc_str = str(world.estimated_landmark.estimated_location)
    print(f"Final Estimated Landmark Position: {est_loc_str}")
    print(f"Final Error Distance (True vs. Est): {world.error_dist:.2f}")
    print(f"Simulation Duration: {duration:.2f} seconds ({steps_per_sec:.1f} steps/sec)")

    if render and episode_frames:
        gif_filename = f"manual_policy_{config_name}_r{target_radius}.gif"
        print(f"\nCreating GIF: {gif_filename}...")
        try:
            save_gif(
                output_filename=gif_filename,
                vis_config=config.visualization,
                frame_paths=episode_frames,
                delete_frames=config.visualization.delete_frames_after_gif
            )
            print(f"GIF saved to {os.path.join(config.visualization.save_dir, gif_filename)}")
        except Exception as e: print(f"Error creating GIF: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a manual approach-and-circle policy in the landmark tracking environment.")
    parser.add_argument(
        "--config", "-c", type=str, default="default",
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=300,
        help="Number of simulation steps to run."
    )
    parser.add_argument(
        "--radius", "-r", type=float, default=20.0,
        help="Target circling radius around the landmark."
    )
    # Speed is now taken from config
    # parser.add_argument(
    #     "--speed", "-sp", type=float, default=1.5,
    #     help="Target speed of the agent."
    # )
    render_grp = parser.add_mutually_exclusive_group()
    render_grp.add_argument(
        "--render", action="store_true", default=True,
        help="Enable visualization and save a GIF (default)"
    )
    render_grp.add_argument(
        "--no-render", dest="render", action="store_false",
        help="Disable visualization"
    )

    args = parser.parse_args()

    run_manual_policy(
        config_name=args.config,
        num_steps=args.steps,
        target_radius=args.radius,
        # speed=args.speed, # Removed
        render=args.render
    )