from particle_filter import TrackedTargetPF
from world_objects import Object, Location, Velocity
from configs import WorldConfig, LeastSquaresConfig
import numpy as np
import random
import time
from least_squares import TrackedTargetLS

class World():
    """
    Represents the simulation environment containing an agent and landmarks.
    """
    def __init__(self, world_config: WorldConfig):
        """
        Initialize the world simulation environment using configuration.

        Args:
            world_config: Configuration object for the world settings.
            pf_config: Configuration object for the particle filter settings.
            ls_config: Optional configuration for least squares estimator.
            estimation_method: Override for estimation method ('particle_filter' or 'least_squares').
        """
        self.world_config = world_config
        self.dt = world_config.dt
        self.success_threshold = world_config.success_threshold
        
        if isinstance(world_config.estimator_config, LeastSquaresConfig):
            self.estimated_landmark = TrackedTargetLS(config=world_config.estimator_config)
        else:
            self.estimated_landmark = TrackedTargetPF(config=world_config.estimator_config)

        # Initialize True Landmark location
        if world_config.randomize_landmark_initial_location:
            ranges = world_config.landmark_randomization_ranges
            true_landmark_location = Location(
                x=random.uniform(*ranges.x_range),
                y=random.uniform(*ranges.y_range),
                depth=random.uniform(*ranges.depth_range)
            )
        else:
            loc_cfg = world_config.landmark_initial_location
            true_landmark_location = Location(x=loc_cfg.x, y=loc_cfg.y, depth=loc_cfg.depth)

        # Initialize True Landmark velocity
        if world_config.randomize_landmark_initial_velocity:
            ranges = world_config.landmark_velocity_randomization_ranges
            true_landmark_velocity = Velocity(
                x=random.uniform(*ranges.vx_range),
                y=random.uniform(*ranges.vy_range),
                z=random.uniform(*ranges.vz_range)
            )
        else:
            vel_cfg = world_config.landmark_initial_velocity
            true_landmark_velocity = Velocity(x=vel_cfg.x, y=vel_cfg.y, z=vel_cfg.z)

        self.true_landmark = Object(location=true_landmark_location, velocity=true_landmark_velocity, name="true_landmark")

        # Initialize Agent location
        if world_config.randomize_agent_initial_location:
            ranges = world_config.agent_randomization_ranges
            agent_location = Location(
                x=random.uniform(*ranges.x_range),
                y=random.uniform(*ranges.y_range),
                depth=random.uniform(*ranges.depth_range)
            )
        else:
            loc_cfg = world_config.agent_initial_location
            agent_location = Location(x=loc_cfg.x, y=loc_cfg.y, depth=loc_cfg.depth)

        # Initialize Agent velocity (always uses fixed config for now)
        agent_vel_cfg = world_config.agent_initial_velocity
        agent_velocity = Velocity(x=agent_vel_cfg.x, y=agent_vel_cfg.y, z=agent_vel_cfg.z)
        self.agent = Object(location=agent_location, velocity=agent_velocity, name="agent")

        self.objects = [self.true_landmark, self.agent]
        self.current_range = self._calculate_range_measurement(self.agent.location, self.true_landmark.location)
        self.reward = 0.0
        self.error_dist = float('inf')
        self.done = False
        self._update_error_dist()
        self.pf_update_time = 0.0  # Track particle filter update time

    def _calculate_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """Helper function for range measurement (3D slant range)."""
        dx, dy, dz = loc1.x - loc2.x, loc1.y - loc2.y, loc1.depth - loc2.depth
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def _get_noisy_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate range measurement with distance-dependent noise.
        The further the distance, the more noise is added.
        """
        true_range = self._calculate_range_measurement(loc1, loc2)
        
        # Base noise level plus distance-dependent component
        base_noise = self.world_config.range_measurement_base_noise
        distance_factor = self.world_config.range_measurement_distance_factor
        
        # Standard deviation of noise increases with distance
        noise_std_dev = base_noise + distance_factor * true_range
        
        # Add Gaussian noise
        noisy_range = true_range + np.random.normal(0, noise_std_dev)
        
        # Ensure measurement is positive
        return max(0.1, noisy_range)

    def _update_error_dist(self):
        """Helper to calculate 2D distance between true and estimated landmark."""
        if self.estimated_landmark.estimated_location:
            est_loc, true_loc = self.estimated_landmark.estimated_location, self.true_landmark.location
            self.error_dist = np.sqrt((est_loc.x - true_loc.x)**2 + (est_loc.y - true_loc.y)**2)
        else: self.error_dist = float('inf')

    def step(self, action: Velocity, training: bool = True):
        """
        Advance the world state by one time step.

        Args:
            action (Velocity): The velocity action applied to the agent.
            training (bool): Flag indicating if rewards/termination should be calculated.
        """
        self.agent.velocity = action; self.agent.update_position(self.dt)
        self.true_landmark.update_position(self.dt)
        
        # Get noisy range measurement
        noisy_range = self._get_noisy_range_measurement(self.agent.location, self.true_landmark.location)
        
        # Store the noisy range for both state encoding and reward calculations
        self.current_range = noisy_range
        
        has_new_range, effective_measurement = True, noisy_range
        
        # Measure particle filter update time
        pf_start_time = time.time()
        self.estimated_landmark.update(dt=self.dt, has_new_range=has_new_range, 
                                     range_measurement=effective_measurement, 
                                     observer_location=self.agent.location)
        self.pf_update_time = time.time() - pf_start_time
        
        self.reward, self.done = 0.0, False
        if training:
            self._calculate_reward(noisy_range)  # Use noisy range for reward calculation

    def _calculate_reward(self, measurement: float):
        self._update_error_dist()
        
        # Calculate variables needed for rewards
        current_distance = measurement  # distance between agent and target
        position_error = self.error_dist  # prediction error between estimated and actual target position
        
        # Get parameters from config
        reward_scale = self.world_config.reward_scale
        distance_threshold = self.world_config.distance_threshold
        error_threshold = self.world_config.error_threshold
        min_distance = self.world_config.min_safe_distance
        max_distance = self.world_config.out_of_range_threshold
        
        # 1. Distance-based reward
        if current_distance > distance_threshold:
            distance_reward = reward_scale * (0.5 - current_distance / 100.0)  # Normalized to reasonable scale
        else:
            distance_reward = 1.0
        
        # 2. Error-based reward
        if position_error != float('inf') and position_error > error_threshold:
            error_reward = reward_scale * (0.5 - position_error / 100.0)  # Normalized to reasonable scale
        elif position_error != float('inf'):
            error_reward = 1.0
        else:
            error_reward = 0.0  # No estimate available
        
        # 3. Terminal reward
        terminal_reward = 0.0
        if current_distance > max_distance:
            terminal_reward = -100.0
            self.done = True
        elif current_distance < min_distance:
            terminal_reward = -1.0
        
        # Calculate final reward
        self.reward = distance_reward + error_reward + terminal_reward
        
        # Clip reward for stability
        self.reward = np.clip(self.reward, -100.0, 100.0)
    
    def encode_state(self) -> tuple:
        """
        Encodes the current state for the RL agent.

        State: (agent_x, agent_y, agent_vx, agent_vy, est_landmark_x, est_landmark_y, est_landmark_depth (0), current_range)
        """
        agent_loc, agent_vel = self.agent.location, self.agent.velocity
        if self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location; landmark_x, landmark_y, landmark_depth = est_loc.x, est_loc.y, 0.0
        else: landmark_x, landmark_y, landmark_depth = 0.0, 0.0, 0.0
        state = (agent_loc.x, agent_loc.y, agent_vel.x, agent_vel.y, landmark_x, landmark_y, landmark_depth, self.current_range)
        assert len(state) == 8, f"Encoded state length {len(state)} != 8"; return state

    def decode_state(self, state: tuple):
        """
        Decodes a state tuple back into world objects (primarily for debugging/testing).
        Note: Does not reconstruct the full particle filter state.
        """
        if len(state) != 8: print(f"Warning: decode_state expected tuple of length 8, got {len(state)}"); return
        self.agent.location.x, self.agent.location.y = state[0], state[1]
        self.agent.velocity.x, self.agent.velocity.y = state[2], state[3]
        if self.estimated_landmark.estimated_location is None: self.estimated_landmark.estimated_location = Location(0,0,0)
        self.estimated_landmark.estimated_location.x, self.estimated_landmark.estimated_location.y, self.estimated_landmark.estimated_location.depth = state[4], state[5], state[6]
        self.current_range = state[7]; self._update_error_dist(); self.done = self.error_dist < self.success_threshold

    def __str__(self):
        """String representation of the world state."""
        est_str = "PF Not Initialized"
        if self.estimated_landmark.estimated_location:
            est_loc, est_vel_str = self.estimated_landmark.estimated_location, ""
            if self.estimated_landmark.estimated_velocity: est_vel = self.estimated_landmark.estimated_velocity; est_vel_str = f", Vel:(vx:{est_vel.x:.2f}, vy:{est_vel.y:.2f})"
            est_str = f"Est Lmk: Pos:(x:{est_loc.x:.2f}, y:{est_loc.y:.2f}){est_vel_str}"
        true_lmk_str, agent_str = f"True Lmk: {self.true_landmark}", f"Agent: {self.agent}"
        return (f"---\n Reward: {self.reward:.4f}, Done: {self.done}\n Range: {self.current_range:.2f}, Err (2D): {self.error_dist:.2f}\n {agent_str}\n {true_lmk_str}\n {est_str}\n---")