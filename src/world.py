from particle_filter import TrackedTargetPF
from world_objects import Object, Location, Velocity
from configs import WorldConfig, LeastSquaresConfig, ParticleFilterConfig, CORE_STATE_DIM, CORE_ACTION_DIM, TRAJECTORY_REWARD_DIM
import numpy as np
import random
import time
import math
from least_squares import TrackedTargetLS
from collections import deque
from typing import Dict, Tuple, Any

class World():
    """
    Represents the simulation environment containing an agent and landmarks.
    Action is yaw_change (float).
    State is now a trajectory of past N steps.
    """
    def __init__(self, world_config: WorldConfig):
        """
        Initialize the world simulation environment using configuration.
        """
        self.world_config = world_config
        self.dt = world_config.dt
        self.success_threshold = world_config.success_threshold
        self.agent_speed = world_config.agent_speed
        self.max_yaw_change = world_config.yaw_angle_range[1] # Assumes symmetric range around 0
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim # state + action + reward

        # Initialize Estimator
        if isinstance(world_config.estimator_config, LeastSquaresConfig):
            self.estimated_landmark = TrackedTargetLS(config=world_config.estimator_config)
            self.estimator_type = 'least_squares'
        elif isinstance(world_config.estimator_config, ParticleFilterConfig):
            self.estimated_landmark = TrackedTargetPF(config=world_config.estimator_config)
            self.estimator_type = 'particle_filter'
        else:
            raise ValueError("Invalid estimator configuration provided in WorldConfig")

        # Initialize True Landmark
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

        # Initialize Agent
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

        initial_heading = random.uniform(0, 2 * math.pi)
        agent_velocity = Velocity(
            x=self.agent_speed * math.cos(initial_heading),
            y=self.agent_speed * math.sin(initial_heading),
            z=0.0
        )
        self.agent = Object(location=agent_location, velocity=agent_velocity, name="agent")

        # World state variables
        self.objects = [self.true_landmark, self.agent]
        self.current_range = self._get_noisy_range_measurement(self.agent.location, self.true_landmark.location)
        self.reward = 0.0
        self.error_dist = float('inf')
        self.done = False
        self._update_error_dist()
        self.pf_update_time = 0.0

        # Initialize trajectory history
        self._trajectory_history = deque(maxlen=self.trajectory_length)
        self._initialize_trajectory_history()

    def _calculate_planar_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """Helper function for 2D range measurement."""
        dx, dy = loc1.x - loc2.x, loc1.y - loc2.y
        return np.sqrt(dx**2 + dy**2)

    def _calculate_true_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """ Calculate the range in 3D space. """
        dx, dy, dz = loc1.x - loc2.x, loc1.y - loc2.y, loc1.depth - loc2.depth
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def _get_noisy_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """Calculate range measurement with distance-dependent noise."""
        planar_range = self._calculate_planar_range_measurement(loc1, loc2)
        true_range = self._calculate_true_range_measurement(loc1, loc2)
        base_noise = self.world_config.range_measurement_base_noise
        distance_factor = self.world_config.range_measurement_distance_factor
        noise_std_dev = base_noise + distance_factor * true_range
        noisy_range = planar_range + np.random.normal(0, noise_std_dev)
        return max(0.1, noisy_range) # Ensure range is positive

    def _update_error_dist(self):
        """Helper to calculate 2D distance between true and estimated landmark."""
        if self.estimated_landmark and self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location
            true_loc = self.true_landmark.location
            self.error_dist = np.sqrt((est_loc.x - true_loc.x)**2 + (est_loc.y - true_loc.y)**2)
        else:
             self.error_dist = float('inf')

    def _get_basic_state_tuple(self) -> Tuple:
        """ Encodes the instantaneous basic state observation. """
        agent_loc, agent_vel = self.agent.location, self.agent.velocity
        est_loc = self.estimated_landmark.estimated_location if self.estimated_landmark.estimated_location else Location(0,0,0)
        landmark_x, landmark_y, landmark_depth = est_loc.x, est_loc.y, 0.0
        return (agent_loc.x, agent_loc.y, agent_vel.x, agent_vel.y,
                landmark_x, landmark_y, landmark_depth, self.current_range)

    def _initialize_trajectory_history(self):
        """Fills the initial trajectory history with padding or initial state."""
        initial_basic_state = self._get_basic_state_tuple()
        initial_action = 0.0
        initial_reward = 0.0
        initial_feature = np.concatenate([
             np.array(initial_basic_state, dtype=np.float32),
             np.array([initial_action], dtype=np.float32),
             np.array([initial_reward], dtype=np.float32)
        ])
        for _ in range(self.trajectory_length):
             self._trajectory_history.append(initial_feature)

    def step(self, yaw_change_normalized: float, training: bool = True, terminal_step: bool = False):
        """
        Advance the world state by one time step using yaw angle change.
        Updates the internal trajectory history.

        Args:
            yaw_change_normalized (float): The normalized yaw change action [-1, 1].
            training (bool): Flag indicating if rewards/termination should be calculated.
            terminal_step (bool): Flag if this is the forced terminal step.
        """
        # 1. Store previous state info AND the reward obtained FROM the previous step
        prev_basic_state = self._get_basic_state_tuple() # State s_t
        prev_action = yaw_change_normalized           # Action a_t
        reward_from_previous_step = self.reward       # Reward r_t (calculated in the *previous* call to step)

        # 2. Apply action and update world dynamics
        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_vx, current_vy = self.agent.velocity.x, self.agent.velocity.y
        current_heading = math.atan2(current_vy, current_vx)
        new_heading = (current_heading + yaw_change + math.pi) % (2 * math.pi) - math.pi
        new_vx = self.agent_speed * math.cos(new_heading)
        new_vy = self.agent_speed * math.sin(new_heading)
        self.agent.velocity = Velocity(new_vx, new_vy, 0.0)
        self.agent.update_position(self.dt)
        self.true_landmark.update_position(self.dt)

        # 3. Get new observation (noisy range)
        noisy_range = self._get_noisy_range_measurement(self.agent.location, self.true_landmark.location)
        self.current_range = noisy_range

        # 4. Update estimator
        pf_start_time = time.time()
        has_new_range = np.random.rand() <= 0.9  # Simulate lost signal
        self.estimated_landmark.update(dt=self.dt, has_new_range=has_new_range,
                         range_measurement=noisy_range,
                         observer_location=self.agent.location)
        self.pf_update_time = time.time() - pf_start_time

        # 5. Update error dist based on NEW estimate
        self._update_error_dist()

        # 6. Calculate reward for the transition (s_t -> s_{t+1}) and store it for the *next* step's history
        if training:
            self._calculate_reward() # Updates self.reward based on state s_{t+1}. This becomes r_{t+1}
        else:
            self.reward = 0.0 # Reset reward for next step if not training

        # 7. Update trajectory history with the state/action *before* the step and reward *from before* that
        current_feature_vector = np.concatenate([
            np.array(prev_basic_state, dtype=np.float32), # State s_t
            np.array([prev_action], dtype=np.float32),    # Action a_t
            np.array([reward_from_previous_step], dtype=np.float32) # Reward r_t
        ])
        self._trajectory_history.append(current_feature_vector)

        self.done = terminal_step

    def _calculate_reward(self):
        """
        Calculate reward based on current state (AFTER step), matching tracking.py logic conceptually.
        """
        self.reward = 0.0
        estimation_error = self.error_dist
        true_agent_landmark_dist = self._calculate_planar_range_measurement(
            self.agent.location, self.true_landmark.location
        )

        if estimation_error != float('inf') and self.estimated_landmark.estimated_location is not None:
            self.reward += np.clip(np.log(1/estimation_error) + 1, -6, 6)

        self.reward *= 0.05

        if true_agent_landmark_dist < 2: # Hard penalty for being too close
            self.reward -= 1

        self.reward -= 0.001 * true_agent_landmark_dist

    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the current state as a trajectory dictionary.
        """
        trajectory = np.array(self._trajectory_history, dtype=np.float32)
        estimator_state = self.estimated_landmark.encode_state()
        last_basic_state = tuple(trajectory[-1, :CORE_STATE_DIM])
        return {
            'basic_state': last_basic_state,
            'full_trajectory': trajectory,
            'estimator_state': estimator_state
        }

    def decode_state(self, state: dict):
        """ Decodes a state dictionary back into world objects, including trajectory history. """
        if 'full_trajectory' not in state or 'estimator_state' not in state:
            print("Warning: Invalid state format for decoding (missing keys)")
            self.reset()
            return

        full_trajectory = state['full_trajectory']
        estimator_state = state['estimator_state']

        if not isinstance(full_trajectory, np.ndarray) or \
           full_trajectory.shape != (self.trajectory_length, self.feature_dim):
            print(f"Warning: decode_state trajectory has wrong shape/type. "
                  f"Expected: ({self.trajectory_length}, {self.feature_dim}), Got: {full_trajectory.shape if isinstance(full_trajectory, np.ndarray) else type(full_trajectory)}")
            self.reset()
            return

        self._trajectory_history.clear()
        for i in range(self.trajectory_length):
            self._trajectory_history.append(full_trajectory[i])

        last_step_features = full_trajectory[-1]
        last_basic_state = last_step_features[:CORE_STATE_DIM]
        # The reward stored here is reward_t, from the transition ending at state s_t (last_basic_state)
        reward_leading_to_current_state = last_step_features[CORE_STATE_DIM + CORE_ACTION_DIM]

        self.agent.location.x, self.agent.location.y = last_basic_state[0], last_basic_state[1]
        self.agent.velocity.x, self.agent.velocity.y = last_basic_state[2], last_basic_state[3]

        est_x, est_y, est_depth = last_basic_state[4], last_basic_state[5], last_basic_state[6]
        if self.estimated_landmark.estimated_location is None:
             self.estimated_landmark.estimated_location = Location(est_x, est_y, est_depth)
        else:
             self.estimated_landmark.estimated_location.x = est_x
             self.estimated_landmark.estimated_location.y = est_y
             self.estimated_landmark.estimated_location.depth = est_depth

        self.current_range = last_basic_state[7]
        self.reward = reward_leading_to_current_state # Restore the reward that led to this state

        self.estimated_landmark.decode_state(estimator_state)
        self._update_error_dist()

        true_agent_landmark_dist = self._calculate_planar_range_measurement(
            self.agent.location, self.true_landmark.location
        )
        self.done = (self.error_dist <= self.success_threshold or
                     true_agent_landmark_dist < self.world_config.collision_threshold)

    def reset(self):
        """ Resets the world to an initial state defined by the config. """
        self.__init__(self.world_config)

    def __str__(self):
        """String representation of the world state."""
        est_str = f"{self.estimator_type} Not Initialized"
        if self.estimated_landmark and self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location
            est_vel_str = ""
            if self.estimated_landmark.estimated_velocity:
                est_vel = self.estimated_landmark.estimated_velocity
                est_vel_str = f", Vel:(vx:{est_vel.x:.2f}, vy:{est_vel.y:.2f})"
            est_depth_str = f", d:{est_loc.depth:.2f}" if hasattr(est_loc, 'depth') else ""
            est_str = f"Est Lmk ({self.estimator_type}): Pos:(x:{est_loc.x:.2f}, y:{est_loc.y:.2f}{est_depth_str}){est_vel_str}"

        true_lmk_str = f"True Lmk: {self.true_landmark}"
        agent_str = f"Agent: {self.agent}"
        true_dist = self._calculate_planar_range_measurement(self.agent.location, self.true_landmark.location)

        return (f"---\n Reward: {self.reward:.4f}, Done: {self.done}\n"
                f" Noisy Range: {self.current_range:.2f}, True Range: {true_dist:.2f}, Err (2D): {self.error_dist:.2f}\n"
                f" {agent_str}\n {true_lmk_str}\n {est_str}\n---")