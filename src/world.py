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
    State is now a trajectory of past N steps, with state features normalized based on fixed world bounds.
    """
    def __init__(self, world_config: WorldConfig):
        """
        Initialize the world simulation environment using configuration.
        """
        self.world_config = world_config
        self.dt = world_config.dt
        self.success_threshold = world_config.success_threshold
        self.agent_speed = world_config.agent_speed
        self.max_yaw_change = world_config.yaw_angle_range[1] 
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim
        self.new_measurement_probability = world_config.new_measurement_probability

        self.max_world_diagonal_range = np.sqrt(
            (world_config.world_x_bounds[1] - world_config.world_x_bounds[0])**2 +
            (world_config.world_y_bounds[1] - world_config.world_y_bounds[0])**2
        )
        if self.max_world_diagonal_range == 0: self.max_world_diagonal_range = 1.0 # Avoid div by zero

        if isinstance(world_config.estimator_config, LeastSquaresConfig):
            self.estimated_landmark = TrackedTargetLS(config=world_config.estimator_config)
            self.estimator_type = 'least_squares'
        elif isinstance(world_config.estimator_config, ParticleFilterConfig):
            self.estimated_landmark = TrackedTargetPF(config=world_config.estimator_config)
            self.estimator_type = 'particle_filter'
        else:
            print(f"Warning: Estimator config type is {type(world_config.estimator_config)}, defaulting to LeastSquares.")
            self.estimated_landmark = TrackedTargetLS(config=LeastSquaresConfig()) 
            self.estimator_type = 'least_squares'

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

        self.objects = [self.true_landmark, self.agent]
        self.current_range = self._get_noisy_range_measurement(self.agent.location, self.true_landmark.location)
        self.reward = 0.0
        self.error_dist = float('inf')
        self.done = False
        self._update_error_dist()
        self.pf_update_time = 0.0

        self._trajectory_history = deque(maxlen=self.trajectory_length) # Stores (raw_basic_state, raw_action, raw_reward)
        self._initialize_trajectory_history()

    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalizes a value from [min_val, max_val] to [-1, 1]."""
        if max_val == min_val:
            return 0.0 if value == min_val else np.sign(value - min_val) # or handle as error
        return np.clip(2 * (value - min_val) / (max_val - min_val) - 1, -1.0, 1.0)

    def _denormalize_value(self, norm_value: float, min_val: float, max_val: float) -> float:
        """Denormalizes a value from [-1, 1] to [min_val, max_val]."""
        if max_val == min_val:
            return min_val # If bounds are same, denormalized is just that bound
        return (norm_value + 1) / 2 * (max_val - min_val) + min_val

    def _normalize_basic_state(self, raw_state_tuple: Tuple) -> Tuple:
        """Normalizes a raw basic state tuple using fixed world bounds."""
        if not self.world_config.normalize_state:
            return raw_state_tuple

        # Unpack the 9 components
        ax_raw, ay_raw, avx_raw, avy_raw, aheading_raw, \
        lx_raw, ly_raw, ldepth_raw, range_raw = raw_state_tuple
        
        ax_norm = self._normalize_value(ax_raw, self.world_config.world_x_bounds[0], self.world_config.world_x_bounds[1])
        ay_norm = self._normalize_value(ay_raw, self.world_config.world_y_bounds[0], self.world_config.world_y_bounds[1])
        
        # Agent speed is constant, so vx/vy are components. Normalize by speed.
        avx_norm = np.clip(avx_raw / self.agent_speed if self.agent_speed > 1e-6 else 0.0, -1.0, 1.0)
        avy_norm = np.clip(avy_raw / self.agent_speed if self.agent_speed > 1e-6 else 0.0, -1.0, 1.0)

        # Normalize heading: raw heading in [-pi, pi] is mapped to [-1, 1]
        aheading_norm = np.clip(aheading_raw / math.pi, -1.0, 1.0)
        
        # Estimated landmark position is also normalized by world bounds
        lx_norm = self._normalize_value(lx_raw, self.world_config.world_x_bounds[0], self.world_config.world_x_bounds[1])
        ly_norm = self._normalize_value(ly_raw, self.world_config.world_y_bounds[0], self.world_config.world_y_bounds[1])
        
        ldepth_norm = self._normalize_value(ldepth_raw, self.world_config.landmark_depth_bounds[0], self.world_config.landmark_depth_bounds[1])
        
        # Normalize range to [0, 1] using max possible diagonal range in 2D world
        range_norm = np.clip(range_raw / self.max_world_diagonal_range, 0.0, 1.0) # Or map to [-1,1] if preferred

        return (ax_norm, ay_norm, avx_norm, avy_norm, aheading_norm,
                lx_norm, ly_norm, ldepth_norm, range_norm)

    def _calculate_planar_range_measurement(self, loc1: Location, loc2: Location) -> float:
        dx, dy = loc1.x - loc2.x, loc1.y - loc2.y
        return np.sqrt(dx**2 + dy**2)

    def _calculate_true_range_measurement(self, loc1: Location, loc2: Location) -> float:
        dx, dy, dz = loc1.x - loc2.x, loc1.y - loc2.y, loc1.depth - loc2.depth
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def _get_noisy_range_measurement(self, loc1: Location, loc2: Location) -> float:
        planar_range = self._calculate_planar_range_measurement(loc1, loc2)
        true_range = self._calculate_true_range_measurement(loc1, loc2)
        base_noise = self.world_config.range_measurement_base_noise
        distance_factor = self.world_config.range_measurement_distance_factor
        noise_std_dev = base_noise + distance_factor * true_range
        noisy_range = planar_range + np.random.normal(0, noise_std_dev)
        return max(0.1, noisy_range) 

    def _update_error_dist(self):
        if self.estimated_landmark and self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location
            true_loc = self.true_landmark.location
            self.error_dist = np.sqrt((est_loc.x - true_loc.x)**2 + (est_loc.y - true_loc.y)**2)
        else:
             self.error_dist = float('inf')

    def _get_basic_state_tuple(self) -> Tuple:
        """ Encodes the instantaneous basic state (RAW values). """
        agent_loc, agent_vel = self.agent.location, self.agent.velocity
        # Calculate agent heading (orientation in XY plane)
        agent_heading_rad = math.atan2(agent_vel.y, agent_vel.x) # radians in [-pi, pi]

        est_loc = self.estimated_landmark.estimated_location if self.estimated_landmark.estimated_location else Location(0,0,0) # Use (0,0,0) if not initialized
        # Use configured depth bounds if estimator has no depth or for consistent state representation
        # Landmark depth for state is estimated depth, or 0 if no depth info from estimator.
        # For normalization, it uses world_config.landmark_depth_bounds.
        # The state tuple needs a consistent depth value. If estimator provides it, use it. Otherwise 0.
        landmark_depth_for_state = est_loc.depth if hasattr(est_loc, 'depth') and est_loc.depth is not None else 0.0

        return (agent_loc.x, agent_loc.y, agent_vel.x, agent_vel.y, agent_heading_rad,
                est_loc.x, est_loc.y, landmark_depth_for_state, self.current_range)

    def _initialize_trajectory_history(self):
        """Fills the initial trajectory history with raw data."""
        initial_raw_basic_state = self._get_basic_state_tuple()
        initial_action = 0.0 # Raw action (already in [-1,1] for normalized yaw change)
        initial_reward = 0.0 # Raw reward
        
        for _ in range(self.trajectory_length):
             self._trajectory_history.append((initial_raw_basic_state, initial_action, initial_reward))

    def step(self, yaw_change_normalized: float, training: bool = True, terminal_step: bool = False):
        """
        Advance the world state by one time step. Stores raw trajectory internally.
        yaw_change_normalized is already in [-1, 1].
        """
        s_t_raw = self._get_basic_state_tuple() # Raw state before action
        a_t_raw = yaw_change_normalized        # Raw action (already normalized for yaw)

        # Apply action and update world dynamics
        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_vx, current_vy = self.agent.velocity.x, self.agent.velocity.y
        current_heading = math.atan2(current_vy, current_vx)
        new_heading = (current_heading + yaw_change + math.pi) % (2 * math.pi) - math.pi
        new_vx = self.agent_speed * math.cos(new_heading)
        new_vy = self.agent_speed * math.sin(new_heading)
        self.agent.velocity = Velocity(new_vx, new_vy, 0.0)
        self.agent.update_position(self.dt)
        self.true_landmark.update_position(self.dt)

        # Get new observation (noisy range)
        noisy_range = self._get_noisy_range_measurement(self.agent.location, self.true_landmark.location)
        self.current_range = noisy_range # This is s_{t+1}.current_range (raw)

        # Update estimator
        pf_start_time = time.time()
        has_new_range = np.random.rand() <= self.new_measurement_probability 
        self.estimated_landmark.update(dt=self.dt, has_new_range=has_new_range,
                         range_measurement=noisy_range,
                         observer_location=self.agent.location) # Estimator uses raw locations
        self.pf_update_time = time.time() - pf_start_time

        self._update_error_dist() # Based on new estimate (part of s_{t+1})

        # Calculate reward r_t for (s_t, a_t) -> s_{t+1}
        if training:
            r_t_raw = self._calculate_reward() # Updates self.reward with r_t (raw)
        else:
            r_t_raw = 0.0 
            self.reward = r_t_raw

        # Append (s_t_raw, a_t_raw, r_t_raw) to history
        self._trajectory_history.append((s_t_raw, a_t_raw, r_t_raw))

        self.done = terminal_step

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current state (AFTER step). Returns raw reward.
        """
        current_reward = 0.0 # Local variable for this step's reward
        estimation_error = self.error_dist
        true_agent_landmark_dist = self._calculate_planar_range_measurement(
            self.agent.location, self.true_landmark.location
        )

        if estimation_error != float('inf') and self.estimated_landmark.estimated_location is not None:
            current_reward += np.clip(np.log(1/max(estimation_error, 1e-6)) + 1, -1, 5) 
        current_reward *= 0.05
        if true_agent_landmark_dist < 1: 
            current_reward -= 1
        current_reward -= 0.0001 * true_agent_landmark_dist
        
        self.reward = current_reward # Update world's current reward
        return current_reward


    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the current state, providing normalized state features if configured.
        'full_trajectory' contains (normalized_s, action, raw_reward).
        'basic_state' contains the last normalized_s from the trajectory.
        """
        raw_trajectory_tuples = list(self._trajectory_history)
        
        # Pre-allocate numpy array for the output trajectory
        # feature_dim = CORE_STATE_DIM (normalized) + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM
        output_trajectory_features = np.zeros((self.trajectory_length, self.feature_dim), dtype=np.float32)

        for i in range(self.trajectory_length):
            s_raw, a_raw, r_raw = raw_trajectory_tuples[i]
            
            s_norm_tuple = self._normalize_basic_state(s_raw)
            
            output_trajectory_features[i, :CORE_STATE_DIM] = np.array(s_norm_tuple, dtype=np.float32)
            output_trajectory_features[i, CORE_STATE_DIM] = float(a_raw) # Action is already in [-1,1]
            output_trajectory_features[i, CORE_STATE_DIM + CORE_ACTION_DIM] = float(r_raw) # Raw reward

        # 'basic_state' for agents that use only the last state (e.g., PPO, T-SAC actor)
        # This is the normalized version of the last basic state in the trajectory history.
        last_normalized_basic_state_tuple = tuple(output_trajectory_features[-1, :CORE_STATE_DIM])
        
        estimator_state_encoded = self.estimated_landmark.encode_state()

        return {
            'basic_state': last_normalized_basic_state_tuple, # Normalized
            'full_trajectory': output_trajectory_features,    # Contains (normalized_s, action, raw_r)
            'estimator_state': estimator_state_encoded
        }

    def decode_state(self, state: dict):
        """ Decodes a state dictionary. Primarily for specific debugging/reset.
            Assumes 'full_trajectory' in state dict contains (normalized_s, action, raw_r).
        """
        if 'full_trajectory' not in state or 'estimator_state' not in state:
            print("Warning: Invalid state format for decoding (missing keys)")
            self.reset(); return

        loaded_full_trajectory = state['full_trajectory'] # (norm_s, action, raw_r)
        estimator_state_to_decode = state['estimator_state']

        if not isinstance(loaded_full_trajectory, np.ndarray) or \
           loaded_full_trajectory.shape != (self.trajectory_length, self.feature_dim):
            print(f"Warning: decode_state trajectory has wrong shape/type. Expected: ({self.trajectory_length}, {self.feature_dim}), Got: {loaded_full_trajectory.shape if isinstance(loaded_full_trajectory, np.ndarray) else type(loaded_full_trajectory)}")
            self.reset(); return

        self._trajectory_history.clear()
        
        # For repopulating _trajectory_history, we need raw states.
        # We also need to set current world agent/landmark raw positions from the *last* state.
        last_norm_s_tuple_from_loaded_traj = tuple(loaded_full_trajectory[-1, :CORE_STATE_DIM])
        
        # De-normalize the last state to set current world raw attributes
        # Unpack 9 components
        ax_n, ay_n, avx_n, avy_n, aheading_n, \
        lx_n, ly_n, ldepth_n, range_n = last_norm_s_tuple_from_loaded_traj


        self.agent.location.x = self._denormalize_value(ax_n, self.world_config.world_x_bounds[0], self.world_config.world_x_bounds[1])
        self.agent.location.y = self._denormalize_value(ay_n, self.world_config.world_y_bounds[0], self.world_config.world_y_bounds[1])
        self.agent.velocity.x = avx_n * self.agent_speed
        self.agent.velocity.y = avy_n * self.agent_speed
        
        # For the estimated landmark, its raw state is restored by its own decode_state
        # The lx_n, ly_n, ldepth_n are part of the *observation*, not true landmark state.
        # We need to restore the estimator's internal state.
        self.estimated_landmark.decode_state(estimator_state_to_decode)
        
        # The range_n is the normalized version of self.current_range
        self.current_range = range_n * self.max_world_diagonal_range

        # Repopulate _trajectory_history with raw values by denormalizing each step
        for i in range(self.trajectory_length):
            s_norm_tuple_i = tuple(loaded_full_trajectory[i, :CORE_STATE_DIM])
            action_i = loaded_full_trajectory[i, CORE_STATE_DIM]
            reward_i = loaded_full_trajectory[i, CORE_STATE_DIM + CORE_ACTION_DIM]

            # Denormalize s_norm_tuple_i to get s_raw_tuple_i
            # Unpack 9 components
            ax_ni, ay_ni, avx_ni, avy_ni, aheading_ni, \
            lx_ni, ly_ni, ldepth_ni, range_ni = s_norm_tuple_i

            s_raw_ax = self._denormalize_value(ax_ni, self.world_config.world_x_bounds[0], self.world_config.world_x_bounds[1])
            s_raw_ay = self._denormalize_value(ay_ni, self.world_config.world_y_bounds[0], self.world_config.world_y_bounds[1])
            s_raw_avx = avx_ni * self.agent_speed
            s_raw_avy = avy_ni * self.agent_speed
            
            # Denormalize heading: [-1, 1] -> [-pi, pi]
            s_raw_aheading = aheading_ni * math.pi
            
            s_raw_lx = self._denormalize_value(lx_ni, self.world_config.world_x_bounds[0], self.world_config.world_x_bounds[1])
            s_raw_ly = self._denormalize_value(ly_ni, self.world_config.world_y_bounds[0], self.world_config.world_y_bounds[1])
            s_raw_ldepth = self._denormalize_value(ldepth_ni, self.world_config.landmark_depth_bounds[0], self.world_config.landmark_depth_bounds[1])
            s_raw_range = range_ni * self.max_world_diagonal_range
            
            s_raw_tuple_i = (s_raw_ax, s_raw_ay, s_raw_avx, s_raw_avy, s_raw_aheading,
                             s_raw_lx, s_raw_ly, s_raw_ldepth, s_raw_range)
            
            self._trajectory_history.append((s_raw_tuple_i, action_i, reward_i))

        self.reward = loaded_full_trajectory[-1, CORE_STATE_DIM + CORE_ACTION_DIM] # Reward that led to the last state in trajectory
        self._update_error_dist() # Recalculate based on restored estimator
        # Done flag is not directly restored from state dict, depends on current conditions
        true_agent_landmark_dist_after_decode = self._calculate_planar_range_measurement(
            self.agent.location, self.true_landmark.location # True landmark may have moved if not also reset
        )
        self.done = (self.error_dist <= self.success_threshold or
                     true_agent_landmark_dist_after_decode < self.world_config.collision_threshold)


    def reset(self):
        """ Resets the world to an initial state defined by the config. """
        self.__init__(self.world_config)

    def __str__(self):
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