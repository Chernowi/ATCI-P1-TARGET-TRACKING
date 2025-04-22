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
        # Success threshold based on estimation error
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
        self.pf_update_time = 0.0 # Specific to estimator, maybe generalize later

        # Initialize trajectory history
        self._trajectory_history = deque(maxlen=self.trajectory_length)
        self._initialize_trajectory_history()

    def _calculate_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """Helper function for 2D range measurement."""
        dx, dy = loc1.x - loc2.x, loc1.y - loc2.y
        return np.sqrt(dx**2 + dy**2)

    def _get_noisy_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """Calculate range measurement with distance-dependent noise."""
        true_range = self._calculate_range_measurement(loc1, loc2)
        base_noise = self.world_config.range_measurement_base_noise
        distance_factor = self.world_config.range_measurement_distance_factor
        noise_std_dev = base_noise + distance_factor * true_range
        noisy_range = true_range + np.random.normal(0, noise_std_dev)
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
        # Use estimate if available, otherwise use agent's position as placeholder? Or zeros? Using zeros.
        est_loc = self.estimated_landmark.estimated_location if self.estimated_landmark.estimated_location else Location(0,0,0)
        landmark_x, landmark_y, landmark_depth = est_loc.x, est_loc.y, 0.0 # Use estimated landmark pos (depth assumed 0 for basic state)

        return (agent_loc.x, agent_loc.y, agent_vel.x, agent_vel.y,
                landmark_x, landmark_y, landmark_depth, self.current_range)

    def _initialize_trajectory_history(self):
        """Fills the initial trajectory history with padding or initial state."""
        initial_basic_state = self._get_basic_state_tuple()
        initial_action = 0.0 # Normalized action
        initial_reward = 0.0
        # Padding with zeros
        # padding_feature = np.zeros(self.feature_dim, dtype=np.float32)
        # Padding with repeated initial state/action/reward
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
        # 1. Store previous state information for the trajectory history
        prev_basic_state = self._get_basic_state_tuple()
        prev_action = yaw_change_normalized
        prev_reward = self.reward # Reward received *from* the previous state transition

        # 2. Apply action and update world dynamics
        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_vx, current_vy = self.agent.velocity.x, self.agent.velocity.y
        current_heading = math.atan2(current_vy, current_vx)
        new_heading = (current_heading + yaw_change + math.pi) % (2 * math.pi) - math.pi # Wrap angle correctly
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
        self.estimated_landmark.update(dt=self.dt, has_new_range=True, # Assume new range is always available after step
                                     range_measurement=noisy_range,
                                     observer_location=self.agent.location)
        self.pf_update_time = time.time() - pf_start_time

        # 5. Update error dist based on NEW estimate
        self._update_error_dist()

        # 6. Calculate reward for the completed transition
        if training:
            self._calculate_reward() # Reward is based on the state *after* the action
        else:
            self.reward = 0.0 # Use current self.reward (which is from prev step) for history if not training

        # 7. Update trajectory history with the state *before* the action
        current_feature_vector = np.concatenate([
            np.array(prev_basic_state, dtype=np.float32),
            np.array([prev_action], dtype=np.float32),
            np.array([self.reward], dtype=np.float32) # Use the newly calculated reward
        ])
        self._trajectory_history.append(current_feature_vector)

        # 8. Check for termination conditions (AFTER reward and history update)
        self.done = False
        true_agent_landmark_dist = self._calculate_range_measurement(
            self.agent.location, self.true_landmark.location
        )

        # Done if estimation error is low enough (success)
        if self.error_dist <= self.success_threshold:
            # print(f"Done: Success threshold reached ({self.error_dist:.3f} <= {self.success_threshold:.3f})") # Debug
            self.done = True
        # Done if agent collides with true landmark
        elif true_agent_landmark_dist < self.world_config.collision_threshold:
            # print(f"Done: Collision threshold reached ({true_agent_landmark_dist:.3f} < {self.world_config.collision_threshold:.3f})") # Debug
            self.done = True
        # Done if forced terminal step
        elif terminal_step:
            # print("Done: Forced terminal step") # Debug
            self.done = True
        # Optional: Done if agent goes too far (out of bounds) - requires config param
        # elif 'world_boundary_range' in self.world_config.__dict__ and \
        #      true_agent_landmark_dist > self.world_config.world_boundary_range:
        #     self.done = True


    def _calculate_reward(self):
        """
        Calculate reward based on current state (AFTER step), matching tracking.py logic conceptually.
        Reward is assigned to self.reward.
        """
        rew = 0.0

        # --- Config parameters ---
        # Error related
        rew_err_th = self.world_config.reward_error_threshold
        low_error_bonus = self.world_config.low_error_bonus
        high_error_penalty_factor = self.world_config.high_error_penalty_factor
        uninitialized_penalty = self.world_config.uninitialized_penalty
        # Distance related (to TRUE landmark)
        rew_dis_th = self.world_config.reward_distance_threshold
        close_distance_bonus = self.world_config.close_distance_bonus
        distance_reward_scale = self.world_config.distance_reward_scale
        max_distance_for_reward = self.world_config.max_distance_for_reward
        # Penalty related
        max_range = self.world_config.max_observable_range
        out_of_range_penalty = self.world_config.out_of_range_penalty
        collision_threshold = self.world_config.collision_threshold
        landmark_collision_penalty = self.world_config.landmark_collision_penalty
        # Optional out of bounds penalty
        # out_of_bounds_penalty = getattr(self.world_config, 'out_of_bounds_penalty', 0.0)
        # world_boundary_range = getattr(self.world_config, 'world_boundary_range', float('inf'))

        # --- Get current state values ---
        estimation_error = self.error_dist # Based on updated estimate
        true_agent_landmark_dist = self._calculate_range_measurement(
            self.agent.location, self.true_landmark.location
        )

        # --- 1. Reward/Penalty based on Estimation Error ---
        # Check if estimator has produced a valid location
        if estimation_error != float('inf') and self.estimated_landmark.estimated_location is not None:
            if estimation_error < rew_err_th:
                # Bonus for low estimation error
                rew += low_error_bonus
            else:
                # Penalize proportionally to how much error exceeds the threshold
                # Ensures penalty increases as error gets worse
                rew -= high_error_penalty_factor * (estimation_error - rew_err_th)
        else:
            # Penalty if the estimator hasn't produced a valid estimate yet
            rew -= uninitialized_penalty

        # --- 2. Reward based on Agent's distance to TRUE Landmark ---
        if true_agent_landmark_dist < rew_dis_th:
            # Bonus for being very close to the true landmark
            rew += close_distance_bonus
        elif true_agent_landmark_dist < max_distance_for_reward:
            # Scaled reward for being reasonably close (encourages approach)
            # Reward decreases as distance increases up to max_distance_for_reward
            rew += distance_reward_scale * (max_distance_for_reward - true_agent_landmark_dist)
        # No specific reward/penalty if beyond max_distance_for_reward but within max_range

        # --- 3. Penalties based on Agent's distance to TRUE Landmark ---
        # Out of Range Penalty (based on true distance)
        if true_agent_landmark_dist > max_range:
            rew -= out_of_range_penalty

        # Landmark Collision Penalty (based on true distance)
        if true_agent_landmark_dist < collision_threshold:
            rew -= landmark_collision_penalty

        # Optional: Out of Bounds Penalty
        # if true_agent_landmark_dist > world_boundary_range:
        #     rew -= out_of_bounds_penalty

        # --- 4. Clipping ---
        # Use clipping values from the original implementation or config
        self.reward = np.clip(rew, -150.0, 150.0)


    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the current state as a trajectory dictionary.
        """
        trajectory = np.array(self._trajectory_history, dtype=np.float32)
        estimator_state = self.estimated_landmark.encode_state()

        # Basic state is now implicitly the last element of the trajectory
        last_basic_state = tuple(trajectory[-1, :CORE_STATE_DIM])

        return {
            'basic_state': last_basic_state, # Last basic state for convenience/PPO
            'full_trajectory': trajectory,   # Shape (N, feature_dim)
            'estimator_state': estimator_state
        }

    def decode_state(self, state: dict):
        """ Decodes a state dictionary back into world objects, including trajectory history. """
        if 'full_trajectory' not in state or 'estimator_state' not in state:
            print("Warning: Invalid state format for decoding (missing keys)")
            # Attempt recovery if possible, or re-initialize? Re-init is safer.
            self.reset()
            return

        full_trajectory = state['full_trajectory']
        estimator_state = state['estimator_state']

        if not isinstance(full_trajectory, np.ndarray) or \
           full_trajectory.shape != (self.trajectory_length, self.feature_dim):
            print(f"Warning: decode_state trajectory has wrong shape/type. "
                  f"Expected: ({self.trajectory_length}, {self.feature_dim}), Got: {full_trajectory.shape if isinstance(full_trajectory, np.ndarray) else type(full_trajectory)}")
            # Attempt recovery or re-initialize
            self.reset()
            return

        # Restore trajectory history deque
        self._trajectory_history.clear()
        for i in range(self.trajectory_length):
            self._trajectory_history.append(full_trajectory[i])

        # Restore agent/landmark/estimator state from the *last* entry in the trajectory
        # This reflects the state *after* the last action was taken.
        last_step_features = full_trajectory[-1]
        last_basic_state = last_step_features[:CORE_STATE_DIM]
        # last_action = last_step_features[CORE_STATE_DIM] # Action leading to this state
        last_reward = last_step_features[CORE_STATE_DIM + CORE_ACTION_DIM] # Reward resulting from the transition *to* this state

        # Restore Agent state
        self.agent.location.x, self.agent.location.y = last_basic_state[0], last_basic_state[1]
        self.agent.velocity.x, self.agent.velocity.y = last_basic_state[2], last_basic_state[3]
        # Agent Z velocity and depth are assumed 0 or handled elsewhere if needed

        # Restore Estimated landmark location (part of basic_state)
        est_x, est_y, est_depth = last_basic_state[4], last_basic_state[5], last_basic_state[6]
        if self.estimated_landmark.estimated_location is None:
             self.estimated_landmark.estimated_location = Location(est_x, est_y, est_depth)
        else:
             self.estimated_landmark.estimated_location.x = est_x
             self.estimated_landmark.estimated_location.y = est_y
             self.estimated_landmark.estimated_location.depth = est_depth # Depth might be 0 here

        self.current_range = last_basic_state[7] # Restore last observed noisy range
        self.reward = last_reward # Restore the reward associated with reaching this state

        # Restore the internal state of the estimator
        # Crucial: Need to ensure the true landmark state is *not* overwritten here,
        # only the estimator's belief. The true landmark is part of the environment dynamics.
        self.estimated_landmark.decode_state(estimator_state)

        # Update error and done status based on restored *estimated* state and the *true* landmark state
        # The true landmark state should ideally be restored separately if saving/loading the whole env state.
        # Assuming self.true_landmark is correctly maintained or restored elsewhere.
        self._update_error_dist()

        # Recalculate done status based on restored state
        true_agent_landmark_dist = self._calculate_range_measurement(
            self.agent.location, self.true_landmark.location
        )
        self.done = (self.error_dist <= self.success_threshold or
                     true_agent_landmark_dist < self.world_config.collision_threshold)
        # Add other done conditions if applicable (e.g., out of bounds)


    def reset(self):
        """ Resets the world to an initial state defined by the config. """
        # Re-initialize agent, landmark, estimator using the stored config
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
            # Include depth if relevant
            est_depth_str = f", d:{est_loc.depth:.2f}" if hasattr(est_loc, 'depth') else ""
            est_str = f"Est Lmk ({self.estimator_type}): Pos:(x:{est_loc.x:.2f}, y:{est_loc.y:.2f}{est_depth_str}){est_vel_str}"

        true_lmk_str = f"True Lmk: {self.true_landmark}"
        agent_str = f"Agent: {self.agent}"
        true_dist = self._calculate_range_measurement(self.agent.location, self.true_landmark.location)

        return (f"---\n Reward: {self.reward:.4f}, Done: {self.done}\n"
                f" Noisy Range: {self.current_range:.2f}, True Range: {true_dist:.2f}, Err (2D): {self.error_dist:.2f}\n"
                f" {agent_str}\n {true_lmk_str}\n {est_str}\n---")
