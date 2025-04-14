from particle_filter import TrackedTargetPF
from world_objects import Object, Location, Velocity
from configs import WorldConfig, LeastSquaresConfig, ParticleFilterConfig
import numpy as np
import random
import time
import math
from least_squares import TrackedTargetLS

class World():
    """
    Represents the simulation environment containing an agent and landmarks.
    Action is now yaw_change (float).
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

        if isinstance(world_config.estimator_config, LeastSquaresConfig):
            self.estimated_landmark = TrackedTargetLS(config=world_config.estimator_config)
            self.estimator_type = 'least_squares'
        elif isinstance(world_config.estimator_config, ParticleFilterConfig):
            self.estimated_landmark = TrackedTargetPF(config=world_config.estimator_config)
            self.estimator_type = 'particle_filter'
        else:
            raise ValueError("Invalid estimator configuration provided in WorldConfig")


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
                depth=random.uniform(*ranges.depth_range) # Agent depth usually 0? Check ranges.
            )
        else:
            loc_cfg = world_config.agent_initial_location
            agent_location = Location(x=loc_cfg.x, y=loc_cfg.y, depth=loc_cfg.depth)

        # Initialize Agent velocity based on a random initial heading and constant speed
        initial_heading = random.uniform(0, 2 * math.pi)
        agent_velocity = Velocity(
            x=self.agent_speed * math.cos(initial_heading),
            y=self.agent_speed * math.sin(initial_heading),
            z=0.0 # Assume agent moves in 2D plane initially
        )
        self.agent = Object(location=agent_location, velocity=agent_velocity, name="agent")

        self.objects = [self.true_landmark, self.agent]
        self.current_range = self._calculate_range_measurement(self.agent.location, self.true_landmark.location)
        self.reward = 0.0
        self.error_dist = float('inf')
        self.done = False
        self._update_error_dist()
        self.pf_update_time = 0.0

    def _calculate_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """Helper function for range measurement (3D slant range)."""
        dx, dy = loc1.x - loc2.x, loc1.y - loc2.y
        # Use depth difference for 3D range
        # dz = loc1.depth - loc2.depth
        # return np.sqrt(dx**2 + dy**2 + dz**2)
        # Using 2D range as seems intended by state representation/PF
        return np.sqrt(dx**2 + dy**2)


    def _get_noisy_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """Calculate range measurement with distance-dependent noise."""
        true_range = self._calculate_range_measurement(loc1, loc2)
        base_noise = self.world_config.range_measurement_base_noise
        distance_factor = self.world_config.range_measurement_distance_factor
        noise_std_dev = base_noise + distance_factor * true_range
        noisy_range = true_range + np.random.normal(0, noise_std_dev)
        return max(0.1, noisy_range)

    def _update_error_dist(self):
        """Helper to calculate 2D distance between true and estimated landmark."""
        if self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location
            true_loc = self.true_landmark.location
            self.error_dist = np.sqrt((est_loc.x - true_loc.x)**2 + (est_loc.y - true_loc.y)**2)
        else:
             self.error_dist = float('inf')

    def step(self, yaw_change_normalized: float, training: bool = True, terminal_step: bool = False):
        """
        Advance the world state by one time step using yaw angle change.

        Args:
            yaw_change_normalized (float): The normalized yaw change action from RL agent [-1, 1].
            training (bool): Flag indicating if rewards/termination should be calculated.
            terminal_step (bool): Flag if this is the forced terminal step.
        """
        # De-normalize yaw change action
        yaw_change = yaw_change_normalized * self.max_yaw_change

        # Update agent heading and velocity
        current_vx = self.agent.velocity.x
        current_vy = self.agent.velocity.y
        current_heading = math.atan2(current_vy, current_vx)

        new_heading = current_heading + yaw_change
        # Normalize angle to [-pi, pi] - optional but good practice
        new_heading = (new_heading + math.pi) % (2 * math.pi) - math.pi

        new_vx = self.agent_speed * math.cos(new_heading)
        new_vy = self.agent_speed * math.sin(new_heading)
        self.agent.velocity = Velocity(new_vx, new_vy, 0.0) # Update agent velocity

        # Update positions
        self.agent.update_position(self.dt)
        self.true_landmark.update_position(self.dt)

        # Check for termination conditions BEFORE calculating reward for the new state
        if self.error_dist <= self.success_threshold:
            self.done = True
        elif terminal_step:
             self.done = True
        # Add other termination conditions? (e.g., out of bounds)

        # Get noisy range measurement for the NEW state
        noisy_range = self._get_noisy_range_measurement(self.agent.location, self.true_landmark.location)
        self.current_range = noisy_range # Store noisy range for state encoding

        # Update estimator
        has_new_range, effective_measurement = True, noisy_range
        pf_start_time = time.time()
        self.estimated_landmark.update(dt=self.dt, has_new_range=has_new_range,
                                     range_measurement=effective_measurement,
                                     observer_location=self.agent.location)
        self.pf_update_time = time.time() - pf_start_time

        # Update error distance based on potentially new estimate
        self._update_error_dist()

        # Calculate reward based on the NEW state and action outcome
        if training:
            self._calculate_reward(noisy_range)
        else:
            self.reward = 0.0 # No reward calculation needed if not training

        # Update done flag based on success threshold AFTER potentially new estimate
        if self.error_dist <= self.success_threshold:
            self.done = True


    def _calculate_reward(self, measurement: float):
        """Calculate reward based on current state."""
        # Error distance (self.error_dist) is updated before calling this
        current_distance = measurement # Use noisy measurement
        position_error = self.error_dist

        reward_scale = self.world_config.reward_scale
        distance_threshold = self.world_config.distance_threshold
        error_threshold = self.world_config.error_threshold
        min_distance = self.world_config.min_safe_distance
        max_distance = self.world_config.out_of_range_threshold

        # 1. Distance-based reward (Encourage being within optimal range)
        if min_distance < current_distance < distance_threshold:
            distance_reward = 1.0 # Good range
        elif current_distance <= min_distance:
            distance_reward = -1.0 # Too close penalty
        else: # current_distance >= distance_threshold
            # Penalize being too far, scaled by how far
            distance_reward = reward_scale * (distance_threshold - current_distance)

        # 2. Error-based reward (Encourage low estimation error)
        if position_error != float('inf') and position_error <= error_threshold:
             error_reward = 10.0 # High reward for accurate estimation
        elif position_error != float('inf'):
             # Penalize large error, scaled by how large
             error_reward = reward_scale * (error_threshold - position_error)
        else:
             error_reward = -1.0 # Penalize having no estimate yet

        # 3. Success Bonus (applied only if termination condition met)
        success_bonus = 0.0
        if self.done and self.error_dist <= self.success_threshold:
             success_bonus = self.world_config.success_bonus

        # 4. Out of Range Penalty (applied only if termination condition met)
        # oob_penalty = 0.0
        # if self.done and current_distance > max_distance: # Only penalize if OOB *causes* termination? Or always? Let's apply if OOB happens
        #      oob_penalty = -self.world_config.out_of_range_penalty

        # Combine rewards
        self.reward = distance_reward + error_reward + success_bonus # + oob_penalty
        # self.reward = distance_reward + error_reward # Simpler version

        # Add step penalty?
        # self.reward -= self.world_config.step_penalty

        # Clip reward for stability
        self.reward = np.clip(self.reward, -150.0, 150.0) # Adjusted clipping range


    def encode_state(self) -> dict:
        """
        Encodes the current state for the RL agent, including estimator state.
        """
        agent_loc, agent_vel = self.agent.location, self.agent.velocity
        if self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location
            landmark_x, landmark_y, landmark_depth = est_loc.x, est_loc.y, 0.0
        else:
            landmark_x, landmark_y, landmark_depth = 0.0, 0.0, 0.0

        basic_state = (agent_loc.x, agent_loc.y, agent_vel.x, agent_vel.y,
                      landmark_x, landmark_y, landmark_depth, self.current_range)

        estimator_state = self.estimated_landmark.encode_state()

        return {
            'basic_state': basic_state,
            'estimator_state': estimator_state
        }

    def encode_state_tuple(self) -> tuple:
        """ Returns just the basic state tuple. """
        state_dict = self.encode_state()
        return state_dict['basic_state']

    def decode_state(self, state: dict):
        """ Decodes a state dictionary back into world objects. """
        if 'basic_state' not in state or 'estimator_state' not in state:
            print("Warning: Invalid state format for decoding")
            return

        basic_state = state['basic_state']
        estimator_state = state['estimator_state']

        if len(basic_state) != 8:
            print(f"Warning: decode_state expected basic_state tuple of length 8, got {len(basic_state)}")
            return

        self.agent.location.x, self.agent.location.y = basic_state[0], basic_state[1]
        self.agent.velocity.x, self.agent.velocity.y = basic_state[2], basic_state[3]

        if self.estimated_landmark.estimated_location is None:
             # If PF/LS wasn't initialized, create a dummy location
             self.estimated_landmark.estimated_location = Location(basic_state[4], basic_state[5], basic_state[6])


        self.current_range = basic_state[7]

        self.estimated_landmark.decode_state(estimator_state)

        self._update_error_dist()
        self.done = self.error_dist < self.success_threshold

    def __str__(self):
        """String representation of the world state."""
        est_str = f"{self.estimator_type} Not Initialized"
        if self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location
            est_vel_str = ""
            if self.estimated_landmark.estimated_velocity:
                est_vel = self.estimated_landmark.estimated_velocity
                est_vel_str = f", Vel:(vx:{est_vel.x:.2f}, vy:{est_vel.y:.2f})"
            est_str = f"Est Lmk ({self.estimator_type}): Pos:(x:{est_loc.x:.2f}, y:{est_loc.y:.2f}){est_vel_str}"
        true_lmk_str = f"True Lmk: {self.true_landmark}"
        agent_str = f"Agent: {self.agent}" # Agent velocity shows current velocity
        return (f"---\n Reward: {self.reward:.4f}, Done: {self.done}\n"
                f" Range: {self.current_range:.2f}, Err (2D): {self.error_dist:.2f}\n"
                f" {agent_str}\n {true_lmk_str}\n {est_str}\n---")