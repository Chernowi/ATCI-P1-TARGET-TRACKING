from particle_filter import TrackedTargetPF
from world_objects import Object, Location, Velocity

import numpy as np


class World():
    def __init__(self, dt: float = 1.0, success_threshold: float = 2.0):  # Add success_threshold parameter
        # --- Particle Filter Configuration ---
        # Match these parameters to how tracking.py initializes its Target/ParticleFilter
        # Example values (adjust as needed):
        num_particles = 1000
        initial_velocity_guess = 0.1
        measurement_noise_stddev = 5  # Corresponds to range noise
        process_noise_pos = 0.02
        process_noise_orient = 0.2
        process_noise_vel = 0.02
        # --- End Configuration ---

        # Use the refactored class
        self.estimated_landmark = TrackedTargetPF(
            num_particles=num_particles,
            initial_velocity_guess=initial_velocity_guess,
            measurement_noise_stddev=measurement_noise_stddev,
            process_noise_pos=process_noise_pos,
            process_noise_orient=process_noise_orient,
            process_noise_vel=process_noise_vel,
            # Add other parameters if defaults in TrackedTargetPF aren't suitable
        )

        true_landmark_location = Location(42, 42, 42)  # Original example
        self.true_landmark = Object(
            location=true_landmark_location, name="true_landmark")
        # Give true landmark a velocity if it should move
        self.true_landmark.velocity = Velocity(
            0.0, 0.0, 0.0)  # Example: Static

        agent_location = Location(0, 0, 0)  # Original example
        self.agent = Object(location=agent_location, name="agent")
        # Agent needs an initial velocity?
        self.agent.velocity = Velocity(0.0, 0.0, 0.0)

        # estimated_landmark is separate logic
        self.objects = [self.true_landmark, self.agent]

        # Calculate initial range (use Location directly)
        self.current_range = self._calculate_range_measurement(
            self.agent.location, self.true_landmark.location)

        self.reward = 0
        self.dt = dt  # Store timestep if needed globally
        
        # Add variables for early termination
        self.success_threshold = success_threshold
        self.error_dist = float('inf')
        self.done = False

    # Helper function for range measurement (can add noise here)
    def _calculate_range_measurement(self, loc1: Location, loc2: Location) -> float:
        # Basic 3D distance, PF currently uses 2D projection internally after this step
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        dz = loc1.depth - loc2.depth  # Use depth too for slant range
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        # --- Optional: Add noise/error similar to tracking.py ---
        # distance *= 1.01 # Systematic error
        # distance += np.random.uniform(-0.001, +0.001) # Noise
        # ---

        # --- Optional: Simulate max range / drop rate ---
        # max_range_setting = 200.0 # Example
        # drop_probability = 0.1 # Example
        # if distance > max_range_setting or np.random.rand() < drop_probability:
        #     return -1.0 # Indicate no valid measurement
        # ---

        # The refactored PF expects the 2D planar range if depth is handled separately.
        # If the PF handles slant range directly, return 'distance'.
        # If it expects planar range (like original seems to), convert back:
        # planar_distance = np.sqrt(max(0, distance**2 - dz**2)) # Ensure non-negative argument
        # return planar_distance

        # Let's assume for now the PF expects the direct slant range 'z'
        return distance

    def step(self, action: Velocity, training: bool = True):
        # Update agent based on action
        self.agent.velocity = action
        self.agent.update_position()  # Updates agent.location

        # Update true landmark (if it moves)
        self.true_landmark.update_position()  # Updates true_landmark.location

        # Calculate the new range measurement
        # Assume has_new_range is always True for simplicity here, adjust if needed
        has_new_range = True
        measurement = self._calculate_range_measurement(
            self.agent.location, self.true_landmark.location)

        # Update the particle filter
        self.estimated_landmark.update(dt=self.dt,  # Pass the time step
                                       has_new_range=has_new_range,
                                       range_measurement=measurement,
                                       observer_location=self.agent.location)  # Pass agent's location object
        if training:
            # Calculate reward based on the *estimated* location from the PF
            if self.estimated_landmark.estimated_location is not None:
                # Use distance between estimated Location and true Location
                est_loc = self.estimated_landmark.estimated_location
                true_loc = self.true_landmark.location
                # Calculate 2D or 3D distance based on what's relevant
                self.error_dist = np.sqrt((est_loc.x - true_loc.x) **
                                    2 + (est_loc.y - true_loc.y)**2)  # 2D example
                self.reward = 1 / (self.error_dist + 1e-6)
                
                # Check if we've reached the success threshold
                if self.error_dist < self.success_threshold:
                    self.done = True
                    self.reward += 10.0  # Bonus reward for success
            else:
                # Handle case where estimate isn't available yet (PF not initialized)
                self.reward = 0  # Or some default low reward
                self.error_dist = float('inf')
                
            self.reward -= 1  # Penalize for each step taken (to encourage efficiency)

        self.current_range = measurement  # Store the latest measurement

    def encode_state(self):
        """Return the state as a tuple (agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, previous_range)
         where landmark is the estimated landmark."""
        agent_x = self.agent.location.x
        agent_y = self.agent.location.y
        agent_vx = self.agent.velocity.x
        agent_vy = self.agent.velocity.y
        if self.estimated_landmark.estimated_location is None:
            landmark_x = self.agent.location.x
            landmark_y = self.agent.location.y
            landmark_depth = self.agent.location.depth
        else:
            landmark_x = self.estimated_landmark.estimated_location.x
            landmark_y = self.estimated_landmark.estimated_location.y
            landmark_depth = self.estimated_landmark.estimated_location.depth

        current_range = self.current_range

        # Return the state as a tuple
        return (agent_x, agent_y, agent_vx, agent_vy,
                landmark_x, landmark_y, landmark_depth, current_range)

    def decode_state(self, state: tuple):
        """Decode the state tuple into the agent and landmark locations and velocities."""
        agent_location = Location(state[0], state[1], 0)
        agent_velocity = Velocity(state[2], state[3], 0)
        landmark_location = Location(state[4], state[5], state[6])

        self.agent.location = agent_location
        self.agent.velocity = agent_velocity
        self.estimated_landmark.location = landmark_location

    def __str__(self):
        est_str = "Not Initialized"
        if self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location
            # Add velocity if desired
            # est_vel = self.estimated_landmark.estimated_velocity
            est_str = f"position: (x:{est_loc.x:.2f}, y:{est_loc.y:.2f}, depth:{est_loc.depth:.2f})"

        return f"""World:
        {"-"*15}
        reward: {self.reward:.4f},
        current_range: {self.current_range:.2f}
        error_dist: {self.error_dist:.2f}
        {"-"*15}
        agent: {self.agent}
        {"-"*15}
        true landmark: {self.true_landmark}
        {"-"*15}
        estimated landmark: {est_str}
        {"-"*15}"""


if __name__ == "__main__":
    world = World()
    print(world)
    for i in range(10):
        world.step(Velocity(1, 1, 0))
    print(world)
