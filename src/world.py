from particle_filter import TrackedTargetPF
from world_objects import Object, Location, Velocity
from configs import WorldConfig, ParticleFilterConfig

import numpy as np


class World():
    """
    Represents the simulation environment containing an agent and landmarks.
    The agent learns to navigate towards a landmark using range measurements
    processed by a particle filter.
    """

    def __init__(self, world_config: WorldConfig, pf_config: ParticleFilterConfig):
        """
        Initialize the world simulation environment using configuration.

        Args:
            world_config: Configuration object for the world settings.
            pf_config: Configuration object for the particle filter settings.
        """
        self.world_config = world_config
        self.pf_config = pf_config
        self.dt = world_config.dt
        self.success_threshold = world_config.success_threshold

        # Initialize the Particle Filter tracker
        self.estimated_landmark = TrackedTargetPF(config=pf_config)

        # Initialize True Landmark
        true_landmark_loc_cfg = world_config.landmark_initial_location
        true_landmark_vel_cfg = world_config.landmark_initial_velocity
        true_landmark_location = Location(
            x=true_landmark_loc_cfg.x, y=true_landmark_loc_cfg.y, depth=true_landmark_loc_cfg.depth
        )
        true_landmark_velocity = Velocity(
            x=true_landmark_vel_cfg.x, y=true_landmark_vel_cfg.y, z=true_landmark_vel_cfg.z
        )
        self.true_landmark = Object(
            location=true_landmark_location, velocity=true_landmark_velocity, name="true_landmark"
        )

        # Initialize Agent
        agent_loc_cfg = world_config.agent_initial_location
        agent_vel_cfg = world_config.agent_initial_velocity
        agent_location = Location(
            x=agent_loc_cfg.x, y=agent_loc_cfg.y, depth=agent_loc_cfg.depth
        )
        agent_velocity = Velocity(
            x=agent_vel_cfg.x, y=agent_vel_cfg.y, z=agent_vel_cfg.z
        )
        self.agent = Object(
            location=agent_location, velocity=agent_velocity, name="agent"
        )

        self.objects = [self.true_landmark, self.agent]

        # Initial calculations
        self.current_range = self._calculate_range_measurement(
            self.agent.location, self.true_landmark.location
        )
        self.reward = 0.0
        self.error_dist = float('inf')
        self.done = False
        self._update_error_dist()

    def _calculate_range_measurement(self, loc1: Location, loc2: Location) -> float:
        """Helper function for range measurement (3D slant range)."""
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        dz = loc1.depth - loc2.depth
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        return distance

    def _update_error_dist(self):
        """Helper to calculate 2D distance between true and estimated landmark."""
        if self.estimated_landmark.estimated_location is not None:
            est_loc = self.estimated_landmark.estimated_location
            true_loc = self.true_landmark.location
            self.error_dist = np.sqrt((est_loc.x - true_loc.x)**2 +
                                      (est_loc.y - true_loc.y)**2)
        else:
            self.error_dist = float('inf')

    def step(self, action: Velocity, training: bool = True):
        """
        Advance the world state by one time step.

        Args:
            action (Velocity): The velocity action applied to the agent.
            training (bool): Flag indicating if rewards/termination should be calculated.
        """
        self.agent.velocity = action
        self.agent.update_position(self.dt)
        self.true_landmark.update_position(self.dt)

        measurement = self._calculate_range_measurement(
            self.agent.location, self.true_landmark.location
        )
        self.current_range = measurement

        has_new_range = True  # Assume measurement always valid unless logic changes
        effective_measurement = measurement

        self.estimated_landmark.update(
            dt=self.dt,
            has_new_range=has_new_range,
            range_measurement=effective_measurement,
            observer_location=self.agent.location
        )

        self.reward = 0.0
        self.done = False
        if training:
            self._update_error_dist()
            if self.error_dist != float('inf'):
                self.reward = 1.0 / (self.error_dist + 1e-6)
            else:
                self.reward = 0.0
            self.reward -= self.world_config.step_penalty
            if self.error_dist < self.success_threshold:
                self.done = True
                self.reward += self.world_config.success_bonus
            if measurement > self.world_config.out_of_range_threshold:
                self.done = True
                self.reward -= self.world_config.out_of_range_penalty

    def encode_state(self) -> tuple:
        """
        Encodes the current state for the RL agent.

        State: (agent_x, agent_y, agent_vx, agent_vy, est_landmark_x, est_landmark_y, est_landmark_depth (0), current_range)
        """
        agent_loc = self.agent.location
        agent_vel = self.agent.velocity

        if self.estimated_landmark.estimated_location is not None:
            est_loc = self.estimated_landmark.estimated_location
            landmark_x = est_loc.x
            landmark_y = est_loc.y
            landmark_depth = 0.0  # PF estimate is 2D
        else:
            # Use default values if PF estimate unavailable
            landmark_x = 0.0
            landmark_y = 0.0
            landmark_depth = 0.0

        state = (
            agent_loc.x, agent_loc.y,
            agent_vel.x, agent_vel.y,
            landmark_x, landmark_y, landmark_depth,
            self.current_range
        )
        assert len(state) == 8, f"Encoded state length {len(state)} != 8"
        return state

    def decode_state(self, state: tuple):
        """
        Decodes a state tuple back into world objects (primarily for debugging/testing).
        Note: Does not reconstruct the full particle filter state.
        """
        if len(state) != 8:
            print(
                f"Warning: decode_state expected tuple of length 8, got {len(state)}")
            return

        self.agent.location.x = state[0]
        self.agent.location.y = state[1]
        self.agent.velocity.x = state[2]
        self.agent.velocity.y = state[3]

        # Set the PF's estimate directly (approximation)
        if self.estimated_landmark.estimated_location is None:
            self.estimated_landmark.estimated_location = Location(0, 0, 0)
        self.estimated_landmark.estimated_location.x = state[4]
        self.estimated_landmark.estimated_location.y = state[5]
        self.estimated_landmark.estimated_location.depth = state[6]

        self.current_range = state[7]
        self._update_error_dist()
        self.done = self.error_dist < self.success_threshold

    def __str__(self):
        """String representation of the world state."""
        est_str = "PF Not Initialized"
        if self.estimated_landmark.estimated_location:
            est_loc = self.estimated_landmark.estimated_location
            est_vel_str = ""
            if self.estimated_landmark.estimated_velocity:
                est_vel = self.estimated_landmark.estimated_velocity
                est_vel_str = f", Vel:(vx:{est_vel.x:.2f}, vy:{est_vel.y:.2f})"
            est_str = f"Est Lmk: Pos:(x:{est_loc.x:.2f}, y:{est_loc.y:.2f}){est_vel_str}"

        true_lmk_str = f"True Lmk: {self.true_landmark}"
        agent_str = f"Agent: {self.agent}"

        return (
            f"--- World State ---\n"
            f" Reward: {self.reward:.4f}, Done: {self.done}\n"
            f" Range: {self.current_range:.2f}, Error (2D): {self.error_dist:.2f}\n"
            f" {agent_str}\n"
            f" {true_lmk_str}\n"
            f" {est_str}\n"
            f"-------------------"
        )


if __name__ == "__main__":
    from configs import DefaultConfig

    cfg = DefaultConfig()
    world_cfg = cfg.world
    pf_cfg = cfg.particle_filter

    print("Initializing world with default configuration...")
    world = World(world_config=world_cfg, pf_config=pf_cfg)
    print("Initial World State:")
    print(world)

    print("\nSimulating 10 steps with action (vx=1, vy=1)...")
    action = Velocity(1, 1, 0)
    for i in range(10):
        world.step(action, training=True)
        print(f"\nAfter Step {i+1}:")
        print(world)

    print("\nSimulation finished.")
