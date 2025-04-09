from typing import Dict, Literal, Tuple
from pydantic import BaseModel, Field


class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(8, description="State dimension (agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, current_range)")
    action_dim: int = Field(2, description="Action dimension (vx, vy)")
    action_scale: float = Field(1, description="Scale actions to reasonable velocity range")
    hidden_dim: int = Field(256, description="Hidden layer dimension")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    lr: float = Field(3e-4, description="Learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.005, description="Target network update rate")
    alpha: float = Field(0.005, description="Temperature parameter")
    auto_tune_alpha: bool = Field(True, description="Whether to auto-tune the alpha parameter")


class PPOConfig(BaseModel):
    """Configuration for the PPO agent"""
    state_dim: int = Field(8, description="State dimension (agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, current_range)")
    action_dim: int = Field(2, description="Action dimension (vx, vy)")
    action_scale: float = Field(1, description="Scale actions to reasonable velocity range")
    hidden_dim: int = Field(256, description="Hidden layer dimension")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    actor_lr: float = Field(1.5e-4, description="Actor learning rate")
    critic_lr: float = Field(1e-3, description="Critic learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    policy_clip: float = Field(0.15, description="PPO clipping parameter")
    n_epochs: int = Field(3, description="Number of optimization epochs per update")
    entropy_coef: float = Field(0.01, description="Entropy coefficient for exploration")
    value_coef: float = Field(0.5, description="Value loss coefficient")
    batch_size: int = Field(64, description="Batch size for training")
    steps_per_update: int = Field(8192, description="Environment steps between PPO updates")


class ReplayBufferConfig(BaseModel):
    """Configuration for the replay buffer"""
    capacity: int = Field(100000, description="Maximum capacity of replay buffer") # Increased capacity
    gamma: float = Field(0.99, description="Discount factor for returns")


class TrainingConfig(BaseModel):
    """Configuration for training"""
    num_episodes: int = Field(10000, description="Number of episodes to train")
    max_steps: int = Field(250, description="Maximum steps per episode")
    batch_size: int = Field(256, description="Batch size for training") # Increased batch size
    save_interval: int = Field(100, description="Interval (in episodes) for saving models")
    log_frequency: int = Field(10, description="Frequency (in episodes) for logging to TensorBoard")
    models_dir: str = Field("sac_models", description="Directory for saving models")
    learning_starts: int = Field(1000, description="Number of steps to collect before starting training updates")
    train_freq: int = Field(1, description="Update the policy every n environment steps")
    gradient_steps: int = Field(1, description="How many gradient steps to perform when training frequency is met")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    num_episodes: int = Field(1, description="Number of episodes for evaluation")
    max_steps: int = Field(300, description="Maximum steps per evaluation episode")
    render: bool = Field(True, description="Whether to render the evaluation")


class Position(BaseModel):
    """Position configuration"""
    x: float = 0
    y: float = 0
    depth: float = 0


class Velocity(BaseModel):
    """Velocity configuration"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class RandomizationRange(BaseModel):
    """Defines ranges for random initialization of position"""
    x_range: Tuple[float, float] = Field((-50.0, 50.0), description="Min/Max X range for randomization")
    y_range: Tuple[float, float] = Field((-50.0, 50.0), description="Min/Max Y range for randomization")
    depth_range: Tuple[float, float] = Field((-50.0, 50.0), description="Min/Max Depth range for randomization")

class VelocityRandomizationRange(BaseModel):
    """Defines ranges for random initialization of velocity"""
    vx_range: Tuple[float, float] = Field((-0.5, 0.5), description="Min/Max Vx range for randomization")
    vy_range: Tuple[float, float] = Field((-0.5, 0.5), description="Min/Max Vy range for randomization")
    vz_range: Tuple[float, float] = Field((-0.1, 0.1), description="Min/Max Vz range for randomization")


class ParticleFilterConfig(BaseModel):
    """Configuration for the particle filter"""
    num_particles: int = Field(1000, description="Number of particles")
    initial_range_stddev: float = Field(0.02, description="Standard deviation for initial particle spread")
    initial_velocity_guess: float = Field(0.1, description="Initial velocity guess for particles")
    estimation_method: Literal["range", "area"] = Field("range", description="Method for estimation (range or area)")
    max_particle_range: float = Field(250.0, description="Maximum range for particles (used in area method or init)")
    process_noise_pos: float = Field(0.02, description="Process noise for position")
    process_noise_orient: float = Field(0.2, description="Process noise for orientation")
    process_noise_vel: float = Field(0.02, description="Process noise for velocity")
    measurement_noise_stddev: float = Field(5.0, description="Standard deviation for measurement noise")
    resampling_method: int = Field(2, description="Method for resampling")
    pf_eval_max_mean_range_error_factor: float = Field(0.1, description="Factor of max_particle_range used as threshold for PF quality check")
    pf_eval_dispersion_threshold: float = Field(5.0, description="Dispersion threshold for PF quality check")


class LeastSquaresConfig(BaseModel):
    """Configuration for the Least Squares estimator"""
    history_size: int = Field(30, description="Number of measurements to keep in history")
    min_points_required: int = Field(3, description="Minimum number of points required for estimation")
    position_buffer_size: int = Field(5, description="Number of position estimates to keep for velocity calculation")
    velocity_smoothing: int = Field(3, description="Number of position points to use for velocity smoothing")
    min_observer_movement: float = Field(0.5, description="Minimum movement required between measurement points")


class VisualizationConfig(BaseModel):
    """Configuration for visualization"""
    save_dir: str = Field("world_snapshots", description="Directory for saving visualizations")
    figure_size: tuple = Field((10, 8), description="Figure size for visualizations")
    max_trajectory_points: int = Field(100, description="Max trajectory points to display")
    gif_frame_duration: float = Field(0.2, description="Duration of each frame in generated GIFs")
    delete_frames_after_gif: bool = Field(True, description="Delete individual PNG frames after creating GIF")


class WorldConfig(BaseModel):
    """Configuration for the world"""
    dt: float = Field(1.0, description="Time step")
    success_threshold: float = Field(0.5, description="Distance threshold for successful landmark detection and early termination")

    agent_initial_location: Position = Field(default_factory=Position, description="Initial agent position (used if randomization is false)")
    landmark_initial_location: Position = Field(default_factory=lambda: Position(x=42, y=42, depth=42), description="Initial landmark position (used if randomization is false)")

    agent_initial_velocity: Velocity = Field(default_factory=Velocity, description="Initial agent velocity")
    landmark_initial_velocity: Velocity = Field(default_factory=Velocity, description="Initial landmark velocity (used if randomization is false)")

    randomize_agent_initial_location: bool = Field(True, description="Randomize agent initial location?")
    randomize_landmark_initial_location: bool = Field(True, description="Randomize landmark initial location?")
    randomize_landmark_initial_velocity: bool = Field(False, description="Randomize landmark initial velocity?") # Default to False

    agent_randomization_ranges: RandomizationRange = Field(default_factory=lambda: RandomizationRange(depth_range=(0.0, 0.0)), description="Ranges for agent location randomization")
    landmark_randomization_ranges: RandomizationRange = Field(default_factory=RandomizationRange, description="Ranges for landmark location randomization")
    landmark_velocity_randomization_ranges: VelocityRandomizationRange = Field(default_factory=VelocityRandomizationRange, description="Ranges for landmark velocity randomization")

    step_penalty: float = Field(0.1, description="Penalty subtracted each step")
    success_bonus: float = Field(100.0, description="Bonus reward upon reaching success threshold")
    out_of_range_penalty: float = Field(100.0, description="Penalty if range exceeds threshold")
    out_of_range_threshold: float = Field(100.0, description="Range threshold for out_of_range_penalty")
    range_measurement_base_noise: float = 0.1  # Base noise level in meters
    range_measurement_distance_factor: float = 0.05  # Noise increases by 5% of distance

    # Reward function parameters
    reward_scale: float = 5.0
    distance_threshold: float = 10.0
    error_threshold: float = 5.0
    min_safe_distance: float = 2.0

    # Landmark estimator config
    estimator_config: ParticleFilterConfig | LeastSquaresConfig = Field(default_factory=LeastSquaresConfig, description="Configuration for the landmark estimator")

class DefaultConfig(BaseModel):
    """Default configuration for the entire application"""
    sac: SACConfig = Field(default_factory=SACConfig, description="SAC agent configuration")
    ppo: PPOConfig = Field(default_factory=PPOConfig, description="PPO agent configuration")
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig, description="Replay buffer configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    world: WorldConfig = Field(default_factory=WorldConfig, description="World configuration")
    particle_filter: ParticleFilterConfig = Field(default_factory=ParticleFilterConfig, description="Particle filter configuration")
    least_squares: LeastSquaresConfig = Field(default_factory=LeastSquaresConfig, description="Least Squares estimator configuration")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization configuration")
    cuda_device: str = Field("cpu", description="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    algorithm: str = Field("sac", description="RL algorithm to use ('sac' or 'ppo')")

default_config = DefaultConfig()

vast_config = DefaultConfig()
vast_config.training.num_episodes = 50000
vast_config.training.max_steps = 200
vast_config.training.save_interval = 5000
vast_config.particle_filter.num_particles = 5000
vast_config.world.randomize_agent_initial_location = True
vast_config.world.randomize_landmark_initial_location = True
vast_config.world.randomize_landmark_initial_velocity = True

CONFIGS: Dict[str, DefaultConfig] = {
    "default": default_config,
    "vast": vast_config,
}
