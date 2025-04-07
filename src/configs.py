from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional, Tuple


class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(
        8, description="State dimension (agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, current_range)")
    action_dim: int = Field(2, description="Action dimension (vx, vy)")
    action_scale: float = Field(
        4.0, description="Scale actions to reasonable velocity range")
    hidden_dim: int = Field(256, description="Hidden layer dimension")
    log_std_min: int = Field(-20,
                             description="Minimum log std for action distribution")
    log_std_max: int = Field(
        2, description="Maximum log std for action distribution")
    lr: float = Field(3e-4, description="Learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.005, description="Target network update rate")
    alpha: float = Field(0.2, description="Temperature parameter")
    auto_tune_alpha: bool = Field(
        True, description="Whether to auto-tune the alpha parameter")


class ReplayBufferConfig(BaseModel):
    """Configuration for the replay buffer"""
    capacity: int = Field(
        1000000, description="Maximum capacity of replay buffer")


class TrainingConfig(BaseModel):
    """Configuration for training"""
    num_episodes: int = Field(1000, description="Number of episodes to train")
    max_steps: int = Field(500, description="Maximum steps per episode")
    batch_size: int = Field(256, description="Batch size for training")
    save_interval: int = Field(100, description="Interval for saving models")
    models_dir: str = Field(
        "sac_models", description="Directory for saving models")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    num_episodes: int = Field(
        1, description="Number of episodes for evaluation")
    max_steps: int = Field(
        300, description="Maximum steps per evaluation episode")
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
    """Defines ranges for random initialization"""
    x_range: Tuple[float, float] = Field(
        (-50.0, 50.0), description="Min/Max X range for randomization")
    y_range: Tuple[float, float] = Field(
        (-50.0, 50.0), description="Min/Max Y range for randomization")
    depth_range: Tuple[float, float] = Field(
        (0, 50.0), description="Min/Max Depth range for randomization")


class WorldConfig(BaseModel):
    """Configuration for the world"""
    dt: float = Field(1.0, description="Time step")
    success_threshold: float = Field(
        3.0, description="Distance threshold for successful landmark detection and early termination")

    agent_initial_location: Position = Field(
        default_factory=Position, description="Initial agent position (used if randomization is false)")
    landmark_initial_location: Position = Field(default_factory=lambda: Position(
        x=42, y=42, depth=42), description="Initial landmark position (used if randomization is false)")

    agent_initial_velocity: Velocity = Field(
        default_factory=Velocity, description="Initial agent velocity")
    landmark_initial_velocity: Velocity = Field(
        default_factory=Velocity, description="Initial landmark velocity")

    randomize_agent_initial_location: bool = Field(
        True, description="Randomize agent initial location?")
    randomize_landmark_initial_location: bool = Field(
        True, description="Randomize landmark initial location?")

    agent_randomization_ranges: RandomizationRange = Field(default_factory=lambda: RandomizationRange(
        depth_range=(0.0, 0.0)), description="Ranges for agent location randomization")
    landmark_randomization_ranges: RandomizationRange = Field(
        default_factory=RandomizationRange, description="Ranges for landmark location randomization")

    step_penalty: float = Field(
        0.05, description="Penalty subtracted each step")
    success_bonus: float = Field(
        100.0, description="Bonus reward upon reaching success threshold")
    out_of_range_penalty: float = Field(
        100.0, description="Penalty if range exceeds threshold")
    out_of_range_threshold: float = Field(
        100.0, description="Range threshold for out_of_range_penalty")


class ParticleFilterConfig(BaseModel):
    """Configuration for the particle filter"""
    num_particles: int = Field(1000, description="Number of particles")
    initial_range_stddev: float = Field(
        0.02, description="Standard deviation for initial particle spread")
    initial_velocity_guess: float = Field(
        0.1, description="Initial velocity guess for particles")
    estimation_method: Literal["range", "area"] = Field(
        "range", description="Method for estimation (range or area)")
    max_particle_range: float = Field(
        250.0, description="Maximum range for particles (used in area method or init)")
    process_noise_pos: float = Field(
        0.02, description="Process noise for position")
    process_noise_orient: float = Field(
        0.2, description="Process noise for orientation")
    process_noise_vel: float = Field(
        0.02, description="Process noise for velocity")
    measurement_noise_stddev: float = Field(
        5.0, description="Standard deviation for measurement noise")
    resampling_method: int = Field(2, description="Method for resampling")
    pf_eval_max_mean_range_error_factor: float = Field(
        0.1, description="Factor of max_particle_range used as threshold for PF quality check")
    pf_eval_dispersion_threshold: float = Field(
        5.0, description="Dispersion threshold for PF quality check")


class VisualizationConfig(BaseModel):
    """Configuration for visualization"""
    save_dir: str = Field(
        "world_snapshots", description="Directory for saving visualizations")
    figure_size: tuple = Field(
        (10, 8), description="Figure size for visualizations")
    max_trajectory_points: int = Field(
        100, description="Max trajectory points to display")
    gif_frame_duration: float = Field(
        0.2, description="Duration of each frame in generated GIFs")
    delete_frames_after_gif: bool = Field(
        True, description="Delete individual PNG frames after creating GIF")


class DefaultConfig(BaseModel):
    """Default configuration for the entire application"""
    sac: SACConfig = Field(default_factory=SACConfig,
                           description="SAC agent configuration")
    replay_buffer: ReplayBufferConfig = Field(
        default_factory=ReplayBufferConfig, description="Replay buffer configuration")
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration")
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration")
    world: WorldConfig = Field(
        default_factory=WorldConfig, description="World configuration")
    particle_filter: ParticleFilterConfig = Field(
        default_factory=ParticleFilterConfig, description="Particle filter configuration")
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig, description="Visualization configuration")


default_config = DefaultConfig()

vast_config = DefaultConfig()
vast_config.training.num_episodes = 50000
vast_config.training.max_steps = 200
vast_config.training.save_interval = 5000
vast_config.particle_filter.num_particles = 5000
vast_config.world.randomize_agent_initial_location = True
vast_config.world.randomize_landmark_initial_location = True

CONFIGS: Dict[str, DefaultConfig] = {
    "default": default_config,
    "vast": vast_config,
}
