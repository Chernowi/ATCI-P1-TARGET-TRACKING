from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional


class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(8, description="State dimension (agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, current_range)")
    action_dim: int = Field(2, description="Action dimension (vx, vy)")
    action_scale: float = Field(2.0, description="Scale actions to reasonable velocity range")
    hidden_dim: int = Field(256, description="Hidden layer dimension")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    lr: float = Field(3e-4, description="Learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.005, description="Target network update rate")
    alpha: float = Field(0.2, description="Temperature parameter")
    auto_tune_alpha: bool = Field(True, description="Whether to auto-tune the alpha parameter")


class ReplayBufferConfig(BaseModel):
    """Configuration for the replay buffer"""
    capacity: int = Field(1000000, description="Maximum capacity of replay buffer")


class TrainingConfig(BaseModel):
    """Configuration for training"""
    num_episodes: int = Field(1000, description="Number of episodes to train")
    max_steps: int = Field(100, description="Maximum steps per episode")
    batch_size: int = Field(256, description="Batch size for training")
    replay_buffer_size: int = Field(1000000, description="Replay buffer size")
    save_interval: int = Field(100, description="Interval for saving models")
    models_dir: str = Field("sac_models", description="Directory for saving models")
    success_threshold: float = Field(2.0, description="Threshold for successful landmark detection")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    num_episodes: int = Field(5, description="Number of episodes for evaluation")
    max_steps: int = Field(100, description="Maximum steps per evaluation episode")
    render: bool = Field(True, description="Whether to render the evaluation")
    success_threshold: float = Field(2.0, description="Threshold for successful landmark detection")


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


class WorldConfig(BaseModel):
    """Configuration for the world"""
    dt: float = Field(1.0, description="Time step")
    success_threshold: float = Field(2.0, description="Distance threshold for successful landmark detection")
    agent_initial_location: Position = Field(default_factory=Position, description="Initial agent position")
    landmark_initial_location: Position = Field(default_factory=lambda: Position(x=42, y=42, depth=42), description="Initial landmark position")
    agent_initial_velocity: Velocity = Field(default_factory=Velocity, description="Initial agent velocity")
    landmark_initial_velocity: Velocity = Field(default_factory=Velocity, description="Initial landmark velocity")


class ParticleFilterConfig(BaseModel):
    """Configuration for the particle filter"""
    num_particles: int = Field(1000, description="Number of particles")
    initial_range_stddev: float = Field(0.02, description="Standard deviation for initial particle spread")
    initial_velocity_guess: float = Field(0.1, description="Initial velocity guess for particles")
    estimation_method: Literal["range", "area"] = Field("range", description="Method for estimation (range or area)")
    max_particle_range: float = Field(250.0, description="Maximum range for particles")
    process_noise_pos: float = Field(0.02, description="Process noise for position")
    process_noise_orient: float = Field(0.2, description="Process noise for orientation")
    process_noise_vel: float = Field(0.02, description="Process noise for velocity")
    measurement_noise_stddev: float = Field(5.0, description="Standard deviation for measurement noise")
    resampling_method: int = Field(2, description="Method for resampling")


class VisualizationConfig(BaseModel):
    """Configuration for visualization"""
    save_dir: str = Field("world_snapshots", description="Directory for saving visualizations")
    figure_size: tuple = Field((10, 8), description="Figure size for visualizations")


class DefaultConfig(BaseModel):
    """Default configuration for the entire application"""
    sac: SACConfig = Field(default_factory=SACConfig, description="SAC agent configuration")
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig, description="Replay buffer configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    world: WorldConfig = Field(default_factory=WorldConfig, description="World configuration")
    particle_filter: ParticleFilterConfig = Field(default_factory=ParticleFilterConfig, description="Particle filter configuration")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization configuration")


# Create a default configuration instance
default_config = DefaultConfig()