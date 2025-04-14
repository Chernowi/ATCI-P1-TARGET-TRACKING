# --- START OF FILE configs.py ---

from typing import Dict, Literal, Tuple, List
from pydantic import BaseModel, Field


class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(8, description="State dimension (agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, current_range)")
    action_dim: int = Field(2, description="Action dimension (vx, vy)")
    action_scale: float = Field(1, description="Scale actions to reasonable velocity range")
    hidden_dims: List[int] = Field([128, 128], description="List of hidden layer dimensions for MLP part")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    lr: float = Field(5e-4, description="Learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.01, description="Target network update rate")
    alpha: float = Field(0.2, description="Temperature parameter (Initial value if auto-tuning)") # Default value for SAC
    auto_tune_alpha: bool = Field(True, description="Whether to auto-tune the alpha parameter")
    use_rnn: bool = Field(True, description="Whether to use RNN layers in Actor/Critic (NOTE: Set to False for standard SAC/T-SAC MLP/Transformer usage)")
    rnn_type: Literal['lstm', 'gru'] = Field('lstm', description="Type of RNN cell (Only used if use_rnn is True)")
    rnn_hidden_size: int = Field(128, description="Hidden size of RNN layers (Only used if use_rnn is True)")
    rnn_num_layers: int = Field(1, description="Number of RNN layers (Only used if use_rnn is True)")
    sequence_length: int = Field(10, description="Length of sequences for RNN training (Only used if use_rnn is True)")


class TSACConfig(SACConfig):
    """Configuration for the Transformer-SAC agent, inheriting from SACConfig"""
    # Override or add Transformer/N-step specific parameters
    use_rnn: bool = Field(False, description="Ensure RNN is disabled for T-SAC's Transformer Critic") # Explicitly disable RNN
    sequence_length: int = Field(10, description="Action chunk length (N) for N-step returns and Transformer input") # N in paper
    embedding_dim: int = Field(128, description="Embedding dimension for states and actions in Transformer Critic")
    transformer_n_layers: int = Field(2, description="Number of Transformer encoder layers in Critic")
    transformer_n_heads: int = Field(4, description="Number of attention heads in Transformer Critic")
    transformer_hidden_dim: int = Field(512, description="Hidden dimension within Transformer layers (FeedForward network)")
    use_layer_norm_actor: bool = Field(True, description="Apply Layer Normalization in Actor MLP layers")
    # Alpha defaults inherited, can be overridden if needed for T-SAC specifically
    alpha: float = Field(0.1, description="Temperature parameter (Initial value if auto-tuning)") # T-SAC might need different tuning


class PPOConfig(BaseModel):
    """Configuration for the PPO agent"""
    state_dim: int = Field(8, description="State dimension (agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, current_range)")
    action_dim: int = Field(2, description="Action dimension (vx, vy)")
    action_scale: float = Field(1, description="Scale actions to reasonable velocity range")
    hidden_dim: int = Field(256, description="Hidden layer dimension")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    actor_lr: float = Field(5e-6, description="Actor learning rate")
    critic_lr: float = Field(1e-3, description="Critic learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    policy_clip: float = Field(0.05, description="PPO clipping parameter")
    n_epochs: int = Field(3, description="Number of optimization epochs per update")
    entropy_coef: float = Field(0.015, description="Entropy coefficient for exploration")
    value_coef: float = Field(0.5, description="Value loss coefficient")
    batch_size: int = Field(1024, description="Batch size for training")
    steps_per_update: int = Field(8192, description="Environment steps between PPO updates")


class ReplayBufferConfig(BaseModel):
    """Configuration for the replay buffer"""
    capacity: int = Field(100000, description="Maximum capacity of replay buffer") # Increased capacity
    gamma: float = Field(0.99, description="Discount factor for returns")


class TrainingConfig(BaseModel):
    """Configuration for training"""
    num_episodes: int = Field(5000, description="Number of episodes to train")
    max_steps: int = Field(300, description="Maximum steps per episode")
    batch_size: int = Field(512, description="Batch size for training") # Increased batch size
    save_interval: int = Field(100, description="Interval (in episodes) for saving models")
    log_frequency: int = Field(1, description="Frequency (in episodes) for logging to TensorBoard")
    models_dir: str = Field("sac_models", description="Directory for saving models") # Keep same dir for now
    learning_starts: int = Field(8000, description="Number of steps to collect before starting training updates")
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
    x_range: Tuple[float, float] = Field((-100.0, 100.0), description="Min/Max X range for randomization")
    y_range: Tuple[float, float] = Field((-100.0, 100.0), description="Min/Max Y range for randomization")
    depth_range: Tuple[float, float] = Field((0, 300), description="Min/Max Depth range for randomization")

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
    history_size: int = Field(10, description="Number of measurements to keep in history")
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
    range_measurement_distance_factor: float = 0.01  # Noise increases by 1% of distance (Adjusted from 5%)

    # Reward function parameters
    reward_scale: float = 0.005
    distance_threshold: float = 50.0
    error_threshold: float = 5.0
    min_safe_distance: float = 2.0

    # Landmark estimator config
    estimator_config: ParticleFilterConfig | LeastSquaresConfig = Field(default_factory=LeastSquaresConfig, description="Configuration for the landmark estimator")

class DefaultConfig(BaseModel):
    """Default configuration for the entire application"""
    sac: SACConfig = Field(default_factory=SACConfig, description="SAC agent configuration")
    tsac: TSACConfig = Field(default_factory=TSACConfig, description="T-SAC agent configuration") # Add T-SAC config
    ppo: PPOConfig = Field(default_factory=PPOConfig, description="PPO agent configuration")
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig, description="Replay buffer configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    world: WorldConfig = Field(default_factory=WorldConfig, description="World configuration")
    particle_filter: ParticleFilterConfig = Field(default_factory=ParticleFilterConfig, description="Particle filter configuration")
    least_squares: LeastSquaresConfig = Field(default_factory=LeastSquaresConfig, description="Least Squares estimator configuration")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization configuration")
    cuda_device: str = Field("cuda:0", description="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    algorithm: str = Field("sac", description="RL algorithm to use ('sac', 'ppo', or 'tsac')") # Add tsac option


default_config = DefaultConfig()

# --- Standard SAC Config (Example) ---
sac_default_config = DefaultConfig()
sac_default_config.algorithm = "sac"
# Keep sac_default_config.sac as default SACConfig


# --- T-SAC Default Config (Example) ---
tsac_default_config = DefaultConfig()
tsac_default_config.algorithm = "tsac"
# Modify tsac_default_config.tsac as needed
tsac_default_config.tsac.sequence_length = 8 # Example sequence length
tsac_default_config.tsac.embedding_dim = 128
tsac_default_config.tsac.transformer_n_layers = 2
tsac_default_config.tsac.transformer_n_heads = 4
tsac_default_config.tsac.transformer_hidden_dim = 256
tsac_default_config.tsac.lr = 1e-4 # T-SAC might need different LR
tsac_default_config.tsac.alpha = 0.1
# Ensure training config is suitable
tsac_default_config.training.learning_starts = 2000 # May need more samples for sequences
tsac_default_config.training.batch_size = 128 # May need smaller batch due to sequence memory


# --- Vast Config (Example for large scale training, maybe T-SAC) ---
vast_config = DefaultConfig()
vast_config.algorithm = "tsac" # Example: Use T-SAC for vast config
vast_config.training.num_episodes = 50000
vast_config.training.max_steps = 200
vast_config.training.save_interval = 5000
vast_config.training.batch_size = 128 # Adjust batch size
vast_config.training.learning_starts = 5000
vast_config.particle_filter.num_particles = 5000 # If using PF estimator
vast_config.world.randomize_agent_initial_location = True
vast_config.world.randomize_landmark_initial_location = True
vast_config.world.randomize_landmark_initial_velocity = True
# Configure T-SAC specifically for vast
vast_config.tsac.hidden_dims = [512, 512, 256]
vast_config.tsac.sequence_length = 16
vast_config.tsac.embedding_dim = 256
vast_config.tsac.transformer_n_layers = 4
vast_config.tsac.transformer_n_heads = 8
vast_config.tsac.transformer_hidden_dim = 1024
vast_config.tsac.lr = 5e-5 # Adjust LR
vast_config.tsac.alpha = 0.05 # Adjust alpha


CONFIGS: Dict[str, DefaultConfig] = {
    "default": default_config, # Default now points to a config potentially using SAC
    "sac_default": sac_default_config, # Explicitly SAC
    "tsac_default": tsac_default_config, # Explicitly T-SAC
    "vast": vast_config, # Vast example using T-SAC
}
# Ensure 'default' config uses a valid algorithm setting if needed
if default_config.algorithm not in ["sac", "ppo", "tsac"]:
    default_config.algorithm = "sac" # Fallback for default