from typing import Dict, Literal, Tuple, List
from pydantic import BaseModel, Field
import math

# Core dimensions
CORE_STATE_DIM = 8 # agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, current_range
CORE_ACTION_DIM = 1 # yaw_change_normalized
TRAJECTORY_REWARD_DIM = 1

class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the basic state tuple within the trajectory")
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)")
    # NOTE: action_scale in WorldConfig (max_yaw_change) is the primary scaling factor now.
    # This might be redundant or used differently in the agent implementation. Assuming agent outputs [-1, 1].
    # action_scale: float = Field(math.pi / 4, description="Maximum magnitude of yaw change action (scales the [-1, 1] output)")
    hidden_dims: List[int] = Field([64, 64], description="List of hidden layer dimensions for MLP part")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    lr: float = Field(5e-5, description="Learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.01, description="Target network update rate")
    alpha: float = Field(0.2, description="Temperature parameter (Initial value if auto-tuning)")
    auto_tune_alpha: bool = Field(True, description="Whether to auto-tune the alpha parameter")
    use_rnn: bool = Field(False, description="Whether to use RNN layers in Actor/Critic (Recommended for trajectory state)")
    rnn_type: Literal['lstm', 'gru'] = Field('lstm', description="Type of RNN cell (Only used if use_rnn is True)")
    rnn_hidden_size: int = Field(128, description="Hidden size of RNN layers (Only used if use_rnn is True)")
    rnn_num_layers: int = Field(1, description="Number of RNN layers (Only used if use_rnn is True)")


class TSACConfig(SACConfig):
    """Configuration for the Transformer-SAC agent, inheriting from SACConfig"""
    use_rnn: bool = Field(False, description="Ensure RNN is disabled for T-SAC's Transformer Critic")
    # action_scale: float = Field(math.pi / 4, description="Maximum magnitude of yaw change action") # Inherited/Overridden if needed
    embedding_dim: int = Field(128, description="Embedding dimension for states and actions in Transformer Critic")
    transformer_n_layers: int = Field(2, description="Number of Transformer encoder layers in Critic")
    transformer_n_heads: int = Field(4, description="Number of attention heads in Transformer Critic")
    transformer_hidden_dim: int = Field(512, description="Hidden dimension within Transformer layers (FeedForward network)")
    use_layer_norm_actor: bool = Field(True, description="Apply Layer Normalization in Actor MLP layers")
    alpha: float = Field(0.1, description="Temperature parameter (Initial value if auto-tuning)") # Override SAC alpha

class PPOConfig(BaseModel):
    """Configuration for the PPO agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the basic state tuple (PPO uses the last state of trajectory)")
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)")
    # action_scale: float = Field(math.pi / 4, description="Maximum magnitude of yaw change action") # See note in SACConfig
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
    capacity: int = Field(100000, description="Maximum capacity of replay buffer (stores full trajectories)")
    gamma: float = Field(0.99, description="Discount factor for returns")

class TrainingConfig(BaseModel):
    """Configuration for training"""
    num_episodes: int = Field(30000, description="Number of episodes to train")
    max_steps: int = Field(300, description="Maximum steps per episode")
    batch_size: int = Field(512, description="Batch size for training (Number of trajectories sampled)")
    save_interval: int = Field(100, description="Interval (in episodes) for saving models")
    log_frequency: int = Field(1, description="Frequency (in episodes) for logging to TensorBoard")
    models_dir: str = Field("models/sac/", description="Directory for saving models")
    learning_starts: int = Field(8000, description="Number of steps to collect before starting training updates")
    train_freq: int = Field(30, description="Update the policy every n environment steps")
    gradient_steps: int = Field(20, description="How many gradient steps to perform when training frequency is met")

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
    # --- Basic World Dynamics ---
    dt: float = Field(1.0, description="Time step")
    agent_speed: float = Field(2.5, description="Constant speed of the agent")
    yaw_angle_range: Tuple[float, float] = Field((-math.pi / 4, math.pi / 4), description="Range of possible yaw angle changes per step [-max_change, max_change]")

    # --- Initial Conditions & Randomization ---
    agent_initial_location: Position = Field(default_factory=Position, description="Initial agent position (used if randomization is false)")
    landmark_initial_location: Position = Field(default_factory=lambda: Position(x=42, y=42, depth=42), description="Initial landmark position (used if randomization is false)")
    landmark_initial_velocity: Velocity = Field(default_factory=Velocity, description="Initial landmark velocity (used if randomization is false)")

    randomize_agent_initial_location: bool = Field(True, description="Randomize agent initial location?")
    randomize_landmark_initial_location: bool = Field(True, description="Randomize landmark initial location?")
    randomize_landmark_initial_velocity: bool = Field(False, description="Randomize landmark initial velocity?")

    agent_randomization_ranges: RandomizationRange = Field(default_factory=lambda: RandomizationRange(depth_range=(0.0, 0.0)), description="Ranges for agent location randomization")
    landmark_randomization_ranges: RandomizationRange = Field(default_factory=RandomizationRange, description="Ranges for landmark location randomization")
    landmark_velocity_randomization_ranges: VelocityRandomizationRange = Field(default_factory=VelocityRandomizationRange, description="Ranges for landmark velocity randomization")

    # --- State Representation ---
    trajectory_length: int = Field(10, description="Number of steps (N) included in the trajectory state")
    trajectory_feature_dim: int = Field(CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM, description="Dimension of features per step in trajectory state (basic_state + prev_action + prev_reward)") # 8+1+1=10

    # --- Observations & Noise ---
    range_measurement_base_noise: float = Field(0.01, description="Base standard deviation of range measurement noise")
    range_measurement_distance_factor: float = Field(0.1, description="Factor by which range noise std dev increases with distance")

    # --- Termination Conditions ---
    success_threshold: float = Field(0.5, description="Estimation error (2D distance) below which the episode is considered successful")
    collision_threshold: float = Field(0.5, description="Agent-Landmark true distance below which a collision occurs (terminates episode)") # Adjusted default slightly

    # --- Reward Function Parameters (Aligned with tracking.py logic) ---
    # -- Estimation Error Reward --
    reward_error_threshold: float = Field(1.0, description="Estimation error threshold for bonus reward") # Similar to old error_threshold
    low_error_bonus: float = Field(1.0, description="Reward bonus when estimation error is below reward_error_threshold")
    high_error_penalty_factor: float = Field(0.1, description="Penalty multiplier for estimation error exceeding threshold (penalty = factor * (error - threshold))") # Needs tuning
    uninitialized_penalty: float = Field(1.0, description="Penalty applied if the estimator hasn't produced a valid estimate yet")

    # -- Agent-Landmark Distance Reward (Based on True Distance) --
    reward_distance_threshold: float = Field(15.0, description="True distance threshold for close distance bonus") # Corresponds to rew_dis_th
    close_distance_bonus: float = Field(1.0, description="Reward bonus when true distance is below reward_distance_threshold")
    distance_reward_scale: float = Field(0.0001, description="Scaling factor for distance reward shaping (reward = scale * (max_dist - current_dist))") # Needs tuning
    max_distance_for_reward: float = Field(50.0, description="Maximum true distance up to which distance shaping reward is applied") # Corresponds to 0.7 * scale factor in tracking.py? Needs tuning.

    # -- Penalties & Bonuses --
    max_observable_range: float = Field(100.0, description="Maximum true distance considered 'in range' for penalty calculation") # Corresponds to set_max_range
    out_of_range_penalty: float = Field(0.1, description="Penalty applied when true distance exceeds max_observable_range")
    landmark_collision_penalty: float = Field(1.0, description="Penalty applied when true distance is below collision_threshold")
    success_bonus: float = Field(100.0, description="Bonus reward added upon reaching success_threshold")
    # step_penalty: float = Field(0.1, description="Penalty subtracted each step") # Optional: Keep if desired, but tracking.py didn't have explicit step penalty

    # --- Old Reward Parameters (Commented out/Removed) ---
    # reward_scale: float = 0.005 # Replaced by new factors
    # distance_threshold: float = 50.0 # Replaced by reward_distance_threshold and max_distance_for_reward
    # error_threshold: float = 1.0 # Replaced by reward_error_threshold
    # min_safe_distance: float = 2.0 # Concept replaced by collision_threshold penalty
    # out_of_range_penalty: float = 100.0 # Replaced by new out_of_range_penalty (usually smaller)
    # out_of_range_threshold: float = 100.0 # Replaced by max_observable_range

    # --- Landmark Estimator ---
    estimator_config: ParticleFilterConfig | LeastSquaresConfig = Field(default_factory=LeastSquaresConfig, description="Configuration for the landmark estimator")


class DefaultConfig(BaseModel):
    """Default configuration for the entire application"""
    sac: SACConfig = Field(default_factory=SACConfig, description="SAC agent configuration")
    tsac: TSACConfig = Field(default_factory=TSACConfig, description="T-SAC agent configuration")
    ppo: PPOConfig = Field(default_factory=PPOConfig, description="PPO agent configuration")
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig, description="Replay buffer configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    world: WorldConfig = Field(default_factory=WorldConfig, description="World configuration")
    particle_filter: ParticleFilterConfig = Field(default_factory=ParticleFilterConfig, description="Particle filter configuration")
    least_squares: LeastSquaresConfig = Field(default_factory=LeastSquaresConfig, description="Least Squares estimator configuration")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization configuration")
    cuda_device: str = Field("cuda:0", description="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    algorithm: str = Field("sac", description="RL algorithm to use ('sac', 'ppo', or 'tsac')")

    # Using model_post_init for Pydantic V2 compatibility
    def model_post_init(self, __context):
        self._sync_sequence_lengths()
        self._resolve_estimator_config()

    def _sync_sequence_lengths(self):
        # Sequence length is primarily determined by world.trajectory_length
        # Agent configs don't need explicit sequence_length if they use world's value
        pass # Placeholder if any future sync is needed

    def _resolve_estimator_config(self):
        # Ensure the estimator config in WorldConfig points to the correct instance
        # based on some logic (e.g., a separate setting or default)
        # Example: Defaulting to LeastSquares, but could be based on another field
        if isinstance(self.world.estimator_config, type): # If it's just the class type
            if self.world.estimator_config == ParticleFilterConfig:
                self.world.estimator_config = self.particle_filter
            elif self.world.estimator_config == LeastSquaresConfig:
                self.world.estimator_config = self.least_squares
            else: # Default case
                 self.world.estimator_config = self.least_squares
        elif not isinstance(self.world.estimator_config, (ParticleFilterConfig, LeastSquaresConfig)):
             # If it's somehow invalid, default it
             print("Warning: Invalid estimator_config type in WorldConfig, defaulting to LeastSquares.")
             self.world.estimator_config = self.least_squares


default_config = DefaultConfig()

# --- T-SAC Default Config (Example) ---
tsac_default_config = DefaultConfig()
tsac_default_config.algorithm = "tsac"
tsac_default_config.world.trajectory_length = 8 # Must match transformer needs
tsac_default_config.tsac.embedding_dim = 128
tsac_default_config.tsac.transformer_n_layers = 2
tsac_default_config.tsac.transformer_n_heads = 4
tsac_default_config.tsac.transformer_hidden_dim = 256
tsac_default_config.tsac.lr = 1e-4
tsac_default_config.tsac.alpha = 0.1
tsac_default_config.training.learning_starts = 2000
tsac_default_config.training.batch_size = 128
tsac_default_config.training.models_dir = "models/tsac/"
# tsac_default_config._sync_sequence_lengths()
# tsac_default_config._resolve_estimator_config()


# --- Vast Config (Example for large scale training, maybe T-SAC) ---
vast_config = DefaultConfig()
vast_config.algorithm = "tsac"
vast_config.world.trajectory_length = 16 # Longer sequence
vast_config.training.num_episodes = 50000
vast_config.training.max_steps = 200
vast_config.training.save_interval = 5000
vast_config.training.batch_size = 128 # Maybe increase?
vast_config.training.learning_starts = 5000
# Set vast config to use particle filter
vast_config.world.estimator_config = vast_config.particle_filter
vast_config.particle_filter.num_particles = 5000
vast_config.world.randomize_agent_initial_location = True
vast_config.world.randomize_landmark_initial_location = True
vast_config.world.randomize_landmark_initial_velocity = True
vast_config.tsac.hidden_dims = [512, 512, 256] # Actor/Critic MLP hidden dims
vast_config.tsac.embedding_dim = 256
vast_config.tsac.transformer_n_layers = 4
vast_config.tsac.transformer_n_heads = 8
vast_config.tsac.transformer_hidden_dim = 1024
vast_config.tsac.lr = 5e-5
vast_config.tsac.alpha = 0.05
# vast_config._sync_sequence_lengths()
# vast_config._resolve_estimator_config()

sac_rnn_config = DefaultConfig()
sac_rnn_config.sac.use_rnn = True
sac_rnn_config.training.models_dir = "models/sac_rnn/"

CONFIGS: Dict[str, DefaultConfig] = {
    "default": default_config,
    "sac_rnn": sac_rnn_config,
    "tsac_default": tsac_default_config,
    "vast": vast_config,
}