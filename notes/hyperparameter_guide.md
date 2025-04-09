# Hyperparameter Guide

This document provides guidance on the configurable parameters for the reinforcement learning agents and environment.

## SAC Agent Configuration

| Parameter | Description | Effect of Changes |
|-----------|-------------|------------------|
| `state_dim` | Dimension of state space | Must match environment's state representation; changing requires modifying network architecture |
| `action_dim` | Dimension of action space | Must match environment's action space; changing requires modifying network architecture |
| `action_scale` | Scales action outputs | Higher values allow more aggressive movement; lower values create more precise but slower movement |
| `hidden_dim` | Neural network size | Larger values increase model capacity but require more computation; too small may underfit |
| `log_std_min/max` | Bounds for action distribution | Controls exploration range; wider bounds allow more exploration but may destabilize training |
| `lr` | Learning rate | Higher values speed up learning but may cause instability; lower values are more stable but learn slower |
| `gamma` | Discount factor | Higher values (closer to 1) prioritize long-term rewards; lower values focus on immediate rewards |
| `tau` | Target network update rate | Higher values adapt faster but may cause instability; lower values are more stable but adapt slower |
| `alpha` | Temperature parameter | Higher values encourage more exploration; lower values favor exploitation |
| `auto_tune_alpha` | Automatic alpha tuning | Enables automatic adjustment of exploration-exploitation balance |

## Replay Buffer Configuration

| Parameter | Description | Effect of Changes |
|-----------|-------------|------------------|
| `capacity` | Buffer size | Larger buffers store more experience but use more memory; too small may cause forgetting of useful experiences |

## Training Configuration

| Parameter | Description | Effect of Changes |
|-----------|-------------|------------------|
| `num_episodes` | Training duration | More episodes allow more learning but take longer; increase if underfitting |
| `max_steps` | Episode length limit | Longer episodes allow more exploration per episode; shorter episodes may train faster |
| `batch_size` | Training batch size | Larger batches give more stable gradients but use more memory; smaller batches may be noisier but update more frequently |
| `save_interval` | Model saving frequency | Lower values save more frequently (safer but slower); higher values run faster but risk losing progress |
| `log_frequency` | Frequency for logging to TensorBoard | Higher frequency gives more detailed training curves but slower training |
| `models_dir` | Directory for saving models | Organization parameter for model storage |
| `learning_starts` | Steps to collect before training | Higher values ensure more initial data but delay learning start |
| `train_freq` | Update frequency (steps) | More frequent updates learn faster but less efficiently |
| `gradient_steps` | Gradient steps when training | More steps extract more from each batch but slower training |

## World Configuration

| Parameter | Description | Effect of Changes |
|-----------|-------------|------------------|
| `dt` | Time step | Smaller values create smoother simulation but slower training; larger values speed up training but may reduce stability |
| `success_threshold` | Goal proximity threshold | Smaller values require more precision; larger values make tasks easier |
| `randomization_ranges` | Initial position/velocity ranges | Wider ranges create more diverse training scenarios but harder learning; narrower ranges focus learning but may reduce generalization |
| `step_penalty` | Per-step penalty | Higher values encourage faster task completion; lower values allow more exploration |
| `success_bonus` | Reward for success | Higher values prioritize task completion; ensure it outweighs accumulated step penalties |
| `out_of_range_penalty` | Penalty for exceeding limits | Higher values more strongly discourage boundary violations |

## Particle Filter Configuration

| Parameter | Description | Effect of Changes |
|-----------|-------------|------------------|
| `num_particles` | Number of particles | More particles improve accuracy but increase computation; fewer particles run faster but less accurately |
| `initial_range_stddev` | Standard deviation for initial particle spread | Larger values create wider initial distribution; smaller values create more concentrated initial estimates |
| `initial_velocity_guess` | Initial velocity guess for particles | Higher values assume faster moving landmarks; lower values assume slower moving landmarks |
| `estimation_method` | Method for estimation ("range" or "area") | "range" uses distance measurements; "area" uses spatial area-based estimation |
| `max_particle_range` | Maximum range for particles | Larger values allow wider search area but may dilute particle concentration |
| `process_noise_pos` | Process noise for position | Higher values accommodate more uncertain landmark movement; lower values assume more predictable movement |
| `process_noise_orient` | Process noise for orientation | Higher values allow more orientation change; lower values assume more consistent orientation |
| `process_noise_vel` | Process noise for velocity | Higher values accommodate more acceleration; lower values assume more constant velocity |
| `measurement_noise_stddev` | Standard deviation for measurement noise | Higher values make filter more tolerant to noisy measurements; lower values trust measurements more |
| `resampling_method` | Method for resampling | Different resampling algorithms (values 1-3) affect particle diversity and filter convergence |
| `pf_eval_max_mean_range_error_factor` | Factor of max_particle_range for error threshold | Higher values allow more estimation error; lower values enforce stricter accuracy |
| `pf_eval_dispersion_threshold` | Dispersion threshold for quality check | Higher values allow more spread in particles; lower values require more concentrated estimates |

## Least Squares Configuration

| Parameter | Description | Effect of Changes |
|-----------|-------------|------------------|
| `history_size` | Number of measurements to keep in history | Larger history provides more data for estimation but uses more memory and may include outdated measurements |
| `min_points_required` | Minimum points required for estimation | Higher values ensure more robust estimates but may delay initial estimations |
| `position_buffer_size` | Number of position estimates to keep | Larger buffers smooth estimated positions but may introduce lag |
| `velocity_smoothing` | Number of points for velocity smoothing | More points give smoother velocity estimates but slower response to changes |
| `min_observer_movement` | Minimum movement required between measurements | Higher values ensure sufficient observer movement for triangulation; lower values use more measurement points |

## PPO Configuration

| Parameter | Description | Effect of Changes |
|-----------|-------------|------------------|
| `state_dim` | State dimension | Must match environment state space |
| `action_dim` | Action dimension | Must match environment action space |
| `action_scale` | Action scale factor | Higher values allow larger actions; lower values restrict action magnitude |
| `hidden_dim` | Hidden layer dimension | Larger networks can learn more complex policies but train slower |
| `log_std_min` | Minimum log std for actions | Higher values enforce more exploration; lower values allow more precision |
| `log_std_max` | Maximum log std for actions | Higher values allow more random exploration; lower values limit exploration |
| `actor_lr` | Actor learning rate | Higher rates update policy faster but may be unstable |
| `critic_lr` | Critic learning rate | Higher rates learn value functions faster but may be unstable |
| `gamma` | Discount factor | Higher values prioritize long-term rewards; lower values focus on immediate rewards |
| `gae_lambda` | GAE lambda parameter | Higher values reduce bias but increase variance in advantage estimation |
| `policy_clip` | PPO clipping parameter | Higher values allow larger policy updates but may destabilize training |
| `n_epochs` | Optimization epochs per update | More epochs extract more from each batch but may cause overfitting |
| `entropy_coef` | Entropy coefficient | Higher values encourage more exploration; lower values focus on exploitation |
| `value_coef` | Value loss coefficient | Higher values prioritize value function accuracy; lower values focus on policy improvement |
| `batch_size` | Batch size for training | Larger batches provide more stable gradients but require more memory |
| `steps_per_update` | Environment steps between updates | More steps provide more diverse data per update but slower training |