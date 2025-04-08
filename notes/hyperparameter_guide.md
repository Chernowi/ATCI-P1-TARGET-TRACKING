This guide explains the effects of changing hyperparameters in configs.py to help you optimize your reinforcement learning system.

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
| `initial_range_stddev` | Initial particle spread | Larger values create wider initial distribution; smaller values create 