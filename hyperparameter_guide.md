# Hyperparameter guide

General Tuning Strategy:
1.  **Start with Defaults:** The provided `default_config` and algorithm-specific bases (`sac_rnn`, `ppo_mlp`, etc.) are good starting points.
2.  **Understand Your Task:** Is it exploration-heavy? Are rewards sparse? Is the state complex? This influences which parameters are most sensitive.
3.  **Tune One (or a Few) at a Time:** Or use more systematic methods like grid search, random search, or Bayesian optimization (e.g., with Optuna, Ray Tune) if you have the computational resources.
4.  **Log Everything:** Use TensorBoard (as set up) to track rewards, losses, and other metrics. This helps identify effects of changes.
5.  **Be Patient:** RL training can take time and many episodes to show clear trends.
6.  **Focus on Stability First:** Ensure losses are decreasing (or stabilizing) and rewards are trending upwards before fine-tuning.

---

Parameter Breakdown:

**I. Core RL Algorithm Configurations (`SACConfig`, `PPOConfig`)**

These are generally the most impactful for agent performance.

*   **`state_dim`, `action_dim`**:
    *   **What:** Dimensions of state and action spaces.
    *   **Impact:** Defined by the environment (`CORE_STATE_DIM`, `CORE_ACTION_DIM`). Not typically tuned unless you change the environment's state/action representation.
    *   **Tuning:** N/A for standard use.

*   **`hidden_dims` (SAC) / `hidden_dim` (PPO)**:
    *   **What:** Sizes of hidden layers in the neural networks (Actor and Critic). SAC uses a list for multiple layers, PPO uses a single value (implying one or more layers of this size, check network architecture).
    *   **Impact:**
        *   *Too small:* Underfitting, agent may not learn complex policies/value functions.
        *   *Too large:* Overfitting (less common in RL than supervised learning, but possible), slower training, higher memory usage.
    *   **Tuning:**
        *   Start with defaults (e.g., `[64, 64]` for SAC, `256` for PPO).
        *   If learning is slow or plateaus low, try increasing (e.g., `[128, 128]`, `[256, 256]` or `512`).
        *   If training is very slow or you suspect overfitting, try decreasing (e.g., `[32, 32]` or `128`).

*   **`actor_lr`, `critic_lr`**:
    *   **What:** Learning rates for the actor and critic optimizers.
    *   **Impact:**
        *   *Too high:* Unstable training, losses might oscillate wildly or diverge. Policy might become erratic.
        *   *Too low:* Very slow convergence.
    *   **Tuning:**
        *   Often the most critical parameters.
        *   Typical range: `1e-5` to `1e-3`.
        *   Try powers of 10 or factors of 3-5 (e.g., `1e-5, 3e-5, 1e-4, 3e-4, 1e-3`).
        *   Sometimes, critic LR is set slightly higher than actor LR. The defaults (`5e-5` for SAC, `5e-6` (actor) and `1e-3` (critic) for PPO) reflect this.

*   **`gamma` (Discount Factor)**:
    *   **What:** How much future rewards are valued relative to immediate rewards. `R = r_0 + gamma*r_1 + gamma^2*r_2 + ...`
    *   **Impact:**
        *   *Closer to 1 (e.g., 0.99, 0.999):* Agent is more "far-sighted," considers long-term rewards. Suitable for tasks with long episodes or delayed rewards.
        *   *Closer to 0 (e.g., 0.9, 0.95):* Agent is more "short-sighted," prioritizes immediate rewards. Can be better if long-term credit assignment is noisy or task is short.
    *   **Tuning:**
        *   Defaults (`0.99`) are common.
        *   If episodes are very long or rewards very sparse, try `0.995` or `0.999`.
        *   If learning seems unstable due to noisy long-term values, try slightly lower like `0.98`, `0.95`.

*   **`use_rnn`, `rnn_type`, `rnn_hidden_size`, `rnn_num_layers`**:
    *   **What:** Parameters for using Recurrent Neural Networks (LSTM or GRU).
    *   **Impact:**
        *   `use_rnn = True`: Essential if the `trajectory_length` from `WorldConfig` is greater than 1 and you want the agent to process the sequence effectively (rather than flattening it or just using the last step for MLP). For PPO, this means using recurrent PPO.
        *   `rnn_type`: `lstm` is often more powerful but `gru` can be faster and work well.
        *   `rnn_hidden_size`: Similar to MLP hidden layer size. Affects capacity of the recurrent layer.
        *   `rnn_num_layers`: Usually 1 or 2. More layers add capacity but increase complexity and training time.
    *   **Tuning:**
        *   First, decide if RNN is needed based on state representation. If `trajectory_length > 1` in `WorldConfig`, RNN is recommended.
        *   `rnn_hidden_size`: Tune like MLP hidden sizes (e.g., 64, 128, 256).
        *   `rnn_num_layers`: Start with 1, try 2 if performance is lacking.

**SAC-Specific Parameters:**

*   **`log_std_min`, `log_std_max`**:
    *   **What:** Bounds on the logarithm of the standard deviation of the action distribution.
    *   **Impact:** Controls the range of exploration. Too narrow might restrict exploration; too wide might lead to very noisy actions.
    *   **Tuning:** Defaults (`-20`, `1` or `2`) are generally robust. `log_std_max` can sometimes be reduced (e.g., to `0` or `-1`) if actions are too erratic.

*   **`alpha_lr`**:
    *   **What:** Learning rate for the entropy temperature `alpha` (if `auto_tune_alpha=True`).
    *   **Impact:** How quickly `alpha` adapts. Similar sensitivity to other LRs.
    *   **Tuning:** Usually same order of magnitude as actor/critic LRs (e.g., `3e-4`, `5e-5`).

*   **`tau` (Target Network Update Rate)**:
    *   **What:** Soft update parameter for target networks. `target_params = tau*online_params + (1-tau)*target_params`.
    *   **Impact:**
        *   *Small `tau` (e.g., 0.001, 0.005):* More stable updates, slower propagation of value changes to target.
        *   *Large `tau` (e.g., 0.01, 0.1):* Faster updates, can be less stable. `tau=1` means hard update.
    *   **Tuning:** `0.005` is common. Try `0.001` for more stability or `0.01` for faster target updates.

*   **`alpha` (Entropy Temperature)**:
    *   **What:** Balances reward maximization and policy entropy (exploration). If `auto_tune_alpha=True`, this is the initial value.
    *   **Impact:**
        *   *Higher `alpha`*: More exploration, "softer" policy.
        *   *Lower `alpha`*: Less exploration, more greedy policy.
    *   **Tuning:**
        *   If `auto_tune_alpha=True` (recommended), the initial value isn't super critical but can affect early exploration. `0.2` is a common start.
        *   If `auto_tune_alpha=False`, this becomes a very important hyperparameter to tune. Try values like `0.05, 0.1, 0.2, 0.5`.

*   **`auto_tune_alpha`**:
    *   **What:** Whether to automatically adjust `alpha` to meet a target entropy.
    *   **Impact:** Generally beneficial, removes `alpha` as a manual hyperparameter.
    *   **Tuning:** Usually set to `True`.

*   **PER Settings (`use_per`, `per_alpha`, `per_beta_start`, `per_beta_end`, `per_beta_anneal_steps`, `per_epsilon`)**:
    *   **What:** Parameters for Prioritized Experience Replay.
    *   **Impact:**
        *   `use_per=True`: Can speed up learning by focusing on "surprising" transitions.
        *   `per_alpha`: Controls how much prioritization is used (0=uniform, 1=full priority). `0.6-0.7` common.
        *   `per_beta_start/end`: For importance sampling correction, annealed from start to end. `0.4` to `1.0` common.
        *   `per_beta_anneal_steps`: How many agent steps to anneal beta. Should be a significant fraction of total training steps.
        *   `per_epsilon`: Small value to ensure non-zero priority.
    *   **Tuning:**
        *   First try `use_per=False`. If performance is good, stick with it for simplicity.
        *   If `use_per=True`, start with common values: `alpha=0.6`, `beta_start=0.4`. Anneal beta over a large portion of training.

**PPO-Specific Parameters:**

*   **`gae_lambda` (Generalized Advantage Estimation Lambda)**:
    *   **What:** Parameter for GAE, trading off bias and variance in advantage estimation.
    *   **Impact:**
        *   `lambda=1`: High variance (like Monte Carlo returns).
        *   `lambda=0`: High bias (like TD(0) error).
    *   **Tuning:** `0.9` to `0.98` common. `0.95` is a good default.

*   **`policy_clip` (Clipping Parameter Epsilon)**:
    *   **What:** Limits the policy update ratio in PPO's surrogate objective.
    *   **Impact:**
        *   *Smaller clip (e.g., 0.05, 0.1):* More conservative updates, can be more stable but slower.
        *   *Larger clip (e.g., 0.2, 0.3):* More aggressive updates, can be faster but less stable.
    *   **Tuning:** `0.05` to `0.2`. Default `0.05` is quite conservative; `0.1` or `0.2` are also common.

*   **`n_epochs` (Optimization Epochs per Update)**:
    *   **What:** Number of times to iterate over the collected rollout data for policy/value updates.
    *   **Impact:**
        *   *More epochs:* More learning from the same data, potentially faster convergence if data is good. Can also lead to overfitting the current batch of data.
        *   *Fewer epochs:* Less chance of overfitting batch, but might need more rollouts.
    *   **Tuning:** `3` to `15`. Default `3` is low; `5-10` often works well.

*   **`entropy_coef` (Entropy Coefficient)**:
    *   **What:** Weight for the entropy bonus in the PPO loss function. Encourages exploration.
    *   **Impact:**
        *   *Higher:* More exploration, prevents premature convergence to suboptimal policies.
        *   *Lower:* Less exploration, more exploitation.
    *   **Tuning:** `0.001` to `0.05`. Default `0.015` is reasonable. If policy converges too quickly or gets stuck, increase this.

*   **`value_coef` (Value Loss Coefficient)**:
    *   **What:** Weight for the value function (critic) loss in the total PPO loss.
    *   **Impact:** Balances actor and critic learning.
    *   **Tuning:** `0.5` is very common. Can sometimes be tuned (e.g., `0.25` to `1.0`).

*   **`batch_size` (PPO batch size)**:
    *   **What:** For MLP PPO, it's the number of transitions used in a mini-batch during the `n_epochs` of updates. For Recurrent PPO, it's the number of *sequences* (rollouts) per batch.
    *   **Impact:**
        *   *Larger:* More stable gradient estimates, but more memory and computation per update step.
        *   *Smaller:* Noisier gradients, but faster iterations within an epoch.
    *   **Tuning:**
        *   *MLP:* Powers of 2 (e.g., `32, 64, 128, 256`). Must be smaller than `steps_per_update`.
        *   *RNN:* Fewer sequences (e.g., `4, 8, 16, 32`). Default `16` for RNN is okay.

*   **`steps_per_update` (Rollout Length)**:
    *   **What:** Number of environment steps collected before performing a PPO update.
    *   **Impact:**
        *   *Larger:* More data per update, potentially more stable updates, but less frequent policy changes. Better for GAE to see longer trajectories.
        *   *Smaller:* More frequent updates, policy adapts faster, but GAE might be less accurate.
    *   **Tuning:** `128` to `4096`. Default `2048` (MLP) or `256` (RNN) are common. RNNs often use shorter rollouts per update cycle due to sequence processing.

**II. Replay Buffer Configuration (`ReplayBufferConfig`) - SAC Specific**

*   **`capacity`**:
    *   **What:** Maximum number of *full trajectories* the replay buffer can store.
    *   **Impact:**
        *   *Too small:* Agent quickly forgets past experiences, potentially leading to "catastrophic forgetting" or instability if good experiences are discarded too soon.
        *   *Too large:* More diverse experiences, but can slow down adaptation to new policy improvements (off-policy learning can suffer from stale data). Higher memory usage.
    *   **Tuning:** `1e5` to `1e6` is common. Default `1e6` is large. If memory is an issue or learning is slow to react, try `1e5` or `5e5`.

**III. Training Configuration (`TrainingConfig`)**

*   **`num_episodes`, `max_steps`**:
    *   **What:** Define the length of training and max steps per episode.
    *   **Impact:** More episodes/steps generally lead to better learning, up to a point. `max_steps` affects episode length, influencing `gamma` choice.
    *   **Tuning:** Set based on computational budget and task complexity. Not hyperparameters for optimizing algorithm performance per se, but for overall training.

*   **`sac_batch_size`**:
    *   **What:** Number of *trajectories* sampled from the replay buffer for SAC training updates.
    *   **Impact:**
        *   *Larger:* More stable gradients, but more computation per update.
        *   *Smaller:* Noisier gradients, less computation per update.
    *   **Tuning:** Powers of 2 (e.g., `32, 64, 128, 256`). Default `32` is relatively small; `64` or `128` are also common.

*   **`save_interval`, `log_frequency`**: For saving models and logging. Not performance hyperparameters.

*   **`learning_starts` (SAC)**:
    *   **What:** Number of environment steps to collect experiences *before* starting any training updates.
    *   **Impact:** Fills the replay buffer with initial random/exploratory data.
        *   *Too small:* Agent starts learning from very few, potentially uninformative experiences.
        *   *Too large:* Delays learning.
    *   **Tuning:** Should be at least `sac_batch_size` (multiplied by trajectory length to get #transitions), often much larger (e.g., `1000` to `10000` steps). Default `8000` is reasonable.

*   **`train_freq` (SAC)**:
    *   **What:** Update the policy every `train_freq` environment steps.
    *   **Impact:**
        *   *Low `train_freq` (e.g., 1, 4):* More frequent updates, potentially faster learning but more computation.
        *   *High `train_freq` (e.g., 30, 50):* Less frequent updates.
    *   **Tuning:** If `gradient_steps > 1`, `train_freq` can be higher. If `gradient_steps = 1`, `train_freq` is often small (e.g., 1 to 10). The default `train_freq=30` with `gradient_steps=20` means a burst of updates.

*   **`gradient_steps` (SAC)**:
    *   **What:** How many gradient update steps to perform when `train_freq` is met.
    *   **Impact:**
        *   *Higher:* More learning from the currently sampled batch. Can improve sample efficiency.
        *   *Lower (e.g., 1):* Standard single update.
    *   **Tuning:** Often `1`. If `train_freq` is high, `gradient_steps` might also be higher (e.g., equal to `train_freq` for "DDPG-style" updates or the default `20`).

*   **`normalize_rewards`**:
    *   **What:** Whether to normalize rewards using a running mean and standard deviation.
    *   **Impact:** Can significantly stabilize training by keeping reward magnitudes consistent, making LR tuning easier. Generally recommended (`True`).
    *   **Tuning:** Usually `True`. If rewards are already well-scaled and stable, might not be necessary.

**IV. World Configuration (`WorldConfig`)**

Many of these define the environment itself, rather than being tunable hyperparameters for a *fixed* environment. However, changing them changes the task difficulty.

*   `dt`, `agent_speed`, `yaw_angle_range`, `world_x_bounds`, `world_y_bounds`, `landmark_depth_bounds`: Environment dynamics and normalization constants. Not typically tuned for agent performance on a fixed task.
*   `normalize_state`: Should be `True` if agent networks expect normalized inputs.
*   `randomize_*_initial_location/velocity`: Affects training distribution and generalization. More randomization makes the task harder but can lead to more robust policies.
*   **`trajectory_length`**:
    *   **What:** Number of past steps included in the state representation for MLP agents. For RNN agents, this might influence how rollouts are processed or initial hidden states are handled if full rollouts aren't used for every single step input during data collection.
    *   **Impact:**
        *   *Longer:* More historical context for the agent. Can be crucial for POMDPs.
        *   *Shorter:* Smaller state, less memory.
    *   **Tuning:** If `use_rnn=False`, this is a key hyperparameter for fixed-history MLP agents. Values from `1` (Markov state) to `5`, `10`, or more. If `use_rnn=True`, the RNN handles history, so this parameter's role might be less direct for the agent's learning, more about data structure.

*   **`range_measurement_base_noise`, `range_measurement_distance_factor`, `new_measurement_probability`**:
    *   **What:** Define observation noise and reliability.
    *   **Impact:** Higher noise or lower probability makes the estimation task (and thus control) harder.
    *   **Tuning:** These are environment properties. For experiments like "default_poor_signal", these are changed to create different task variants.

*   `success_threshold`, `collision_threshold`: Define episode termination conditions.
*   **Reward Function Parameters** (`reward_error_threshold`, `low_error_bonus`, etc.):
    *   **What:** Coefficients and thresholds for the reward function.
    *   **Impact:** *Massively* influences agent behavior. This is "reward engineering."
    *   **Tuning:** Requires careful thought about the desired behavior. Small changes can lead to very different policies. Experiment extensively if default rewards don't work.
        *   *Penalties vs. Bonuses:* Ensure a good balance.
        *   *Shaping Rewards:* Small, dense rewards (like `distance_reward_scale`) can guide learning, but too much shaping can lead to optimizing for the shaping reward instead of the true goal.

*   `estimator_config`: Selects `ParticleFilterConfig` or `LeastSquaresConfig`. A structural choice, not a typical hyperparameter.

**V. Estimator Configurations (`ParticleFilterConfig`, `LeastSquaresConfig`)**

These tune the landmark estimator used by the agent.

*   **`ParticleFilterConfig`**:
    *   `num_particles`:
        *   **Impact:** More particles -> better estimation accuracy, but higher computational cost.
        *   **Tuning:** Balance accuracy needs with simulation speed.
    *   `initial_range_stddev`, `process_noise_*`, `measurement_noise_stddev`:
        *   **Impact:** These define the PF's internal model of uncertainty. They should ideally reflect the true uncertainties in the `WorldConfig`, but can sometimes be tuned slightly to improve PF robustness or responsiveness. If the PF's `measurement_noise_stddev` is much lower than the world's actual noise, the PF might become overconfident and diverge.
    *   `pf_eval_max_mean_range_error_factor`, `pf_eval_dispersion_threshold`: Control PF re-initialization logic if it seems to have diverged.

*   **`LeastSquaresConfig`**:
    *   `history_size`:
        *   **Impact:** Number of measurements used. More points -> smoother estimate, less sensitive to noise, but slower to react to landmark maneuvers.
    *   `min_points_required`: Minimum points for an estimate.
    *   `location_smoothing_factor`: EMA for location updates. Lower -> smoother/slower.
    *   `min_observer_movement`: Threshold to ensure geometric diversity in measurements.

**VI. Other Configurations**

*   `EvaluationConfig`: Defines how evaluation runs are performed. Not for training performance.
*   `VisualizationConfig`: For saving visualizations. Not for training performance.
*   `cuda_device`, `algorithm`: High-level choices for execution.

