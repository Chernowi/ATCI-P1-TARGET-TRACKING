import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import time
from collections import deque
from tqdm import tqdm
from world import World
# from world_objects import Velocity # No longer needed
from configs import DefaultConfig, PPOConfig, TrainingConfig
from torch.utils.tensorboard import SummaryWriter
import math # For PolicyNetwork initialization

class PPOMemory:
    """Memory buffer for PPO algorithm to store trajectories."""

    def __init__(self, batch_size=64):
        self.states = []
        self.actions = [] # Action is float
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, action, probs, vals, reward, done):
        """Store a transition in memory."""
        # state is tuple, action is float, probs/vals are arrays/floats
        self.states.append(state) # Store basic_state tuple
        self.actions.append(action) # Store float action
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        """Clear the memory after an update."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        """Generate batches for training."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices) # Shuffle indices for random batches
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # Convert stored lists to numpy arrays before batching
        states_arr = np.array(self.states, dtype=np.float32)
        # Action shape (n_states,) -> (n_states, 1)
        actions_arr = np.array(self.actions, dtype=np.float32).reshape(-1, 1)
        probs_arr = np.array(self.probs, dtype=np.float32) # Shape (n_states, 1)
        vals_arr = np.array(self.vals, dtype=np.float32) # Shape (n_states, 1)
        rewards_arr = np.array(self.rewards, dtype=np.float32) # Shape (n_states,)
        dones_arr = np.array(self.dones, dtype=np.float32) # Shape (n_states,)

        return states_arr, actions_arr, probs_arr, vals_arr, rewards_arr, dones_arr, batches

    def __len__(self):
        return len(self.states)


class PolicyNetwork(nn.Module):
    """Actor network for PPO that outputs action means and standard deviations."""

    def __init__(self, config: PPOConfig):
        super(PolicyNetwork, self).__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim # Should be 1
        self.hidden_dim = config.hidden_dim
        # self.action_scale = config.action_scale # No longer used here
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        action_mean = self.mean(x)
        action_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_log_std)

        return action_mean, action_std

    def sample(self, state):
        """Sample action from the policy distribution."""
        mean, std = self.forward(state)
        distribution = Normal(mean, std)

        x_t = distribution.sample()
        action_normalized = torch.tanh(x_t) # Squash to [-1, 1]

        # Calculate log probability with tanh correction
        log_prob = distribution.log_prob(x_t) - torch.log(1 - action_normalized.pow(2) + 1e-6)
        # Sum over action dimension (dim=1), keep batch dim
        log_prob = log_prob.sum(1, keepdim=True)

        return action_normalized, log_prob # Return normalized action

    def evaluate(self, state, action_normalized):
        """Evaluate log probability and entropy for a given state-action pair."""
        mean, std = self.forward(state)
        distribution = Normal(mean, std)

        # Un-squash action from [-1, 1] to original Gaussian space for log_prob calc
        # action_normalized shape (batch, 1)
        action_tanh = torch.clamp(action_normalized, -0.9999, 0.9999)
        action_original_space = torch.atanh(action_tanh)

        log_prob = distribution.log_prob(action_original_space) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        # Sum over action dimension (dim=1), keep batch dim
        log_prob = log_prob.sum(1, keepdim=True)

        entropy = distribution.entropy().sum(1, keepdim=True)

        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network for PPO that estimates the value function."""

    def __init__(self, config: PPOConfig):
        super(ValueNetwork, self).__init__()
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value


class PPO:
    """Proximal Policy Optimization algorithm implementation."""

    def __init__(self, config: PPOConfig, device: torch.device = None):
        """Initialize PPO agent using configuration."""
        self.config = config
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip
        self.n_epochs = config.n_epochs
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
        # self.action_scale = config.action_scale # Not used here

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"PPO Agent using device: {self.device}")

        self.actor = PolicyNetwork(config).to(self.device)
        self.critic = ValueNetwork(config).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.memory = PPOMemory(batch_size=config.batch_size)

    def select_action(self, state, evaluate=False):
        """Select action (normalized yaw change [-1, 1]) based on state."""
        state_tuple = state['basic_state']
        state = torch.FloatTensor(state_tuple).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                # Use mean action for evaluation
                action_mean, _ = self.actor.forward(state)
                action_normalized = torch.tanh(action_mean)
                log_prob = None # Not needed for eval
                value = None    # Not needed for eval
            else:
                # Sample action for training
                action_normalized, log_prob = self.actor.sample(state)
                value = self.critic(state)

                # Store tuple state, float action, log_prob, value
                self.memory.store(
                    state_tuple,
                    action_normalized.cpu().numpy()[0, 0], # Store float action
                    log_prob.cpu().numpy()[0], # Store log_prob (shape (1,))
                    value.cpu().numpy()[0],    # Store value (shape (1,))
                    0, False # Placeholder reward, done
                )

        # Return the normalized action [-1, 1] as a float
        return action_normalized.detach().cpu().numpy()[0, 0]

    def store_transition(self, reward, done):
        """Store reward and done flag for the last transition."""
        # Check necessary because select_action stores placeholders
        if len(self.memory.rewards) < len(self.memory.states):
            self.memory.rewards.append(reward)
            self.memory.dones.append(done)
        elif len(self.memory.rewards) == len(self.memory.states):
             # If called again after placeholders are filled, update the last one
             self.memory.rewards[-1] = reward
             self.memory.dones[-1] = done

    def update_parameters(self):
        """Update PPO parameters based on collected trajectories."""
        if len(self.memory) < self.config.batch_size:
            return None # Not enough data yet

        # Calculate returns and advantages using memory data
        returns = self._compute_returns_and_advantages() # Shape (n_steps,)

        actor_losses, critic_losses, entropies = [], [], []

        states_arr, actions_arr, old_log_probs_arr, values_arr, _, _, batches = self.memory.generate_batches()
        # Shapes: states(N, s_dim), actions(N, 1), old_log_probs(N, 1), values(N, 1)

        # Calculate advantages: shape (N,) -> (N, 1)
        advantages = returns - values_arr.squeeze()
        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Normalize advantages
        if len(advantages) > 1:
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert returns to tensor: shape (N,) -> (N, 1)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Optimize policy and value networks for n_epochs
        for _ in range(self.n_epochs):
            for batch_indices in batches:
                # Extract batch data using indices
                batch_states = torch.tensor(states_arr[batch_indices], dtype=torch.float32).to(self.device)
                batch_actions = torch.tensor(actions_arr[batch_indices], dtype=torch.float32).to(self.device) # Shape (batch, 1)
                batch_old_log_probs = torch.tensor(old_log_probs_arr[batch_indices], dtype=torch.float32).to(self.device) # Shape (batch, 1)
                batch_returns = returns[batch_indices] # Shape (batch, 1)
                batch_advantages = advantages[batch_indices] # Shape (batch, 1)

                # Evaluate actions and calculate ratio
                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions) # Shapes (batch, 1)
                new_values = self.critic(batch_states) # Shape (batch, 1)

                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                critic_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Optimize networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        # Clear memory after update
        self.memory.clear()

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies)
        }

    def _compute_returns_and_advantages(self):
        """Compute returns using Generalized Advantage Estimation (GAE)."""
        rewards = np.array(self.memory.rewards) # Shape (N,)
        dones = np.array(self.memory.dones)     # Shape (N,)
        values = np.array(self.memory.vals).squeeze() # Shape (N, 1) -> (N,)

        returns = np.zeros_like(rewards)
        gae = 0.0

        # Calculate GAE starting from the last step
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                # No next state for the very last transition in the buffer
                # Bootstrap with 0 if done, else use the value estimate?
                # PPO typically collects full trajectories, so next_value is tricky
                # Common approach: if last state is not terminal, bootstrap with V(s_last), else 0
                # Let's assume the buffer ends with a potentially non-terminal state.
                # Need V(s_N) - but memory only has V(s_0) to V(s_{N-1})
                # Recalculate V(s_N) if needed, or use V(s_{N-1}) as approx if done?
                # Let's use V(s_N-1) if done[N-1] is False, else 0.
                # A cleaner way: ensure value is calculated for the state *after* the last action.
                # Revisit this - for now, using V(s_N-1) if not done.
                # The current memory stores V(s_t) alongside a_t, r_t, done_t.
                # So values[step] is V(s_step). We need V(s_{step+1}).
                next_value = values[step] if not dones[step] else 0 # Approximation
            else:
                 # V(s_{step+1}) = values[step + 1]
                 next_value = values[step + 1]

            delta = rewards[step] + self.gamma * next_value * (1 - int(dones[step])) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[step])) * gae
            returns[step] = gae + values[step] # Return is GAE + V(s_t)

        return returns # Shape (N,)

    def save_model(self, path: str):
        """Save model and optimizer states."""
        print(f"Saving PPO model to {path}...")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'device_type': self.device.type
        }, path)

    def load_model(self, path: str):
        """Load model and optimizer states."""
        if not os.path.exists(path):
            print(f"Warning: PPO model file not found at {path}. Skipping loading.")
            return
        print(f"Loading PPO model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"PPO model loaded successfully from {path}")


def train_ppo(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    """Train the PPO agent on the environment defined by the config."""
    ppo_config = config.ppo
    train_config = config.training
    world_config = config.world
    cuda_device = config.cuda_device

    log_dir = os.path.join("runs", f"ppo_training_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Device Setup
    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Warning: Multi-GPU not standard for PPO. Using single specified/default GPU: {cuda_device}")
            device = torch.device(cuda_device)
        else:
            device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try:
                torch.cuda.set_device(device)
                print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e:
                 print(f"Warning: Could not set CUDA device {cuda_device}. Error: {e}")
                 device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU for training")

    agent = PPO(config=ppo_config, device=device)
    os.makedirs(train_config.models_dir, exist_ok=True)

    episode_rewards = []
    timing_metrics = {
        'env_step_time': deque(maxlen=100),
        'parameter_update_time': deque(maxlen=100)
    }

    total_steps = 0
    update_frequency = ppo_config.steps_per_update
    pbar = tqdm(range(1, train_config.num_episodes + 1), desc="Training PPO", unit="episode")

    for episode in pbar:
        env = World(world_config=world_config)
        state = env.encode_state() # state is dict
        episode_reward = 0
        episode_steps = 0

        episode_step_times = []
        episode_param_update_times = []
        learn_steps = 0 # Track steps collected for the current update cycle

        for step_in_episode in range(train_config.max_steps):
            # Select action (float, normalized yaw change)
            action_normalized = agent.select_action(state, evaluate=False)

            step_start_time = time.time()
            # Pass float action to env.step
            env.step(action_normalized, training=True)
            step_time = time.time() - step_start_time
            episode_step_times.append(step_time)
            timing_metrics['env_step_time'].append(step_time)

            reward = env.reward
            next_state = env.encode_state() # next_state is dict
            done = env.done

            # Store reward and done for the transition selected just before
            agent.store_transition(reward, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            learn_steps += 1

            # Check if enough steps collected for an update
            if learn_steps >= update_frequency:
                update_start_time = time.time()
                losses = agent.update_parameters()
                update_time = time.time() - update_start_time
                episode_param_update_times.append(update_time)
                timing_metrics['parameter_update_time'].append(update_time)
                learn_steps = 0 # Reset step counter for next update cycle

                if losses:
                    writer.add_scalar('Loss/Actor', losses['actor_loss'], total_steps)
                    writer.add_scalar('Loss/Critic', losses['critic_loss'], total_steps)
                    writer.add_scalar('Policy/Entropy', losses['entropy'], total_steps)

            if done:
                break

        # End of episode logging
        episode_rewards.append(episode_reward)

        if episode % train_config.log_frequency == 0:
            if timing_metrics['env_step_time']:
                 avg_step_time = np.mean(timing_metrics['env_step_time'])
                 writer.add_scalar('Time/Environment_Step_ms_Avg100', avg_step_time * 1000, total_steps)
            if timing_metrics['parameter_update_time']:
                 avg_param_update_time = np.mean(timing_metrics['parameter_update_time'])
                 writer.add_scalar('Time/Parameter_Update_ms_Avg100', avg_param_update_time * 1000, total_steps)

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Error/Distance_EndEpisode', env.error_dist, total_steps)

            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar.set_postfix({'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps})

        if episode % train_config.save_interval == 0:
            save_path = os.path.join(train_config.models_dir, f"ppo_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

    pbar.close()
    writer.close()
    print(f"Training finished. Total steps: {total_steps}")

    final_save_path = os.path.join(train_config.models_dir, f"ppo_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_ppo(agent=agent, config=config)

    return agent, episode_rewards


def evaluate_ppo(agent: PPO, config: DefaultConfig):
    """Evaluate the trained PPO agent using evaluation configuration."""
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    # --- Conditional import for visualization ---
    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            import imageio.v2 as imageio # Needed by save_gif
            vis_available = True
            print("Visualization enabled.")
        except ImportError:
            print("Visualization libraries not found (matplotlib, imageio). Rendering disabled.")
            vis_available = False
            eval_config.render = False # Force disable rendering
    else:
        vis_available = False
        print("Rendering disabled by config.")

    eval_rewards = []
    success_count = 0
    all_episode_gif_paths = []

    agent.actor.eval()
    agent.critic.eval()

    print(f"\nRunning PPO Evaluation for {eval_config.num_episodes} episodes...")
    for episode in range(eval_config.num_episodes):
        env = World(world_config=world_config)
        state = env.encode_state() # state is dict
        episode_reward = 0
        episode_frames = []

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            try:
                initial_frame_file = visualize_world(
                    world=env,
                    vis_config=vis_config,
                    filename=f"ppo_eval_ep{episode+1}_frame_000_initial.png",
                    collect_for_gif=True
                )
                if initial_frame_file and os.path.exists(initial_frame_file):
                    episode_frames.append(initial_frame_file)
            except Exception as e:
                print(f"Warning: Visualization failed for initial state. Error: {e}")

        for step in range(eval_config.max_steps):
            # Action is float (normalized yaw change)
            action_normalized = agent.select_action(state, evaluate=True)

            # Step environment with float action
            env.step(action_normalized, training=False) # training=False for eval
            reward = env.reward
            next_state = env.encode_state() # next_state is dict
            done = env.done

            if eval_config.render and vis_available:
                try:
                    frame_file = visualize_world(
                        world=env,
                        vis_config=vis_config,
                        filename=f"ppo_eval_ep{episode+1}_frame_{step+1:03d}.png",
                        collect_for_gif=True
                    )
                    if frame_file and os.path.exists(frame_file):
                        episode_frames.append(frame_file)
                except Exception as e:
                     print(f"Warning: Visualization failed for step {step+1}. Error: {e}")

            state = next_state
            episode_reward += reward

            if done:
                if env.error_dist <= world_config.success_threshold:
                    success_count += 1
                    print(f"  Episode {episode+1}: Success! Found landmark at step {step+1} (Error: {env.error_dist:.4f} <= threshold {world_config.success_threshold})")
                else:
                    print(f"  Episode {episode+1}: Terminated early at step {step+1} (Not success). Final Error: {env.error_dist:.4f}")
                break

        if not done:
             success = env.error_dist <= world_config.success_threshold
             status = "Success!" if success else "Failure."
             if success: success_count +=1
             print(f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps} reached). Final Error: {env.error_dist:.4f}. {status}")

        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        if eval_config.render and vis_available and episode_frames:
            gif_filename = f"ppo_eval_episode_{episode+1}.gif"
            try:
                gif_path = save_gif(
                    output_filename=gif_filename,
                    vis_config=vis_config,
                    frame_paths=episode_frames,
                    delete_frames=vis_config.delete_frames_after_gif
                )
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e:
                print(f"  Warning: Failed to create or save PPO GIF for episode {episode+1}. Error: {e}")
            if vis_config.delete_frames_after_gif: # Clean up frames
                 for frame in episode_frames:
                     if os.path.exists(frame):
                         try: os.remove(frame)
                         except OSError as ose: print(f"    Warning: Could not delete PPO frame file {frame}: {ose}")


    agent.actor.train()
    agent.critic.train()

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0

    print("\n--- PPO Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Reward: {avg_eval_reward:.2f} +/- {std_eval_reward:.2f}")
    print(f"Success Rate (Error <= {world_config.success_threshold}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_episode_gif_paths:
        print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available:
        print("Rendering was enabled in config, but visualization libraries were not found.")
    elif not eval_config.render:
        print("Rendering was disabled.")
    print("--- End PPO Evaluation ---\n")

    return eval_rewards, success_rate