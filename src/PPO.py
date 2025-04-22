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
from configs import DefaultConfig, PPOConfig, TrainingConfig, CORE_STATE_DIM # Import core state dim
from torch.utils.tensorboard import SummaryWriter
import math

class PPOMemory:
    """Memory buffer for PPO algorithm. Stores individual basic states."""

    def __init__(self, batch_size=64):
        # Stores individual transitions, not trajectories
        self.states = [] # List of basic_state tuples
        self.actions = [] # List of float actions
        self.probs = []   # List of float log probs
        self.vals = []    # List of float values
        self.rewards = [] # List of float rewards
        self.dones = []   # List of bool dones
        self.batch_size = batch_size

    def store(self, basic_state, action, probs, vals, reward, done):
        """Store a transition in memory."""
        self.states.append(basic_state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = [], [], [], [], [], []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        states_arr = np.array(self.states, dtype=np.float32) # (N, state_dim)
        actions_arr = np.array(self.actions, dtype=np.float32).reshape(-1, 1) # (N, 1)
        probs_arr = np.array(self.probs, dtype=np.float32).reshape(-1, 1) # (N, 1) - Ensure shape
        vals_arr = np.array(self.vals, dtype=np.float32).reshape(-1, 1) # (N, 1) - Ensure shape
        rewards_arr = np.array(self.rewards, dtype=np.float32) # (N,)
        dones_arr = np.array(self.dones, dtype=np.float32) # (N,)

        return states_arr, actions_arr, probs_arr, vals_arr, rewards_arr, dones_arr, batches

    def __len__(self):
        return len(self.states)


class PolicyNetwork(nn.Module):
    """Actor network for PPO. Takes basic_state as input."""
    def __init__(self, config: PPOConfig):
        super(PolicyNetwork, self).__init__()
        self.state_dim = config.state_dim # Should be CORE_STATE_DIM (8)
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, basic_state): # Takes basic_state now
        x = F.relu(self.fc1(basic_state))
        x = F.relu(self.fc2(x))
        action_mean = self.mean(x)
        action_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

    def sample(self, basic_state):
        mean, std = self.forward(basic_state)
        distribution = Normal(mean, std)
        x_t = distribution.sample()
        action_normalized = torch.tanh(x_t)
        log_prob = distribution.log_prob(x_t) - torch.log(1 - action_normalized.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action_normalized, log_prob

    def evaluate(self, basic_state, action_normalized):
        mean, std = self.forward(basic_state)
        distribution = Normal(mean, std)
        action_tanh = torch.clamp(action_normalized, -0.99999, 0.99999) # Clamp before atanh
        action_original_space = torch.atanh(action_tanh)
        log_prob = distribution.log_prob(action_original_space) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        entropy = distribution.entropy().sum(1, keepdim=True)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network for PPO. Takes basic_state as input."""
    def __init__(self, config: PPOConfig):
        super(ValueNetwork, self).__init__()
        self.state_dim = config.state_dim # Should be CORE_STATE_DIM (8)
        self.hidden_dim = config.hidden_dim
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, basic_state): # Takes basic_state now
        x = F.relu(self.fc1(basic_state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value

class PPO:
    """Proximal Policy Optimization algorithm implementation."""
    def __init__(self, config: PPOConfig, device: torch.device = None):
        self.config = config
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip
        self.n_epochs = config.n_epochs
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        self.actor = PolicyNetwork(config).to(self.device)
        self.critic = ValueNetwork(config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.memory = PPOMemory(batch_size=config.batch_size)

    def select_action(self, state: dict, evaluate=False):
        """Select action based on the *last basic state* in the trajectory."""
        # Extract the last basic state from the full trajectory
        last_basic_state_tuple = state['basic_state'] # World now provides this directly
        # Or: full_trajectory = state['full_trajectory'] # (N, feat_dim)
        # last_basic_state_tuple = tuple(full_trajectory[-1, :CORE_STATE_DIM])

        state_tensor = torch.FloatTensor(last_basic_state_tuple).to(self.device).unsqueeze(0) # (1, state_dim)

        with torch.no_grad():
            if evaluate:
                action_mean, _ = self.actor.forward(state_tensor)
                action_normalized = torch.tanh(action_mean)
            else:
                action_normalized, log_prob = self.actor.sample(state_tensor)
                value = self.critic(state_tensor)

                # Store the *individual basic state* and associated info
                self.memory.store(
                    last_basic_state_tuple,
                    action_normalized.cpu().numpy()[0, 0], # float action
                    log_prob.cpu().numpy()[0, 0], # float log_prob (already shape (1,1))
                    value.cpu().numpy()[0, 0],    # float value (already shape (1,1))
                    0, False # Placeholder reward, done - will be updated later
                )

        return action_normalized.detach().cpu().numpy()[0, 0] # Return float action

    def store_transition(self, reward, done):
        """Store reward and done flag for the last transition in memory."""
        # Only update if the last entry is still using placeholder values
        if self.memory.rewards and self.memory.rewards[-1] == 0 and self.memory.dones[-1] is False:
            self.memory.rewards[-1] = reward
            self.memory.dones[-1] = done
        elif len(self.memory.rewards) < len(self.memory.states):
            # This case should not happen if select_action always stores
             print("Warning: PPO store_transition mismatch")
             # Fallback: Append if lists are shorter
             self.memory.rewards.append(reward)
             self.memory.dones.append(done)


    def update_parameters(self):
        if len(self.memory) < self.config.batch_size:
            return None # Not enough individual transitions stored

        returns = self._compute_returns_and_advantages() # Shape (n_steps,)
        actor_losses, critic_losses, entropies = [], [], []

        # Generate batches using individual basic states stored in memory
        states_arr, actions_arr, old_log_probs_arr, values_arr, _, _, batches = self.memory.generate_batches()
        # Shapes: states(N, s_dim), actions(N, 1), old_log_probs(N, 1), values(N, 1)

        advantages = returns - values_arr.squeeze() # (N,)
        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(self.device) # (N, 1)
        if len(advantages) > 1: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device) # (N, 1)

        for _ in range(self.n_epochs):
            for batch_indices in batches:
                batch_states = torch.tensor(states_arr[batch_indices], dtype=torch.float32).to(self.device) # (batch, s_dim)
                batch_actions = torch.tensor(actions_arr[batch_indices], dtype=torch.float32).to(self.device)
                batch_old_log_probs = torch.tensor(old_log_probs_arr[batch_indices], dtype=torch.float32).to(self.device)
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate using the basic states
                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                new_values = self.critic(batch_states)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

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

        self.memory.clear() # Clear memory (individual transitions)

        return {'actor_loss': np.mean(actor_losses), 'critic_loss': np.mean(critic_losses), 'entropy': np.mean(entropies)}


    def _compute_returns_and_advantages(self):
        """Compute returns using GAE based on stored individual transitions."""
        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)
        values = np.array(self.memory.vals).squeeze() # Squeeze to (N,)

        returns = np.zeros_like(rewards)
        gae = 0.0
        last_value = 0 # Assume 0 if last state was terminal
        # If last state wasn't terminal, we need its value V(s_N)
        if not dones[-1]:
            # Get last basic state tuple and compute its value
            last_basic_state_tuple = self.memory.states[-1]
            last_state_tensor = torch.FloatTensor(last_basic_state_tuple).to(self.device).unsqueeze(0)
            with torch.no_grad():
                last_value = self.critic(last_state_tensor).cpu().numpy()[0, 0]

        # Iterate backwards
        for step in reversed(range(len(rewards))):
            # Value of next state: V(s_{t+1})
            # If t is the last step (N-1), next_value is last_value calculated above
            # Otherwise, next_value is values[t+1] (stored value of s_{t+1})
            next_value = last_value if step == len(rewards) - 1 else values[step + 1]

            delta = rewards[step] + self.gamma * next_value * (1 - int(dones[step])) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[step])) * gae
            returns[step] = gae + values[step] # Return R_t = GAE_t + V(s_t)

        return returns # Shape (N,)

    def save_model(self, path: str):
        print(f"Saving PPO model to {path}...")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'device_type': self.device.type
        }, path)

    def load_model(self, path: str):
        if not os.path.exists(path): print(f"Warn: PPO model file not found: {path}. Skip loading."); return
        print(f"Loading PPO model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"PPO model loaded successfully from {path}")


def train_ppo(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    ppo_config = config.ppo
    train_config = config.training
    world_config = config.world
    cuda_device = config.cuda_device

    log_dir = os.path.join("runs", f"ppo_traj_{int(time.time())}") # Suffix added
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Device Setup (same as before)
    if torch.cuda.is_available():
        if use_multi_gpu: print(f"Warn: Multi-GPU not standard for PPO. Using: {cuda_device}")
        device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try: torch.cuda.set_device(device); print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e: print(f"Warn: Could not set CUDA device {cuda_device}. E: {e}"); device = torch.device("cuda:0")
    else:
        device = torch.device("cpu"); print("GPU not available, using CPU.")

    agent = PPO(config=ppo_config, device=device)
    os.makedirs(train_config.models_dir, exist_ok=True)

    # Checkpoint loading (same as before)
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("ppo_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming PPO training from: {latest_model_path}")
        agent.load_model(latest_model_path)
        try:
            step_str = latest_model_path.split('_step')[-1].split('.pt')[0]
            ep_str = latest_model_path.split('_ep')[-1].split('_')[0]
            total_steps = int(step_str)
            start_episode = int(ep_str) + 1
            print(f"Resuming at step {total_steps}, episode {start_episode}")
        except (IndexError, ValueError): print("Warn: Could not parse steps/episode from filename.")
    else:
        print("\nStarting PPO training from scratch.")

    # --- Training Loop ---
    episode_rewards = []
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    update_frequency = ppo_config.steps_per_update
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training PPO", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    world = World(world_config=world_config) # Initialize once

    for episode in pbar:
        world.reset()
        state = world.encode_state() # state is dict with trajectory
        episode_reward = 0
        episode_steps = 0
        learn_steps = 0 # Track steps collected for the current update cycle

        for step_in_episode in range(train_config.max_steps):
            # Select action using last basic state from trajectory
            action_normalized = agent.select_action(state, evaluate=False) # Stores basic_state in memory

            step_start_time = time.time()
            world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            step_time = time.time() - step_start_time
            timing_metrics['env_step_time'].append(step_time)

            reward = world.reward
            next_state = world.encode_state() # Get next state dict
            done = world.done

            # Store reward and done for the *individual transition* stored earlier
            agent.store_transition(reward, done)

            state = next_state # Update world state dict for next iteration
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            learn_steps += 1

            # Check if enough *individual transitions* collected for an update
            if learn_steps >= update_frequency:
                update_start_time = time.time()
                losses = agent.update_parameters() # Uses memory with individual transitions
                update_time = time.time() - update_start_time
                if losses:
                    timing_metrics['parameter_update_time'].append(update_time)
                    writer.add_scalar('Loss/Actor', losses['actor_loss'], total_steps)
                    writer.add_scalar('Loss/Critic', losses['critic_loss'], total_steps)
                    writer.add_scalar('Policy/Entropy', losses['entropy'], total_steps)
                learn_steps = 0 # Reset counter

            if done:
                break

        # --- Logging (End of Episode) ---
        episode_rewards.append(episode_reward)
        if episode % train_config.log_frequency == 0:
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Environment_Step_ms_Avg100', np.mean(timing_metrics['env_step_time']) * 1000, total_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Parameter_Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time']) * 1000, total_steps)
            elif learn_steps > 0: writer.add_scalar('Time/Parameter_Update_ms_Avg100', 0, total_steps) # No updates yet in this cycle

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Error/Distance_EndEpisode', world.error_dist, total_steps)

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
    print(f"PPO Training finished. Total steps: {total_steps}")

    final_save_path = os.path.join(train_config.models_dir, f"ppo_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_ppo(agent=agent, config=config) # Pass full config

    return agent, episode_rewards


def evaluate_ppo(agent: PPO, config: DefaultConfig): # Takes full config
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    # Conditional visualization import
    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            import imageio.v2 as imageio
            vis_available = True; print("Visualization enabled.")
        except ImportError:
            print("Vis libs not found. Rendering disabled."); vis_available = False; eval_config.render = False
    else: vis_available = False; print("Rendering disabled by config.")

    eval_rewards = []
    success_count = 0
    all_episode_gif_paths = []

    agent.actor.eval()
    agent.critic.eval()

    print(f"\nRunning PPO Evaluation for {eval_config.num_episodes} episodes...")
    world = World(world_config=world_config) # Create world instance

    for episode in range(eval_config.num_episodes):
        world.reset()
        state = world.encode_state() # Initial state dict with trajectory
        episode_reward = 0
        episode_frames = []

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            try:
                fname = f"ppo_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config, filename=fname, collect_for_gif=True)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warn: Vis failed init state. E: {e}")

        for step in range(eval_config.max_steps):
            # PPO selects action based on last basic state in trajectory
            action_normalized = agent.select_action(state, evaluate=True)

            world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward
            next_state = world.encode_state()
            done = world.done

            if eval_config.render and vis_available:
                try:
                    fname = f"ppo_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config, filename=fname, collect_for_gif=True)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1}. E: {e}")

            state = next_state
            episode_reward += reward

            if done:
                if world.error_dist <= world_config.success_threshold:
                    success_count += 1
                    print(f"  Episode {episode+1}: Success! Step {step+1} (Err: {world.error_dist:.2f} <= Thresh {world_config.success_threshold})")
                else:
                    print(f"  Episode {episode+1}: Terminated early Step {step+1}. Final Err: {world.error_dist:.2f}")
                break

        if not done: # Max steps reached
             success = world.error_dist <= world_config.success_threshold
             status = "Success!" if success else "Failure."
             if success: success_count +=1
             print(f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps}). Final Err: {world.error_dist:.2f}. {status}")

        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        # GIF Saving
        if eval_config.render and vis_available and episode_frames:
            gif_filename = f"ppo_eval_episode_{episode+1}.gif"
            try:
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"Warn: Failed GIF save ep {episode+1}. E: {e}")
            if vis_config.delete_frames_after_gif: # Clean up frames
                 cleaned_count = 0
                 for frame in episode_frames:
                     if os.path.exists(frame):
                         try: os.remove(frame); cleaned_count += 1
                         except OSError as ose: print(f"    Warn: Could not delete PPO frame file {frame}: {ose}")

    agent.actor.train()
    agent.critic.train()

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0

    print("\n--- PPO Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Reward: {avg_eval_reward:.2f} +/- {std_eval_reward:.2f}")
    print(f"Success Rate (Error <= {world_config.success_threshold}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_episode_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering enabled but libs not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End PPO Evaluation ---\n")

    return eval_rewards, success_rate