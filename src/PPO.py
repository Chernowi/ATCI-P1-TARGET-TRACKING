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
from world_objects import Velocity
from visualization import visualize_world, reset_trajectories, save_gif
from configs import DefaultConfig, PPOConfig, TrainingConfig
from torch.utils.tensorboard import SummaryWriter


class PPOMemory:
    """Memory buffer for PPO algorithm to store trajectories."""
    
    def __init__(self, batch_size=64):
        self.states = []
        self.actions = []
        self.probs = []  # Log probs of actions
        self.vals = []   # Value estimates
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def store(self, state, action, probs, vals, reward, done):
        """Store a transition in memory."""
        self.states.append(state)
        self.actions.append(action)
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
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        states = [self.states[idx] for idx in indices]
        actions = [self.actions[idx] for idx in indices]
        probs = [self.probs[idx] for idx in indices]
        vals = [self.vals[idx] for idx in indices]
        rewards = [self.rewards[idx] for idx in indices]
        dones = [self.dones[idx] for idx in indices]
        
        return states, actions, probs, vals, rewards, dones, batches
    
    def __len__(self):
        return len(self.states)


class PolicyNetwork(nn.Module):
    """Actor network for PPO that outputs action means and standard deviations."""
    
    def __init__(self, config: PPOConfig):
        super(PolicyNetwork, self).__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.action_scale = config.action_scale
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
        action = torch.tanh(x_t)  # Squash to [-1, 1]
        
        # Calculate log probability with tanh correction
        log_prob = distribution.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state, action):
        """Evaluate log probability and entropy for a given state-action pair."""
        mean, std = self.forward(state)
        distribution = Normal(mean, std)
        
        # Un-squash action from [-1, 1] for proper log prob calculation
        action_tanh = action
        action = torch.atanh(torch.clamp(action_tanh, -0.999, 0.999))
        
        log_prob = distribution.log_prob(action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
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
        self.action_scale = config.action_scale
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"PPO Agent using device: {self.device}")
        
        # Initialize networks
        self.actor = PolicyNetwork(config).to(self.device)
        self.critic = ValueNetwork(config).to(self.device)
        
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # Set up memory buffer
        self.memory = PPOMemory(batch_size=config.batch_size)
        
    def select_action(self, state, evaluate=False):
        """Select action based on state."""
        state_tuple = state['basic_state']
        state = torch.FloatTensor(state_tuple).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            if evaluate:
                action_mean, _ = self.actor.forward(state)
                action_normalized = torch.tanh(action_mean)
            else:
                action_normalized, log_prob = self.actor.sample(state)
                value = self.critic(state)
                
                # Store in memory for later update
                self.memory.store(
                    state_tuple, 
                    action_normalized.cpu().numpy()[0], 
                    log_prob.cpu().numpy()[0],
                    value.cpu().numpy()[0],
                    0, False  # Reward and done will be stored later
                )
                
        action_scaled = action_normalized.detach().cpu().numpy()[0] * self.action_scale
        return action_scaled
    
    def store_transition(self, reward, done):
        """Store reward and done flag for the last transition."""
        if len(self.memory.rewards) < len(self.memory.states):
            self.memory.rewards[-1] = reward
            self.memory.dones[-1] = done
    
    def update_parameters(self):
        """Update PPO parameters based on collected trajectories."""
        if len(self.memory) < self.config.batch_size:
            return None
            
        # Calculate returns and advantages
        returns = self._compute_returns_and_advantages()
        
        actor_losses, critic_losses, entropies = [], [], []
        
        # Optimize policy and value networks for n_epochs
        for _ in range(self.n_epochs):
            # Generate batches from memory
            states, actions, old_log_probs, values, rewards, dones, batches = self.memory.generate_batches()
            
            advantages = [returns[i] - values[i] for i in range(len(returns))]
            advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device)
            
            # Normalize advantages for stable training
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Process each batch
            states_array = np.array(states)
            actions_array = np.array(actions)
            old_log_probs_array = np.array(old_log_probs)
            returns_array = np.array(returns)
            values_array = np.array(values)
            
            for batch_indices in batches:
                # Extract batch data
                batch_states = torch.tensor(states_array[batch_indices], dtype=torch.float32).to(self.device)
                batch_actions = torch.tensor(actions_array[batch_indices], dtype=torch.float32).to(self.device)
                batch_old_log_probs = torch.tensor(old_log_probs_array[batch_indices], dtype=torch.float32).to(self.device)
                batch_returns = torch.tensor(returns_array[batch_indices], dtype=torch.float32).to(self.device)
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions and calculate ratio
                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                new_values = self.critic(batch_states)
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                critic_loss = F.mse_loss(new_values, batch_returns.unsqueeze(1))
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Optimize networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                # Optional: gradient clipping
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
        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)
        values = np.array([v[0] for v in self.memory.vals])
        
        returns = np.zeros_like(rewards)
        gae = 0
        
        # Calculate GAE
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                # For the last step, we don't have a next value, use the last value
                next_value = values[step]
            else:
                next_value = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_value * (1 - int(dones[step])) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[step])) * gae
            returns[step] = gae + values[step]
        
        return returns
    
    def save_model(self, path: str):
        """Save model and optimizer states."""
        print(f"Saving model to {path}...")
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
            print(f"Warning: Model file not found at {path}. Skipping loading.")
            return
        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded successfully from {path}")


def train_ppo(config: DefaultConfig, use_multi_gpu: bool = False):
    """Train the PPO agent on the environment defined by the config."""
    ppo_config = config.ppo
    train_config = config.training
    world_config = config.world
    cuda_device = config.cuda_device
    
    # Create logs directory for TensorBoard
    log_dir = os.path.join("runs", f"ppo_training_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            device = torch.device("cuda")
        else:
            device = torch.device(cuda_device)
            print(f"Using device: {device}")
            if 'cuda' in cuda_device and not cuda_device == 'cuda':
                try:
                    device_idx = int(cuda_device.split(':')[1])
                    if 0 <= device_idx < torch.cuda.device_count():
                        print(f"GPU: {torch.cuda.get_device_name(device_idx)}")
                except (IndexError, ValueError):
                    print(f"Warning: Invalid CUDA device string '{cuda_device}'. Using default CUDA device.")
                    device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU for training")
    
    agent = PPO(config=ppo_config, device=device)
    os.makedirs(train_config.models_dir, exist_ok=True)
    
    episode_rewards = []
    
    timing_metrics = {
        'env_step_time': [],
        'parameter_update_time': []
    }
    
    total_steps = 0
    pbar = tqdm(range(1, train_config.num_episodes + 1),
                desc="Training", unit="episode")
    
    # Define update frequency in steps
    update_frequency = ppo_config.steps_per_update
    
    for episode in pbar:
        env = World(world_config=world_config)
        state = env.encode_state()
        episode_reward = 0
        episode_steps = 0
        
        episode_step_times = []
        episode_param_update_times = []
        
        for step_in_episode in range(train_config.max_steps):
            # Select action
            action_scaled = agent.select_action(state, evaluate=False)
            action_obj = Velocity(x=action_scaled[0], y=action_scaled[1], z=0.0)
            
            # Take step in environment
            step_start_time = time.time()
            env.step(action_obj, training=True)
            step_time = time.time() - step_start_time
            episode_step_times.append(step_time)
            
            reward = env.reward
            next_state = env.encode_state()
            done = env.done
            
            # Store reward and done flag for the most recent transition
            agent.store_transition(reward, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Check if it's time to update
            if total_steps % update_frequency == 0:
                update_start_time = time.time()
                losses = agent.update_parameters()
                update_time = time.time() - update_start_time
                episode_param_update_times.append(update_time)
                
                if losses:
                    writer.add_scalar('Loss/Actor', losses['actor_loss'], total_steps)
                    writer.add_scalar('Loss/Critic', losses['critic_loss'], total_steps)
                    writer.add_scalar('Policy/Entropy', losses['entropy'], total_steps)
            
            if done:
                break
            
        # End of episode logging
        episode_rewards.append(episode_reward)
        
        if episode % train_config.log_frequency == 0:
            # Log training metrics
            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Error/Distance_EndEpisode', env.error_dist, total_steps)
            
            # Log time metrics
            if episode_step_times:
                avg_step_time = np.mean(episode_step_times)
                timing_metrics['env_step_time'].append(avg_step_time)
                writer.add_scalar('Time/Environment_Step_ms', avg_step_time * 1000, total_steps)
            
            if episode_param_update_times:
                avg_param_update_time = np.mean(episode_param_update_times)
                timing_metrics['parameter_update_time'].append(avg_param_update_time)
                writer.add_scalar('Time/Parameter_Update_ms', avg_param_update_time * 1000, total_steps)
            
            # Log average reward
            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)
        
        # Update progress bar
        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar.set_postfix({'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps})
        
        # Save model
        if episode % train_config.save_interval == 0:
            save_path = os.path.join(train_config.models_dir, f"ppo_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)
    
    pbar.close()
    writer.close()
    print(f"Training finished. Total steps: {total_steps}")
    
    # Save final model
    final_save_path = os.path.join(train_config.models_dir, f"ppo_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)
    
    return agent, episode_rewards


def evaluate_ppo(agent: PPO, config: DefaultConfig):
    """Evaluate the trained PPO agent using evaluation configuration."""
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization
    
    eval_rewards = []
    success_count = 0
    all_episode_gif_paths = []
    
    agent.actor.eval()
    agent.critic.eval()
    
    print(f"\nRunning Evaluation for {eval_config.num_episodes} episodes...")
    for episode in range(eval_config.num_episodes):
        env = World(world_config=world_config)
        state = env.encode_state()
        episode_reward = 0
        episode_frames = []
        
        if eval_config.render:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            try:
                initial_frame_file = visualize_world(
                    world=env,
                    vis_config=vis_config,
                    filename=f"eval_ep{episode+1}_frame_000_initial.png",
                    collect_for_gif=True
                )
                if initial_frame_file:
                    episode_frames.append(initial_frame_file)
            except Exception as e:
                print(f"Warning: Visualization failed for initial state. Error: {e}")
        
        for step in range(eval_config.max_steps):
            action_scaled = agent.select_action(state, evaluate=True)
            action_obj = Velocity(x=action_scaled[0], y=action_scaled[1], z=0.0)
            env.step(action_obj, training=False)
            reward = env.reward
            next_state = env.encode_state()
            done = env.done
            
            if eval_config.render:
                try:
                    frame_file = visualize_world(
                        world=env,
                        vis_config=vis_config,
                        filename=f"eval_ep{episode+1}_frame_{step+1:03d}.png",
                        collect_for_gif=True
                    )
                    if frame_file:
                        episode_frames.append(frame_file)
                except Exception as e:
                    print(f"Warning: Visualization failed for step {step+1}. Error: {e}")
            
            state = next_state
            episode_reward += reward
            
            if done:
                if env.error_dist <= world_config.success_threshold:
                    success_count += 1
                    print(
                        f"  Episode {episode+1}: Success! Found landmark at step {step+1} (Error: {env.error_dist:.2f} <= threshold {world_config.success_threshold})")
                else:
                    print(
                        f"  Episode {episode+1}: Terminated early at step {step+1} (Not success - e.g., OOB). Final Error: {env.error_dist:.2f}"
                    )
                break
        
        if not done:
            print(
                f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps} reached). Final Error: {env.error_dist:.2f}")
        
        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")
        
        if eval_config.render and episode_frames:
            gif_filename = f"eval_episode_{episode+1}.gif"
            try:
                gif_path = save_gif(
                    output_filename=gif_filename,
                    vis_config=vis_config,
                    frame_paths=episode_frames,
                    delete_frames=vis_config.delete_frames_after_gif
                )
                if gif_path:
                    all_episode_gif_paths.append(gif_path)
            except Exception as e:
                print(f"Warning: Failed to create or save GIF for episode {episode+1}. Error: {e}")
                if vis_config.delete_frames_after_gif:
                    for frame in episode_frames:
                        if os.path.exists(frame):
                            try:
                                os.remove(frame)
                            except OSError:
                                pass
    
    agent.actor.train()
    agent.critic.train()
    
    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0
    print("\n--- Evaluation Summary ---")
    print(f"Average Evaluation Reward: {avg_eval_reward:.2f}")
    print(
        f"Success Rate (reaching threshold): {success_rate:.2f} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and all_episode_gif_paths:
        print(
            f"Individual episode GIFs saved in the '{os.path.abspath(vis_config.save_dir)}' directory.")
    elif eval_config.render:
        print("Rendering was enabled, but no GIFs were successfully created.")
    print("--- End Evaluation ---")
    
    return eval_rewards