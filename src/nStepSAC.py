import os
import time
import random
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from world import World
from world_objects import Velocity
from SAC import Actor, Critic  

class NStepReplayBuffer:
    """Experience replay buffer to store and sample n-step transitions."""

    def __init__(self, config):
        """Initialize replay buffer using configuration."""
        self.buffer = deque(maxlen=config.capacity)
        self.n_step = config.n_steps
        self.gamma = config.gamma
        self.episode_buffer = []  # Temporary buffer for episode transitions
        
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the episode buffer."""
        self.episode_buffer.append((state, action, reward, next_state, done))
        
        # If episode buffer contains at least n transitions or we reached terminal state, process n-step returns
        if len(self.episode_buffer) >= self.n_step or done:
            self._process_episode_buffer(done)
    
    def _process_episode_buffer(self, terminal_state):
        """Process episode buffer to create n-step returns."""
        # Process all transitions that can form n-step returns
        while len(self.episode_buffer) > 0:
            # Get oldest transition
            state, action, reward, _, _ = self.episode_buffer[0]
            
            # Calculate n-step return
            n_step_return = 0
            done = False
            
            # Look ahead up to n steps
            for i in range(min(self.n_step, len(self.episode_buffer))):
                n_step_return += (self.gamma**i) * self.episode_buffer[i][2]  # Add discounted reward
                if self.episode_buffer[i][4]:  # If done
                    done = True
                    break
            
            # Get next state after n steps (or terminal state)
            if done or i + 1 >= len(self.episode_buffer):
                next_state = self.episode_buffer[i][3]  # Last available next state
            else:
                next_state = self.episode_buffer[i+1][3]  # State after n steps
                
            # Store n-step transition in main buffer
            self.buffer.append((state, action, n_step_return, next_state, done))
            
            # Remove processed transition
            self.episode_buffer.pop(0)
            
            # Stop if we processed a terminal state
            if done:
                self.episode_buffer.clear()
                break
    
    def reset_episode(self):
        """Reset episode buffer (call at end of episode if not done naturally)."""
        if self.episode_buffer:
            self._process_episode_buffer(terminal_state=True)
            
    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action),
                np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class NStepSAC:
    """Soft Actor-Critic algorithm implementation with n-step returns."""

    def __init__(self, config, device=None):
        """Initialize SAC agent using configuration."""
        self.config = config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.action_scale = config.action_scale
        self.auto_tune_alpha = config.auto_tune_alpha
        self.n_steps = config.n_steps
        
        # Compute adjusted gamma for n-step returns
        self.n_step_gamma = self.gamma ** self.n_steps

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"N-Step SAC Agent using device: {self.device}, n_steps: {self.n_steps}")

        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config).to(self.device)
        self.critic_target = Critic(config).to(self.device)

        # Initialize target critic weights to match critic weights
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([config.action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)

    def select_action(self, state, evaluate=False):
        """Select action based on state."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, action_mean_squashed = self.actor.sample(state)
                # Use deterministic mean for evaluation
                action_normalized = action_mean_squashed
            else:
                action_normalized, _, _ = self.actor.sample(state)  # Sample stochastically for training
        action_scaled = action_normalized.detach().cpu().numpy()[0] * self.action_scale
        return action_scaled

    def update_parameters(self, memory, batch_size):
        """Perform a single SAC update step using n-step returns from memory."""
        if len(memory) < batch_size:
            return None

        state_batch, action_batch_scaled, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        # Rescale actions back to [-1, 1] for network input
        action_batch_normalized = torch.FloatTensor(action_batch_scaled / self.action_scale).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # --- Critic Update ---
        with torch.no_grad():
            next_action_normalized, next_log_prob, _ = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action_normalized)
            target_q_min = torch.min(target_q1, target_q2)
            target_q_entropy = target_q_min - self.alpha * next_log_prob
            
            # Use n-step returns - note that reward_batch already contains the n-step return
            # We use n_step_gamma because we're looking n steps into the future
            y = reward_batch + (1 - done_batch) * self.n_step_gamma * target_q_entropy

        current_q1, current_q2 = self.critic(state_batch, action_batch_normalized)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        for param in self.critic.parameters():
            param.requires_grad = False  # Freeze critic during actor update

        action_pi_normalized, log_prob_pi, _ = self.actor.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, action_pi_normalized)
        q_pi_min = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob_pi - q_pi_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad = True  # Unfreeze critic

        # --- Alpha Update (Entropy Tuning) ---
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Target Network Update ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }

    def save_model(self, path):
        """Save model and optimizer states."""
        print(f"Saving model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'device_type': self.device.type,
            'n_steps': self.n_steps
        }
        if self.auto_tune_alpha:
            save_dict['log_alpha_state_dict'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path):
        """Load model and optimizer states."""
        if not os.path.exists(path):
            print(f"Warning: Model file not found at {path}. Skipping loading.")
            return
        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if 'n_steps' in checkpoint:
            self.n_steps = checkpoint['n_steps']
            self.n_step_gamma = self.gamma ** self.n_steps
            print(f"Loaded n_steps: {self.n_steps}")

        if self.auto_tune_alpha and 'log_alpha_state_dict' in checkpoint:
            self.log_alpha = checkpoint['log_alpha_state_dict'].to(self.device)
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp().item()

        # Ensure target weights are updated after loading
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        print(f"Model loaded successfully from {path}")


def train_n_step_sac(config, use_multi_gpu=False):
    """Train the N-Step SAC agent."""
    sac_config = config.sac
    train_config = config.training
    buffer_config = config.replay_buffer
    world_config = config.world
    pf_config = config.particle_filter
    cuda_device = config.cuda_device 

    # Create logs directory for TensorBoard
    log_dir = os.path.join("runs", f"sac_training_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("To view logs, run: tensorboard --logdir=runs")

    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            device = torch.device("cuda")
        else:
            # Use the configured CUDA device
            device = torch.device(cuda_device)
            print(f"Using device: {device}")
            if 'cuda' in cuda_device and not cuda_device == 'cuda':
                device_idx = int(cuda_device.split(':')[1])
                if 0 <= device_idx < torch.cuda.device_count():
                    print(f"GPU: {torch.cuda.get_device_name(device_idx)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU for training")

    agent = NStepSAC(config=sac_config, device=device)
    memory = NStepReplayBuffer(config=buffer_config)

    os.makedirs(train_config.models_dir, exist_ok=True)

    episode_rewards = []
    all_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': []}
    
    pbar = tqdm(range(1, train_config.num_episodes + 1), desc="Training", unit="episode")

    for episode in pbar:
        # World requires both world_config and pf_config
        env = World(world_config=world_config, pf_config=pf_config)
        state = env.encode_state()
        episode_reward = 0
        episode_steps = 0

        for step in range(train_config.max_steps):
            action_scaled = agent.select_action(state, evaluate=False)
            action_obj = Velocity(x=action_scaled[0], y=action_scaled[1], z=0.0)
            env.step(action_obj, training=True)
            reward = env.reward
            next_state = env.encode_state()
            done = env.done
            
            # Add transition to n-step buffer
            memory.push(state, action_scaled, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            if done:
                break

        # Make sure we process any remaining transitions in the episode buffer
        memory.reset_episode()

        num_updates = episode_steps
        episode_avg_losses = {'critic_loss': 0, 'actor_loss': 0, 'alpha': 0}
        update_count = 0
        if len(memory) >= train_config.batch_size:
            for _ in range(num_updates):
                losses = agent.update_parameters(memory, train_config.batch_size)
                if losses:
                    episode_avg_losses['critic_loss'] += losses['critic_loss']
                    episode_avg_losses['actor_loss'] += losses['actor_loss']
                    episode_avg_losses['alpha'] = losses['alpha']
                    update_count += 1
            if update_count > 0:
                episode_avg_losses['critic_loss'] /= update_count
                episode_avg_losses['actor_loss'] /= update_count
                all_losses['critic_loss'].append(episode_avg_losses['critic_loss'])
                all_losses['actor_loss'].append(episode_avg_losses['actor_loss'])
                all_losses['alpha'].append(episode_avg_losses['alpha'])

        episode_rewards.append(episode_reward)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Steps/Episode', episode_steps, episode)
        writer.add_scalar('Error/Distance', env.error_dist, episode)
        
        if update_count > 0:
            writer.add_scalar('Loss/Critic', episode_avg_losses['critic_loss'], episode)
            writer.add_scalar('Loss/Actor', episode_avg_losses['actor_loss'], episode)
            writer.add_scalar('Alpha/Value', episode_avg_losses['alpha'], episode)

        if episode % 10 == 0:
            lookback = min(10, episode)
            avg_reward = np.mean(episode_rewards[-lookback:])
            writer.add_scalar('Reward/Average_10', avg_reward, episode)
            pbar_postfix = {'avg_reward': f'{avg_reward:.2f}'}
            if update_count > 0:
                pbar_postfix['crit_loss'] = f"{episode_avg_losses['critic_loss']:.3f}"
                pbar_postfix['act_loss'] = f"{episode_avg_losses['actor_loss']:.3f}"
                pbar_postfix['alpha'] = f"{episode_avg_losses['alpha']:.3f}"
            pbar.set_postfix(pbar_postfix)

        if episode % train_config.save_interval == 0:
            save_path = os.path.join(train_config.models_dir, f"sac_ep{episode}.pt")
            agent.save_model(save_path)

    pbar.close()
    writer.close()
    return agent, episode_rewards


def evaluate_n_step_sac(agent, config):
    """Evaluate the trained N-Step SAC agent."""
    # This function can use the same evaluation logic as regular SAC
    # since the difference is only in training
    from SAC import evaluate_sac
    return evaluate_sac(agent, config)