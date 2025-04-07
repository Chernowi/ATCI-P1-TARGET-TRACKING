import os
import numpy np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque
from tqdm import tqdm
from world import World
from world_objects import Velocity
from visualization import visualize_world


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), 
                np.array(reward, dtype=np.float32), 
                np.array(next_state), np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Policy network that outputs action mean and log_std."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)  # Constrain to [-1, 1]
        
        # Calculate log probability, adding correction for tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """Q-function network that estimates expected returns."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        # Q1
        x1 = torch.cat([state, action], 1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)
        
        # Q2
        x2 = torch.cat([state, action], 1)
        x2 = F.relu(self.fc3(x2))
        x2 = F.relu(self.fc4(x2))
        q2 = self.q2(x2)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        x1 = torch.cat([state, action], 1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)
        return q1


class SAC:
    """Soft Actor-Critic algorithm implementation."""
    
    def __init__(self, state_dim, action_dim, action_scale=1.0, lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, auto_tune_alpha=True, device=None):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_scale = action_scale
        self.auto_tune_alpha = auto_tune_alpha
        
        # Enhanced device detection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # Copy weights from critic to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        if auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
            
        return action.detach().cpu().numpy()[0] * self.action_scale
    
    def update_parameters(self, memory, batch_size):
        if len(memory) < batch_size:
            return
            
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
        
        # Move all data to device at once for better parallelism
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            # Select action according to policy
            next_action, next_log_prob, _ = self.actor.sample(next_state_batch)
            
            # Compute target Q-value
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
            
        # Compute current Q-values
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Freeze critic networks for actor update
        for param in self.critic.parameters():
            param.requires_grad = False
            
        # Compute actor loss
        action, log_prob, _ = self.actor.sample(state_batch)
        q1, q2 = self.critic(state_batch, action)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - q).mean()
        
        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Unfreeze critic networks
        for param in self.critic.parameters():
            param.requires_grad = True
            
        # Update alpha if auto-tuning
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            
        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }
    
    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'device': self.device.type  # Store device type for proper loading
        }, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)  # Load to the correct device
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def train_sac(env, num_episodes=1000, max_steps=100, batch_size=256, replay_buffer_size=1000000,
              save_interval=100, models_dir="sac_models", success_threshold=2.0, use_multi_gpu=False):
    """Train the SAC agent on the given environment."""
    
    # Get state and action dimensions
    state_dim = 8  # agent_x, agent_y, agent_vx, agent_vy, landmark_x, landmark_y, landmark_depth, current_range
    action_dim = 2  # vx, vy (ignoring vz dimension)
    action_scale = 2.0  # Scale actions to reasonable velocity range
    
    # Enhanced GPU initialization
    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            device = torch.device("cuda")
            # Set up for DataParallel if needed
            # Note: DataParallel is only useful for model inference/forward passes with large batches
            # Most RL algorithms don't benefit much from DataParallel since gradient updates are often sequential
        else:
            device = torch.device("cuda")
            print(f"Using 1 GPU for training: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU for training")
    
    # Initialize agent and replay buffer
    agent = SAC(state_dim, action_dim, action_scale=action_scale, device=device)
    memory = ReplayBuffer(replay_buffer_size)
    
    # Create directory for saving models
    os.makedirs(models_dir, exist_ok=True)
    
    # Track episode rewards
    episode_rewards = []
    
    # Create progress bar for episodes
    pbar = tqdm(range(1, num_episodes + 1), desc="Training", unit="episode")
    
    for episode in pbar:
        env = World(dt=1.0, success_threshold=success_threshold)  # Reset environment with success threshold
        episode_reward = 0
        state = env.encode_state()
        
        # Set up episode buffer for more efficient batch processing
        episode_data = []
        
        for step in range(max_steps):
            # Select action
            action_tensor = agent.select_action(state)
            
            # Apply to environment - only use vx and vy, set vz to 0
            action = Velocity(action_tensor[0], action_tensor[1], 0.0)
            env.step(action)
            
            # Get reward and next state
            reward = env.reward
            next_state = env.encode_state()
            done = env.done or (step == max_steps - 1)  # Episode ends if success or max steps reached
            
            # Store transition
            episode_data.append((state, action_tensor, reward, next_state, done))
            
            # Prepare for next step
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Add all transitions to replay buffer at once
        for transition in episode_data:
            memory.push(*transition)
            
        # Perform multiple updates after episode completed
        num_updates = min(len(episode_data), 10)  # Set a reasonable number of updates per episode
        for _ in range(num_updates):
            if len(memory) >= batch_size:
                agent.update_parameters(memory, batch_size)
                
        episode_rewards.append(episode_reward)
        
        # Update progress bar with reward info
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            success_count = sum(1 for i in range(episode-10, episode) if i >= 0 and episode_rewards[i] > 0)
            success_rate = success_count/min(10, episode)
            pbar.set_postfix({
                'avg_reward': f'{avg_reward:.2f}', 
                'success_rate': f'{success_rate:.2f}'
            })
        
        # Save model
        if episode % save_interval == 0:
            agent.save_model(f"{models_dir}/sac_ep{episode}.pt")
    
    return agent, episode_rewards


def evaluate_sac(agent, env, num_episodes=5, max_steps=100, render=True, success_threshold=2.0):
    """Evaluate the trained SAC agent."""
    
    eval_rewards = []
    success_count = 0
    
    # Ensure agent is in evaluation mode
    agent.actor.eval()
    agent.critic.eval()
    
    for episode in range(num_episodes):
        env = World(dt=1.0, success_threshold=success_threshold)  # Reset environment with success threshold
        episode_reward = 0
        state = env.encode_state()
        
        # Visualization setup
        if render:
            from visualization import reset_trajectories, visualize_world, save_gif
            snapshot_dir = "world_snapshots"
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # Reset trajectories and frames for this episode
            reset_trajectories()
            
            # Visualize initial state
            visualize_world(env, filename=f"eval_ep{episode+1}_frame_000_initial.png")
        
        for step in range(max_steps):
            # Select action (deterministic evaluation)
            with torch.no_grad():
                action_tensor = agent.select_action(state, evaluate=True)
            
            # Apply to environment - only use vx and vy, set vz to 0
            action = Velocity(action_tensor[0], action_tensor[1], 0.0)
            
            # Apply to environment
            env.step(action, training=False)
            
            # Visualize if requested
            if render:
                visualize_world(env, filename=f"eval_ep{episode+1}_frame_{step+1:03d}.png")
                
            # Get reward and next state
            reward = env.reward
            state = env.encode_state()
            episode_reward += reward
            
            # Check for early termination
            if env.done:
                success_count += 1
                print(f"Success! Found landmark with error {env.error_dist:.2f} < threshold {success_threshold}")
                break
            
        eval_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}: Total Reward: {episode_reward:.2f}, Error: {env.error_dist:.2f}")
        
        # Create a GIF for this episode if rendering
        if render:
            from visualization import save_gif
            save_gif(f"eval_episode_{episode+1}.gif", duration=0.2, delete_frames=True)
    
    # Return agent to training mode if needed
    agent.actor.train()
    agent.critic.train()
    
    # Create a combined GIF of all episodes if multiple were evaluated
    if render and num_episodes > 1:
        from visualization import create_gif_from_files
        create_gif_from_files("eval_episode_*.gif", "evaluation_all_episodes.gif", duration=1.0)
    
    print(f"Average Evaluation Reward: {np.mean(eval_rewards):.2f}")
    print(f"Success Rate: {success_count/num_episodes:.2f} ({success_count}/{num_episodes})")
    print(f"GIFs saved in the '{os.path.abspath('world_snapshots')}' directory.")
    return eval_rewards
