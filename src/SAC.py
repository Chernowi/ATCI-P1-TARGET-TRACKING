import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import time
from collections import deque
from tqdm import tqdm
from world import World
from world_objects import Velocity
from visualization import visualize_world, reset_trajectories, save_gif, create_gif_from_files
from configs import DefaultConfig, ReplayBufferConfig, SACConfig
# Add TensorBoard import
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""

    def __init__(self, config: ReplayBufferConfig):
        """Initialize replay buffer using configuration."""
        self.buffer = deque(maxlen=config.capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

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


class Actor(nn.Module):
    """Policy network (Actor) for SAC."""

    def __init__(self, config: SACConfig):
        """Initialize actor network using configuration."""
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Linear(config.hidden_dim, config.action_dim)

        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, state):
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        """Sample action from the policy distribution."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick (for backpropagation)
        x_t = normal.rsample()
        action = torch.tanh(x_t)  # Squash action to [-1, 1] range

        # Calculate log probability with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # Return squashed action, log_prob, and squashed mean (for eval)
        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """Q-function network (Critic) for SAC."""

    def __init__(self, config: SACConfig):
        """Initialize critic network using configuration."""
        super(Critic, self).__init__()
        state_dim = config.state_dim
        action_dim = config.action_dim
        hidden_dim = config.hidden_dim

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture (to mitigate overestimation)
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        """Forward pass returning both Q-values."""
        sa = torch.cat([state, action], 1)  # Concatenate state and action

        # Q1 path
        x1 = F.relu(self.fc1(sa))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)

        # Q2 path
        x2 = F.relu(self.fc3(sa))
        x2 = F.relu(self.fc4(x2))
        q2 = self.q2(x2)

        return q1, q2

    def q1_forward(self, state, action):
        """Forward pass returning only Q1 value (potentially useful for TD3)."""
        sa = torch.cat([state, action], 1)
        x1 = F.relu(self.fc1(sa))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)
        return q1


class SAC:
    """Soft Actor-Critic algorithm implementation."""

    def __init__(self, config: SACConfig, device: torch.device = None):
        """Initialize SAC agent using configuration."""
        self.config = config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.action_scale = config.action_scale
        self.auto_tune_alpha = config.auto_tune_alpha

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"SAC Agent using device: {self.device}")

        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config).to(self.device)
        self.critic_target = Critic(config).to(self.device)

        # Initialize target critic weights to match critic weights
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False  # Target networks are not trained directly

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.lr)

        if self.auto_tune_alpha:
            self.target_entropy = - \
                torch.prod(torch.Tensor(
                    [config.action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
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
                action_normalized, _, _ = self.actor.sample(
                    state)  # Sample stochastically for training
        action_scaled = action_normalized.detach().cpu().numpy()[
            0] * self.action_scale
        return action_scaled

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform a single SAC update step using a batch from memory."""
        if len(memory) < batch_size:
            return None

        state_batch, action_batch_scaled, reward_batch, next_state_batch, done_batch = memory.sample(
            batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        # Rescale actions back to [-1, 1] for network input
        action_batch_normalized = torch.FloatTensor(
            action_batch_scaled / self.action_scale).to(self.device)
        reward_batch = torch.FloatTensor(
            reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # --- Critic Update ---
        with torch.no_grad():
            next_action_normalized, next_log_prob, _ = self.actor.sample(
                next_state_batch)
            target_q1, target_q2 = self.critic_target(
                next_state_batch, next_action_normalized)
            target_q_min = torch.min(target_q1, target_q2)
            target_q_entropy = target_q_min - self.alpha * next_log_prob
            y = reward_batch + (1 - done_batch) * self.gamma * \
                target_q_entropy  # Bellman target

        current_q1, current_q2 = self.critic(
            state_batch, action_batch_normalized)
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
            alpha_loss = -(self.log_alpha * (log_prob_pi +
                           self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Target Network Update ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }

    def save_model(self, path: str):
        """Save model and optimizer states."""
        print(f"Saving model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'device_type': self.device.type
        }
        if self.auto_tune_alpha:
            save_dict['log_alpha_state_dict'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        """Load model and optimizer states."""
        if not os.path.exists(path):
            print(
                f"Warning: Model file not found at {path}. Skipping loading.")
            return
        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(
            checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(
            checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])

        if self.auto_tune_alpha and 'log_alpha_state_dict' in checkpoint:
            self.log_alpha = checkpoint['log_alpha_state_dict'].to(self.device)
            self.alpha_optimizer.load_state_dict(
                checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp().item()

        # Ensure target weights are updated after loading
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        print(f"Model loaded successfully from {path}")


def train_sac(config: DefaultConfig, use_multi_gpu: bool = False):
    """Train the SAC agent on the environment defined by the config."""
    sac_config = config.sac
    train_config = config.training
    buffer_config = config.replay_buffer
    world_config = config.world
    pf_config = config.particle_filter
    cuda_device = config.cuda_device  # Get the configured CUDA device

    # Create logs directory for TensorBoard
    log_dir = os.path.join("runs", f"sac_training_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("To view logs, run: tensorboard --logdir=runs")

    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training (Note: SAC updates are often sequential)")
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

    agent = SAC(config=sac_config, device=device)
    memory = ReplayBuffer(config=buffer_config)

    os.makedirs(train_config.models_dir, exist_ok=True)

    episode_rewards = []
    all_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': []}
    
    # Add timing metric tracking
    timing_metrics = {
        'env_step_time': [],
        'parameter_update_time': []
    }
    
    pbar = tqdm(range(1, train_config.num_episodes + 1),
                desc="Training", unit="episode")

    for episode in pbar:
        # World requires both world_config and pf_config
        env = World(world_config=world_config, pf_config=pf_config)
        state = env.encode_state()
        episode_reward = 0
        episode_steps = 0
        episode_transitions = []
        
        # Timing metrics for this episode
        episode_step_times = []
        episode_param_update_times = []

        for step in range(train_config.max_steps):
            action_scaled = agent.select_action(state, evaluate=False)
            action_obj = Velocity(
                x=action_scaled[0], y=action_scaled[1], z=0.0)
            
            # Measure environment step time
            step_start_time = time.time()
            env.step(action_obj, training=True)
            step_time = time.time() - step_start_time
            episode_step_times.append(step_time)
            
            reward = env.reward
            next_state = env.encode_state()
            done = env.done
            episode_transitions.append(
                (state, action_scaled, reward, next_state, done))
            state = next_state
            episode_reward += reward
            episode_steps += 1
            if done:
                break

        for transition in episode_transitions:
            memory.push(*transition)

        num_updates = episode_steps
        episode_avg_losses = {'critic_loss': 0, 'actor_loss': 0, 'alpha': 0}
        update_count = 0
        if len(memory) >= train_config.batch_size:
            for _ in range(num_updates):
                # Measure parameter update time
                update_start_time = time.time()
                losses = agent.update_parameters(
                    memory, train_config.batch_size)
                update_time = time.time() - update_start_time
                episode_param_update_times.append(update_time)
                
                if losses:
                    episode_avg_losses['critic_loss'] += losses['critic_loss']
                    episode_avg_losses['actor_loss'] += losses['actor_loss']
                    episode_avg_losses['alpha'] = losses['alpha']
                    update_count += 1
            if update_count > 0:
                episode_avg_losses['critic_loss'] /= update_count
                episode_avg_losses['actor_loss'] /= update_count
                all_losses['critic_loss'].append(
                    episode_avg_losses['critic_loss'])
                all_losses['actor_loss'].append(
                    episode_avg_losses['actor_loss'])
                all_losses['alpha'].append(episode_avg_losses['alpha'])

        episode_rewards.append(episode_reward)
        
        # Calculate average timing metrics for this episode
        if episode_step_times:
            avg_step_time = np.mean(episode_step_times)
            timing_metrics['env_step_time'].append(avg_step_time)
            writer.add_scalar('Time/Environment_Step_ms', avg_step_time * 1000, episode)
        
        if episode_param_update_times:
            avg_param_update_time = np.mean(episode_param_update_times)
            timing_metrics['parameter_update_time'].append(avg_param_update_time)
            writer.add_scalar('Time/Parameter_Update_ms', avg_param_update_time * 1000, episode)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Steps/Episode', episode_steps, episode)
        writer.add_scalar('Error/Distance', env.error_dist, episode)
        writer.add_scalar('Time/Particle_Filter_ms', env.pf_update_time * 1000, episode)
        
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
            save_path = os.path.join(
                train_config.models_dir, f"sac_ep{episode}.pt")
            agent.save_model(save_path)

        # Log hyperparameters to tensorboard
        writer.add_hparams(
            {"n_steps": sac_config.n_steps, "lr": sac_config.lr, "gamma": sac_config.gamma},
            {"hparam/avg_reward": np.mean(episode_rewards[-10:])}
        )

    pbar.close()
    writer.close()
    return agent, episode_rewards


def evaluate_sac(agent: SAC, config: DefaultConfig):
    """Evaluate the trained SAC agent using evaluation configuration."""
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization
    pf_config = config.particle_filter  # Needed for World initialization

    eval_rewards = []
    success_count = 0
    all_episode_gif_paths = []

    agent.actor.eval()
    agent.critic.eval()

    print(f"\nRunning Evaluation for {eval_config.num_episodes} episodes...")
    for episode in range(eval_config.num_episodes):
        # World requires both world_config and pf_config
        env = World(world_config=world_config, pf_config=pf_config)
        state = env.encode_state()
        episode_reward = 0
        episode_frames = []

        if eval_config.render:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            initial_frame_file = visualize_world(
                world=env,
                vis_config=vis_config,
                filename=f"eval_ep{episode+1}_frame_000_initial.png",
                collect_for_gif=True  # This flag might be redundant if frame list is managed here
            )
            if initial_frame_file:
                episode_frames.append(initial_frame_file)

        for step in range(eval_config.max_steps):
            action_scaled = agent.select_action(state, evaluate=True)
            action_obj = Velocity(
                x=action_scaled[0], y=action_scaled[1], z=0.0)
            env.step(action_obj, training=False)
            reward = env.reward
            next_state = env.encode_state()
            done = env.done

            if eval_config.render:
                frame_file = visualize_world(
                    world=env,
                    vis_config=vis_config,
                    filename=f"eval_ep{episode+1}_frame_{step+1:03d}.png",
                    collect_for_gif=True  # This flag might be redundant
                )
                if frame_file:
                    episode_frames.append(frame_file)

            state = next_state
            episode_reward += reward

            if done:
                success_count += 1
                print(
                    f"  Episode {episode+1}: Success! Found landmark at step {step+1} (Error: {env.error_dist:.2f} < threshold {world_config.success_threshold})")
                break

        if not done:
            print(
                f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps} reached). Final Error: {env.error_dist:.2f}")

        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        if eval_config.render and episode_frames:
            gif_filename = f"eval_episode_{episode+1}.gif"
            gif_path = save_gif(
                output_filename=gif_filename,
                vis_config=vis_config,
                frame_paths=episode_frames,
                delete_frames=vis_config.delete_frames_after_gif
            )
            if gif_path:
                all_episode_gif_paths.append(gif_path)

    agent.actor.train()
    agent.critic.train()

    if eval_config.render and len(all_episode_gif_paths) > 1:
        # Skipping combined GIF creation from other GIFs as it's complex/lossy
        print(
            f"\nCombined GIF creation skipped. View individual episode GIFs: {all_episode_gif_paths}")

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    success_rate = success_count / \
        eval_config.num_episodes if eval_config.num_episodes > 0 else 0
    print(f"\n--- Evaluation Summary ---")
    print(f"Average Evaluation Reward: {avg_eval_reward:.2f}")
    print(
        f"Success Rate: {success_rate:.2f} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render:
        print(
            f"Individual episode GIFs saved in the '{os.path.abspath(vis_config.save_dir)}' directory.")
    print(f"--- End Evaluation ---")

    return eval_rewards
