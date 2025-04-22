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
from configs import DefaultConfig, ReplayBufferConfig, SACConfig, TrainingConfig, WorldConfig
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, List, Dict, Any

# Import the vectorized environment
try:
    from vec_env import SubprocVecEnv, make_env
    VEC_ENV_AVAILABLE = True
except ImportError:
    print("Warning: vec_env.py not found or cloudpickle not installed. Parallel environment execution disabled.")
    VEC_ENV_AVAILABLE = False
    SubprocVecEnv = None # Define as None if import fails


# --- ReplayBuffer, Actor, Critic (Keep previous versions) ---
# (Paste the versions from the previous response that handle trajectory state)
class ReplayBuffer:
    """Experience replay buffer storing full state trajectories."""

    def __init__(self, config: ReplayBufferConfig, world_config: WorldConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim

    def push(self, state, action, reward, next_state, done):
        """ Add a new experience to memory. """
        # state and next_state are dicts containing 'full_trajectory'
        trajectory = state['full_trajectory']
        next_trajectory = next_state['full_trajectory']
        # Ensure action is float
        if isinstance(action, (np.ndarray, list)): action = action[0]
        # Ensure reward is float
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        if trajectory.shape != (self.trajectory_length, self.feature_dim):
             # print(f"Warning: Pushing trajectory with incorrect shape {trajectory.shape}. Expected ({self.trajectory_length}, {self.feature_dim})")
             return # Skip if incorrect shape
        if next_trajectory.shape != (self.trajectory_length, self.feature_dim):
             # print(f"Warning: Pushing next_trajectory with incorrect shape {next_trajectory.shape}. Expected ({self.trajectory_length}, {self.feature_dim})")
             return # Skip if incorrect shape

        self.buffer.append((trajectory, float(action), float(reward), next_trajectory, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        """Sample a batch of experiences from memory."""
        if len(self.buffer) < batch_size:
            # print(f"Warning: Not enough samples in buffer ({len(self.buffer)}) to sample batch size ({batch_size}).")
            return None

        batch = random.sample(self.buffer, batch_size)
        trajectory, action, reward, next_trajectory, done = zip(*batch)

        # Convert to numpy arrays
        trajectory_arr = np.array(trajectory, dtype=np.float32)
        action_arr = np.array(action, dtype=np.float32).reshape(-1, 1) # Shape (batch, 1)
        reward_arr = np.array(reward, dtype=np.float32) # Shape (batch,)
        next_trajectory_arr = np.array(next_trajectory, dtype=np.float32)
        done_arr = np.array(done, dtype=np.float32) # Shape (batch,)

        return (trajectory_arr, action_arr, reward_arr, next_trajectory_arr, done_arr)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Policy network (Actor) for SAC, optionally with RNN."""

    def __init__(self, config: SACConfig, world_config: WorldConfig):
        super(Actor, self).__init__()
        self.config = config
        self.world_config = world_config
        self.use_rnn = config.use_rnn
        self.state_dim = config.state_dim # Dimension of basic state
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # RNN processes basic states
            if config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                   num_layers=config.rnn_num_layers, batch_first=True)
            elif config.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            else:
                raise ValueError(f"Unsupported RNN type: {config.rnn_type}")
            mlp_input_dim = config.rnn_hidden_size
        else:
            # MLP uses only the last basic state
            mlp_input_dim = self.state_dim
            self.rnn = None

        self.layers = nn.ModuleList()
        current_dim = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.mean = nn.Linear(current_dim, self.action_dim)
        self.log_std = nn.Linear(current_dim, self.action_dim)

        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, state_trajectory: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass through the network."""
        # state_trajectory shape: (batch, seq_len, feature_dim)
        # feature_dim includes basic_state, prev_action, prev_reward

        if self.use_rnn:
            # Extract basic state sequence for RNN
            basic_state_sequence = state_trajectory[:, :, :self.state_dim] # Shape: (batch, seq_len, state_dim)
            # RNN processing
            rnn_output, next_hidden_state = self.rnn(basic_state_sequence, hidden_state)
            # Use the output corresponding to the last time step
            mlp_input = rnn_output[:, -1, :] # Shape: (batch, rnn_hidden_size)
        else:
            # Use only the last basic state from the trajectory
            mlp_input = state_trajectory[:, -1, :self.state_dim] # Shape: (batch, state_dim)
            next_hidden_state = None

        x = mlp_input
        for layer in self.layers:
            x = layer(x)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, next_hidden_state

    def sample(self, state_trajectory: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Sample action (normalized) from the policy distribution."""
        mean, log_std, next_hidden_state = self.forward(state_trajectory, hidden_state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        action_normalized = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t) - torch.log(1 - action_normalized.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action_normalized, log_prob, torch.tanh(mean), next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        """Return initial hidden state for RNN."""
        if not self.use_rnn:
            return None
        h_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.config.rnn_type == 'lstm':
            c_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
            return (h_zeros, c_zeros)
        elif self.config.rnn_type == 'gru':
            return h_zeros
        return None

class Critic(nn.Module):
    """Q-function network (Critic) for SAC, optionally with RNN."""

    def __init__(self, config: SACConfig, world_config: WorldConfig):
        super(Critic, self).__init__()
        self.config = config
        self.world_config = world_config
        self.use_rnn = config.use_rnn
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # RNN processes basic states
            rnn_cell = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU

            self.rnn1 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            self.rnn2 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            mlp_input_dim = config.rnn_hidden_size + self.action_dim # MLP takes final RNN state + action
        else:
            # MLP uses only the last basic state + action
            mlp_input_dim = self.state_dim + self.action_dim
            self.rnn1, self.rnn2 = None, None

        # Q1 architecture
        self.q1_layers = nn.ModuleList()
        q1_mlp_input = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.q1_layers.append(nn.Linear(q1_mlp_input, hidden_dim))
            self.q1_layers.append(nn.ReLU())
            q1_mlp_input = hidden_dim
        self.q1_out = nn.Linear(q1_mlp_input, 1)

        # Q2 architecture
        self.q2_layers = nn.ModuleList()
        q2_mlp_input = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.q2_layers.append(nn.Linear(q2_mlp_input, hidden_dim))
            self.q2_layers.append(nn.ReLU())
            q2_mlp_input = hidden_dim
        self.q2_out = nn.Linear(q2_mlp_input, 1)

    def forward(self, state_trajectory: torch.Tensor, action: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass returning both Q-values."""
        # state_trajectory shape: (batch, seq_len, feature_dim)
        # action shape: (batch, action_dim) - Action corresponding to the *last* state in the trajectory

        if self.use_rnn:
            basic_state_sequence = state_trajectory[:, :, :self.state_dim] # (batch, seq_len, state_dim)
            h1_in, h2_in = None, None
            if isinstance(hidden_state, tuple) and len(hidden_state) == 2:
                 h1_in, h2_in = hidden_state

            rnn_out1, next_h1 = self.rnn1(basic_state_sequence, h1_in)
            rnn_out2, next_h2 = self.rnn2(basic_state_sequence, h2_in)
            next_hidden_state = (next_h1, next_h2)

            # Use final RNN hidden state and concatenate with action
            mlp_input1 = torch.cat([rnn_out1[:, -1, :], action], dim=1)
            mlp_input2 = torch.cat([rnn_out2[:, -1, :], action], dim=1)
        else:
            # Use last basic state and concatenate with action
            last_basic_state = state_trajectory[:, -1, :self.state_dim] # (batch, state_dim)
            mlp_input1 = torch.cat([last_basic_state, action], dim=1)
            mlp_input2 = torch.cat([last_basic_state, action], dim=1)
            next_hidden_state = None

        # Q1 calculation
        x1 = mlp_input1
        for layer in self.q1_layers:
            x1 = layer(x1)
        q1 = self.q1_out(x1)

        # Q2 calculation
        x2 = mlp_input2
        for layer in self.q2_layers:
            x2 = layer(x2)
        q2 = self.q2_out(x2)

        return q1, q2, next_hidden_state

    def q1_forward(self, state_trajectory: torch.Tensor, action: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """ Forward pass for Q1 only. """
        if self.use_rnn:
            basic_state_sequence = state_trajectory[:, :, :self.state_dim]
            h1_in = hidden_state[0] if isinstance(hidden_state, tuple) and hidden_state is not None else None
            rnn_out1, next_h1 = self.rnn1(basic_state_sequence, h1_in)
            next_hidden_state = (next_h1, None) # Return only Q1's hidden state progression
            mlp_input1 = torch.cat([rnn_out1[:, -1, :], action], dim=1)
        else:
            last_basic_state = state_trajectory[:, -1, :self.state_dim]
            mlp_input1 = torch.cat([last_basic_state, action], dim=1)
            next_hidden_state = None

        x1 = mlp_input1
        for layer in self.q1_layers:
            x1 = layer(x1)
        q1 = self.q1_out(x1)
        return q1, next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        """Return initial hidden state tuple (h1, h2) for RNNs."""
        if not self.use_rnn: return None
        h_zeros = lambda: torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.config.rnn_type == 'lstm':
            c_zeros = lambda: torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
            h1 = (h_zeros(), c_zeros())
            h2 = (h_zeros(), c_zeros())
            return (h1, h2)
        elif self.config.rnn_type == 'gru':
            h1 = h_zeros()
            h2 = h_zeros()
            return (h1, h2)
        return None


class SAC:
    """Soft Actor-Critic algorithm implementation with trajectory states."""

    def __init__(self, config: SACConfig, world_config: WorldConfig, device: torch.device = None):
        self.config = config
        self.world_config = world_config # Need world config for trajectory info
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha
        self.use_rnn = config.use_rnn
        self.trajectory_length = world_config.trajectory_length
        self.state_dim = config.state_dim # Basic state dim
        self.action_dim = config.action_dim

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"SAC Agent using device: {self.device}")
        if self.use_rnn:
             print(f"SAC Agent using RNN: Type={config.rnn_type}, Hidden={config.rnn_hidden_size}, Layers={config.rnn_num_layers}, SeqLen={self.trajectory_length}")
        else:
             print(f"SAC Agent using MLP (Processing last state of {self.trajectory_length}-step trajectory)")

        self.actor = Actor(config, world_config).to(self.device)
        self.critic = Critic(config, world_config).to(self.device)
        self.critic_target = Critic(config, world_config).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for target_param in self.critic_target.parameters():
            target_param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device) # Keep as tensor

    # Keep single select_action for evaluation/non-parallel training
    def select_action(self, state: dict, actor_hidden_state: Optional[Tuple] = None, evaluate: bool = False) -> Tuple[float, Optional[Tuple]]:
        """Select action for a single environment state."""
        state_trajectory = state['full_trajectory']
        state_tensor = torch.FloatTensor(state_trajectory).to(self.device).unsqueeze(0) # (1, N, feat_dim)

        with torch.no_grad():
            if evaluate:
                _, _, action_mean_squashed, next_actor_hidden_state = self.actor.sample(state_tensor, actor_hidden_state)
                action_normalized = action_mean_squashed
            else:
                action_normalized, _, _, next_actor_hidden_state = self.actor.sample(state_tensor, actor_hidden_state)

        action_normalized_float = action_normalized.detach().cpu().numpy()[0, 0]
        return action_normalized_float, next_actor_hidden_state

    # Add select_action_batch for parallel environments
    def select_action_batch(self, state_trajectory_batch: np.ndarray, actor_hidden_state_batch: Optional[Tuple] = None, evaluate: bool = False) -> Tuple[np.ndarray, Optional[Tuple]]:
        """Select actions for a batch of environment states."""
        # state_trajectory_batch shape: (batch_size, N, feature_dim)
        state_tensor = torch.FloatTensor(state_trajectory_batch).to(self.device)

        with torch.no_grad():
            if evaluate:
                _, _, action_mean_squashed, next_actor_hidden_state_batch = self.actor.sample(state_tensor, actor_hidden_state_batch)
                actions_normalized = action_mean_squashed
            else:
                actions_normalized, _, _, next_actor_hidden_state_batch = self.actor.sample(state_tensor, actor_hidden_state_batch)

        # actions_normalized shape: (batch_size, action_dim)
        actions_normalized_np = actions_normalized.detach().cpu().numpy()
        # If action_dim is 1, squeeze it to (batch_size,) for vec_env step compatibility
        if self.action_dim == 1:
             actions_normalized_np = actions_normalized_np.squeeze(axis=1)

        return actions_normalized_np, next_actor_hidden_state_batch

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform a single SAC update step using a batch of trajectories."""
        if len(memory) < batch_size:
            return None

        sampled_batch = memory.sample(batch_size)
        if sampled_batch is None:
            # print("Warning: Failed to sample batch from replay buffer.")
            return None

        # Shapes: state/next_state (b, N, feat_dim), action (b, 1), reward/done (b,)
        state_batch, action_batch_normalized, reward_batch, next_state_batch, done_batch = sampled_batch

        # Move batch to device
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch_normalized = torch.FloatTensor(action_batch_normalized).to(self.device) # (b, 1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1) # (b, 1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1) # (b, 1)

        # Get initial hidden states for RNNs if used (for the batch)
        initial_actor_hidden = self.actor.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_hidden = self.critic.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_target_hidden = self.critic_target.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None

        # --- Critic Update ---
        with torch.no_grad():
            # Get next action and log prob using the *next* state trajectory
            next_action_normalized, next_log_prob, _, _ = self.actor.sample(next_state_batch, initial_actor_hidden)

            # Get target Q values using the *next* state trajectory and *next* action
            target_q1, target_q2, _ = self.critic_target(next_state_batch, next_action_normalized, initial_critic_target_hidden)
            target_q_min = torch.min(target_q1, target_q2)

            # Calculate target value Y
            current_alpha = self.log_alpha.exp().item() # Get current alpha value
            target_q_entropy = target_q_min - current_alpha * next_log_prob
            y = reward_batch + (1.0 - done_batch) * self.gamma * target_q_entropy

        # --- Calculate Current Q values ---
        # Use the *current* state trajectory and the *actual action taken*
        current_q1, current_q2, _ = self.critic(state_batch, action_batch_normalized, initial_critic_hidden)

        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Freeze critic parameters
        for param in self.critic.parameters():
            param.requires_grad = False

        # Calculate actor loss
        # Get actions and log probs for the *current* state trajectory
        action_pi_normalized, log_prob_pi, _, _ = self.actor.sample(state_batch, initial_actor_hidden)

        # Get Q-values for the current state trajectory and the *policy's* action
        # Pass initial_critic_hidden again - assumes stateless forward pass within batch
        q1_pi, q2_pi, _ = self.critic(state_batch, action_pi_normalized, initial_critic_hidden)
        q_pi_min = torch.min(q1_pi, q2_pi)

        current_alpha = self.log_alpha.exp().item() # Get current alpha value
        actor_loss = (current_alpha * log_prob_pi - q_pi_min).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for param in self.critic.parameters():
            param.requires_grad = True

        # --- Alpha Update ---
        if self.auto_tune_alpha:
            # Use log_prob_pi from actor update (detached)
            alpha_loss = -(self.log_alpha * (log_prob_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item() # Update Python value

        # --- Target Network Update (Soft Update) ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }

    # save_model and load_model remain the same as previous version

    def save_model(self, path: str):
        print(f"Saving SAC model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'device_type': self.device.type
        }
        if self.auto_tune_alpha:
            save_dict['log_alpha'] = self.log_alpha # Save tensor directly
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            print(f"Warning: SAC model file not found at {path}. Skipping loading.")
            return
        print(f"Loading SAC model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if self.auto_tune_alpha and 'log_alpha' in checkpoint:
            # Load log_alpha tensor
            self.log_alpha = checkpoint['log_alpha'].to(self.device)
            if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True)
            # Re-initialize optimizer with the loaded tensor
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            try:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except ValueError as e:
                 print(f"Warning: Could not load SAC alpha optimizer state: {e}. Reinitializing.")
                 self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            self.alpha = self.log_alpha.exp().item()
        elif not self.auto_tune_alpha:
            # If alpha is fixed, update self.alpha based on saved log_alpha if available,
            # otherwise it keeps the initial config value.
             if 'log_alpha' in checkpoint:
                 self.log_alpha = checkpoint['log_alpha'].to(self.device)
                 self.alpha = self.log_alpha.exp().item()


        # Ensure target network sync after loading
        self.critic_target.load_state_dict(self.critic.state_dict())
        for target_param in self.critic_target.parameters():
            target_param.requires_grad = False

        self.actor.train()
        self.critic.train()
        self.critic_target.train() # Set target to train mode, though grads are off

        print(f"SAC model loaded successfully from {path}")


# --- Updated Training Function ---
def train_sac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True, num_envs: int = 4):
    """ Train SAC using parallel environments """
    sac_config = config.sac
    train_config = config.training
    buffer_config = config.replay_buffer
    world_config = config.world
    cuda_device = config.cuda_device

    if num_envs > 1 and not VEC_ENV_AVAILABLE:
        print("Warning: Requested parallel environments but SubprocVecEnv not available. Running with 1 environment.")
        num_envs = 1
    elif num_envs <= 0:
        print("Warning: num_envs must be positive. Setting to 1.")
        num_envs = 1

    print(f"Using {num_envs} parallel environment(s).")

    learning_starts = train_config.learning_starts
    gradient_steps = train_config.gradient_steps
    train_freq = train_config.train_freq # How often to run updates per env step
    batch_size = train_config.batch_size
    log_frequency_ep = train_config.log_frequency # Log based on completed episodes
    save_interval_ep = train_config.save_interval # Save based on completed episodes

    # Device Setup
    # ... (same as before) ...
    if torch.cuda.is_available():
        if use_multi_gpu: print(f"Warn: Multi-GPU not standard for SAC. Using: {cuda_device}")
        device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try: torch.cuda.set_device(device); print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e: print(f"Warn: Could not set CUDA device {cuda_device}. E: {e}"); device = torch.device("cuda:0")
    else:
        device = torch.device("cpu"); print("GPU not available, using CPU.")


    # --- Initialization ---
    log_dir = os.path.join("runs", f"sac_parallel_{num_envs}env_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    agent = SAC(config=sac_config, world_config=world_config, device=device)
    memory = ReplayBuffer(config=buffer_config, world_config=world_config)
    os.makedirs(train_config.models_dir, exist_ok=True)

    # --- Load Checkpoint ---
    # ... (same as before) ...
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("sac_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0
    completed_episodes_total = 0 # Track total completed episodes across all envs
    start_episode = 1 # This now represents completed episodes count start
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming training from: {latest_model_path}")
        agent.load_model(latest_model_path)
        # Try parsing total steps and completed episodes from filename (adjust if format changes)
        try:
            step_str = latest_model_path.split('_step')[-1].split('.pt')[0]
            # Assuming filename might store completed episodes now, e.g., _comp_ep
            ep_str = latest_model_path.split('_comp_ep')[-1].split('_')[0]
            total_steps = int(step_str)
            completed_episodes_total = int(ep_str)
            start_episode = completed_episodes_total + 1
            print(f"Resuming at total_steps {total_steps}, completed_episodes {completed_episodes_total}")
        except (IndexError, ValueError):
             print("Warning: Could not parse steps/episode from filename. Starting counts from 0.")
    else:
        print("\nStarting training from scratch.")


    # --- Environment Setup ---
    if num_envs > 1:
        env_fns = [make_env(world_config, seed=i) for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = World(world_config=world_config) # Fallback to single env if num_envs=1

    # --- Training Loop ---
    # Get initial states
    if isinstance(vec_env, SubprocVecEnv):
        current_states_list = vec_env.reset() # List of state dicts
    else: # Single env case
        current_states_list = [vec_env.reset()] # Wrap in list

    # Initialize hidden states for the batch
    actor_hidden_states = agent.actor.get_initial_hidden_state(num_envs, device) if agent.use_rnn else None

    # Tracking across environments
    current_episode_rewards = np.zeros(num_envs)
    current_episode_steps = np.zeros(num_envs, dtype=int)
    all_completed_episode_rewards = deque(maxlen=100) # Store rewards of completed episodes for avg log

    # Progress bar based on total completed episodes
    pbar = tqdm(total=train_config.num_episodes, desc="Training SAC (Parallel)", initial=completed_episodes_total)

    training_start_time = time.time()

    while completed_episodes_total < train_config.num_episodes:

        # Check if learning should start
        if total_steps < learning_starts:
            # Take random actions if not learning yet (or use initial policy evaluation)
            # For simplicity, let's use the policy but don't update
            actions_normalized_batch, next_actor_hidden_states = agent.select_action_batch(
                 np.stack([s['full_trajectory'] for s in current_states_list]), # Stack trajectories
                 actor_hidden_state_batch=actor_hidden_states,
                 evaluate=True # Use mean action before learning starts? Or sample? Sample is fine.
            )
        else:
            # Select actions for the batch using the agent
            actions_normalized_batch, next_actor_hidden_states = agent.select_action_batch(
                 np.stack([s['full_trajectory'] for s in current_states_list]),
                 actor_hidden_state_batch=actor_hidden_states,
                 evaluate=False
            )

        # Step environments
        step_start_time = time.time()
        if isinstance(vec_env, SubprocVecEnv):
             # actions_normalized_batch shape should be (num_envs,) if action_dim=1
             next_states_list, rewards_batch, dones_batch, infos_list = vec_env.step(actions_normalized_batch)
        else: # Single env case
             vec_env.step(actions_normalized_batch[0], training=True, terminal_step=(current_episode_steps[0] >= train_config.max_steps -1))
             next_states_list = [vec_env.encode_state()]
             rewards_batch = np.array([vec_env.reward])
             dones_batch = np.array([vec_env.done])
             infos_list = [{'error_dist': vec_env.error_dist}] # Wrap info
        step_time = time.time() - step_start_time
        # Note: Can't easily average step time across parallel envs here

        # Process results and store experience
        for i in range(num_envs):
            # Ensure state/next_state are dicts for pushing
            current_state_dict = current_states_list[i]
            next_state_dict = next_states_list[i]

            # Push individual transitions (or rather, the state/next_state trajectories)
            memory.push(current_state_dict, actions_normalized_batch[i], rewards_batch[i], next_state_dict, dones_batch[i])

            current_episode_rewards[i] += rewards_batch[i]
            current_episode_steps[i] += 1
            total_steps += 1 # Increment total steps based on interactions

            if dones_batch[i]:
                completed_episodes_total += 1
                pbar.update(1) # Update progress bar for each completed episode

                # Log completed episode reward immediately
                writer.add_scalar('Reward/CompletedEpisodeRaw', current_episode_rewards[i], completed_episodes_total)
                writer.add_scalar('Steps/CompletedEpisode', current_episode_steps[i], completed_episodes_total)
                writer.add_scalar('Error/Distance_EpisodeEnd', infos_list[i].get('error_dist', float('inf')), completed_episodes_total)

                all_completed_episode_rewards.append(current_episode_rewards[i]) # Store for averaging

                # Reset tracking for this env
                current_episode_rewards[i] = 0.0
                current_episode_steps[i] = 0

                # Reset RNN hidden state for this env index
                if agent.use_rnn and actor_hidden_states is not None:
                    if isinstance(actor_hidden_states, tuple): # LSTM state is (h, c)
                        actor_hidden_states[0][:, i, :] = 0.0 # Reset h state for env i
                        actor_hidden_states[1][:, i, :] = 0.0 # Reset c state for env i
                    else: # GRU state is just h
                        actor_hidden_states[:, i, :] = 0.0 # Reset h state for env i

                # Check saving interval based on completed episodes
                if completed_episodes_total % save_interval_ep == 0:
                    save_path = os.path.join(
                        train_config.models_dir, f"sac_comp_ep{completed_episodes_total}_step{total_steps}.pt")
                    agent.save_model(save_path)

        # Update current states and hidden states for the next iteration
        current_states_list = next_states_list
        if agent.use_rnn:
            actor_hidden_states = next_actor_hidden_states

        # Agent updates (triggered less frequently relative to total steps)
        if total_steps >= learning_starts and total_steps % (train_freq * num_envs) == 0: # Adjust freq based on num_envs? Or keep based on total steps? Keep based on total_steps for now.
             # if total_steps >= learning_starts: # Check every step if learning started
             if total_steps >= learning_starts and total_steps % train_freq == 0:
                 # Perform gradient steps
                 updates_made = 0
                 update_times = []
                 for _ in range(gradient_steps * num_envs): # Scale gradient steps? Or keep 1 per trigger? Let's keep simple first.
                 # for _ in range(gradient_steps):
                      if len(memory) >= batch_size:
                           update_start_time = time.time()
                           losses = agent.update_parameters(memory, batch_size)
                           update_times.append(time.time() - update_start_time)
                           if losses:
                                # Log losses immediately? Or average over the gradient steps? Average is better.
                                writer.add_scalar('Loss/Critic_Step', losses['critic_loss'], total_steps)
                                writer.add_scalar('Loss/Actor_Step', losses['actor_loss'], total_steps)
                                writer.add_scalar('Alpha/Value_Step', losses['alpha'], total_steps)
                                updates_made += 1
                      else:
                           break # Stop if buffer becomes too small

                 if updates_made > 0 and update_times:
                      avg_update_time = np.mean(update_times)
                      writer.add_scalar('Time/Parameter_Update_ms', avg_update_time * 1000, total_steps)


        # Periodic Logging (based on total_steps or completed_episodes)
        # Using completed_episodes for logging interval seems more consistent across runs
        if completed_episodes_total > 0 and completed_episodes_total % (log_frequency_ep * num_envs) == 0 : # Log less frequently
             if all_completed_episode_rewards:
                 avg_reward_100 = np.mean(all_completed_episode_rewards)
                 writer.add_scalar('Reward/Average_100_Completed', avg_reward_100, completed_episodes_total)

             # Log other stats like buffer size, total steps
             writer.add_scalar('Progress/Total_Steps', total_steps, completed_episodes_total)
             writer.add_scalar('Progress/Buffer_Size', len(memory), completed_episodes_total)
             writer.add_scalar('Progress/Completed_Episodes', completed_episodes_total, total_steps)
             current_time = time.time()
             elapsed_time = current_time - training_start_time
             steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
             writer.add_scalar('Performance/Steps_Per_Second', steps_per_sec, total_steps)

             # Update pbar postfix less frequently
             if all_completed_episode_rewards:
                 pbar.set_postfix({'avg_rew_100': f'{avg_reward_100:.2f}', 'steps': total_steps, 'SPS': f'{steps_per_sec:.0f}'})


    # --- End of Training ---
    pbar.close()
    writer.close()
    if isinstance(vec_env, SubprocVecEnv): vec_env.close() # Close parallel environments

    print(f"Training finished. Total steps: {total_steps}, Completed episodes: {completed_episodes_total}")
    final_save_path = os.path.join(
        train_config.models_dir, f"sac_final_comp_ep{completed_episodes_total}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         # Evaluation usually runs on a single environment
         evaluate_sac(agent=agent, config=config)

    # Return agent and rewards (maybe return the deque of completed rewards)
    return agent, list(all_completed_episode_rewards)


# --- Evaluation Function (evaluate_sac) ---
# Keep the previous version of evaluate_sac, as evaluation is typically done
# sequentially on a single environment instance.
def evaluate_sac(agent: SAC, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    # --- Conditional visualization import ---
    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            import imageio.v2 as imageio
            vis_available = True; print("Visualization enabled.")
        except ImportError:
            print("Visualization libraries not found. Rendering disabled."); vis_available = False; eval_config.render = False
    else: vis_available = False; print("Rendering disabled by config.")

    eval_rewards = []
    success_count = 0
    all_episode_gif_paths = []

    agent.actor.eval()
    agent.critic.eval() # Also set critic to eval mode

    print(f"\nRunning SAC Evaluation for {eval_config.num_episodes} episodes...")
    # Create a single world instance for evaluation
    world = World(world_config=world_config)

    for episode in range(eval_config.num_episodes):
        world.reset() # Reset world at the start of each episode
        state = world.encode_state() # Initial state dict
        episode_reward = 0
        episode_frames = []

        # Initialize hidden state for single env eval
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=agent.device) if agent.use_rnn else None

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            try:
                fname = f"sac_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config=vis_config, filename=fname, collect_for_gif=True)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
                elif initial_frame_file: print(f"Warn: Initial frame path returned but not found: {initial_frame_file}")
            except Exception as e: print(f"Warn: Vis failed init state. E: {e}")

        for step in range(eval_config.max_steps):
            # Use single select_action for evaluation
            action_normalized, next_actor_hidden_state = agent.select_action(
                state, actor_hidden_state=actor_hidden_state, evaluate=True
            )

            # Step the single environment
            world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward # Get reward from world (usually 0 in eval)
            next_state = world.encode_state()
            done = world.done

            if eval_config.render and vis_available:
                try:
                    fname = f"sac_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config=vis_config, filename=fname, collect_for_gif=True)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                    elif frame_file: print(f"Warn: Frame path returned but not found: {frame_file} step {step+1}")
                except Exception as e: print(f"Warn: Vis failed step {step+1}. E: {e}")

            state = next_state
            if agent.use_rnn: actor_hidden_state = next_actor_hidden_state
            episode_reward += reward # Accumulate reward (might be 0)

            if done:
                if world.error_dist <= world_config.success_threshold:
                    success_count += 1
                    print(f"  Episode {episode+1}: Success! Step {step+1} (Err: {world.error_dist:.2f} <= Thresh {world_config.success_threshold})")
                else:
                    print(f"  Episode {episode+1}: Terminated Step {step+1}. Final Err: {world.error_dist:.2f}, Collision: {world._calculate_range_measurement(world.agent.location, world.true_landmark.location) < world_config.collision_threshold}")
                break

        if not done: # Reached max steps
             success = world.error_dist <= world_config.success_threshold
             status = "Success!" if success else "Failure."
             if success: success_count +=1
             print(f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps}). Final Err: {world.error_dist:.2f}. {status}")

        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        # --- GIF Saving ---
        if eval_config.render and vis_available and episode_frames:
            gif_filename = f"sac_eval_episode_{episode+1}.gif"
            try:
                print(f"  Saving GIF for episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"  Warn: Failed GIF save ep {episode+1}. E: {e}")
            if vis_config.delete_frames_after_gif: # Clean up frames
                 cleaned_count = 0
                 for frame in episode_frames:
                     if os.path.exists(frame):
                         try: os.remove(frame); cleaned_count += 1
                         except OSError as ose: print(f"    Warn: Could not delete SAC frame file {frame}: {ose}")

    agent.actor.train() # Set actor back to train mode
    agent.critic.train() # Set critic back to train mode

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0

    print("\n--- SAC Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Reward: {avg_eval_reward:.2f} +/- {std_eval_reward:.2f}")
    print(f"Success Rate (Error <= {world_config.success_threshold}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_episode_gif_paths:
        print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering enabled but libs not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End SAC Evaluation ---\n")

    return eval_rewards, success_rate

