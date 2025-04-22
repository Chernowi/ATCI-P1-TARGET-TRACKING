import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import time
import math
from collections import deque
from tqdm import tqdm
from world import World
from configs import DefaultConfig, ReplayBufferConfig, TSACConfig, TrainingConfig, CORE_STATE_DIM, CORE_ACTION_DIM, TRAJECTORY_REWARD_DIM
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, List

# --- Replay Buffer (Identical to SAC's new buffer) ---

class ReplayBuffer:
    """Experience replay buffer storing full state trajectories."""

    def __init__(self, config: ReplayBufferConfig, world_config: WorldConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim

    def push(self, state, action, reward, next_state, done):
        """ Add a new experience to memory. """
        trajectory = state['full_trajectory']
        next_trajectory = next_state['full_trajectory']
        if isinstance(action, (np.ndarray, list)): action = action[0]
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        if trajectory.shape != (self.trajectory_length, self.feature_dim):
             print(f"Warn: Push traj shape {trajectory.shape} != ({self.trajectory_length}, {self.feature_dim})")
             return
        if next_trajectory.shape != (self.trajectory_length, self.feature_dim):
             print(f"Warn: Push next_traj shape {next_trajectory.shape} != ({self.trajectory_length}, {self.feature_dim})")
             return

        self.buffer.append((trajectory, float(action), float(reward), next_trajectory, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        """Sample a batch of experiences from memory."""
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        trajectory, action, reward, next_trajectory, done = zip(*batch)

        trajectory_arr = np.array(trajectory, dtype=np.float32)
        action_arr = np.array(action, dtype=np.float32).reshape(-1, 1)
        reward_arr = np.array(reward, dtype=np.float32)
        next_trajectory_arr = np.array(next_trajectory, dtype=np.float32)
        done_arr = np.array(done, dtype=np.float32)

        return (trajectory_arr, action_arr, reward_arr, next_trajectory_arr, done_arr)

    def __len__(self):
        return len(self.buffer)

# --- Positional Encoding ---

class PositionalEncoding(nn.Module):
    """ Standard sinusoidal positional encoding """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        return x + self.pe[:, :x.size(1)]


# --- T-SAC Actor (MLP based, uses last state) ---

class Actor(nn.Module):
    """Policy network (Actor) for T-SAC. Uses the last basic_state from trajectory."""

    def __init__(self, config: TSACConfig, world_config: WorldConfig):
        super(Actor, self).__init__()
        self.config = config
        self.world_config = world_config
        self.use_layer_norm = config.use_layer_norm_actor
        self.state_dim = config.state_dim # Basic state dim (8)
        self.action_dim = config.action_dim
        # self.trajectory_length = world_config.trajectory_length # Not directly used by MLP actor

        self.layers = nn.ModuleList()
        mlp_input_dim = self.state_dim # Input is the last basic state
        for i, hidden_dim in enumerate(config.hidden_dims):
            linear_layer = nn.Linear(mlp_input_dim, hidden_dim)
            nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
            if linear_layer.bias is not None:
                 fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
                 bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                 nn.init.uniform_(linear_layer.bias, -bound, bound)
            self.layers.append(linear_layer)
            if self.use_layer_norm:
                 self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            mlp_input_dim = hidden_dim

        self.mean = nn.Linear(config.hidden_dims[-1], self.action_dim)
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)

        self.log_std = nn.Linear(config.hidden_dims[-1], self.action_dim)
        nn.init.xavier_uniform_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, 0)

        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, state_trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using only the last basic state."""
        # state_trajectory shape: (batch, seq_len, feature_dim)
        last_basic_state = state_trajectory[:, -1, :self.state_dim] # (batch, state_dim)

        x = last_basic_state
        for layer in self.layers:
            x = layer(x)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state_trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (normalized) from the policy distribution."""
        mean, log_std = self.forward(state_trajectory)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        action_normalized = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t) - torch.log(1 - action_normalized.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action_normalized, log_prob, torch.tanh(mean)


# --- T-SAC Critic (Transformer-based) ---

class TransformerCritic(nn.Module):
    """Q-function network (Critic) for T-SAC using a Transformer."""

    def __init__(self, config: TSACConfig, world_config: WorldConfig):
        super(TransformerCritic, self).__init__()
        self.config = config
        self.world_config = world_config
        self.state_dim = config.state_dim # Basic state dim (8)
        self.action_dim = config.action_dim
        self.embedding_dim = config.embedding_dim
        self.sequence_length = world_config.trajectory_length # Use N from world

        self.state_embed = nn.Linear(self.state_dim, self.embedding_dim)
        self.action_embed = nn.Linear(self.action_dim, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(self.embedding_dim, max_len=self.sequence_length + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=config.transformer_n_heads,
            dim_feedforward=config.transformer_hidden_dim,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_n_layers)

        # Output head predicts Q-values for the sequence
        self.q_head = nn.Linear(self.embedding_dim, 1)

    def _generate_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """ Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, initial_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer critic.
        Args:
            initial_state (torch.Tensor): Initial basic state s_t. Shape: (batch, state_dim)
            action_sequence (torch.Tensor): Sequence of actions a_t to a_{t+N-1}. Shape: (batch, sequence_length, action_dim)
        Returns:
            torch.Tensor: Sequence of Q-value predictions Q(s_t, a_t..k) for k=t..t+N-1. Shape: (batch, sequence_length, 1)
        """
        batch_size, seq_len, _ = action_sequence.shape
        device = initial_state.device

        if seq_len != self.sequence_length:
             print(f"Warning: Critic received action sequence length {seq_len}, expected {self.sequence_length}")
             # Handle mismatch? Pad? Truncate? Error? For now, proceed but maybe problematic.
             # This shouldn't happen if buffer sampling is correct.

        # Embed initial state and action sequence
        state_emb = self.state_embed(initial_state).unsqueeze(1) # (batch, 1, embed_dim)
        action_emb = self.action_embed(action_sequence)          # (batch, seq_len, embed_dim)

        # Concatenate: [s_t_emb, a_t_emb, a_{t+1}_emb, ..., a_{t+N-1}_emb]
        transformer_input_emb = torch.cat([state_emb, action_emb], dim=1) # (batch, seq_len + 1, embed_dim)

        # Add positional encoding
        transformer_input_pos = self.pos_encoder(transformer_input_emb)

        # Create causal mask for the transformer
        mask = self._generate_mask(seq_len + 1, device)

        # Pass through transformer
        transformer_output = self.transformer_encoder(transformer_input_pos, mask=mask) # (batch, seq_len + 1, embed_dim)

        # Extract outputs corresponding to action inputs (ignore output for s_t)
        # Output at index k (1 to seq_len) corresponds to input a_{t+k-1}
        action_transformer_output = transformer_output[:, 1:, :] # (batch, seq_len, embed_dim)

        # Predict Q-values from transformer outputs
        q_predictions = self.q_head(action_transformer_output) # (batch, seq_len, 1)
        return q_predictions


# --- T-SAC Agent ---

class TSAC:
    """Transformer-based Soft Actor-Critic algorithm implementation."""

    def __init__(self, config: TSACConfig, world_config: WorldConfig, device: torch.device = None):
        self.config = config
        self.world_config = world_config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha
        self.sequence_length = world_config.trajectory_length # N
        self.action_dim = config.action_dim

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"T-SAC Agent using device: {self.device}")
        print(f"T-SAC using Sequence Length (N): {self.sequence_length}")

        self.actor = Actor(config, world_config).to(self.device)
        self.critic1 = TransformerCritic(config, world_config).to(self.device)
        self.critic2 = TransformerCritic(config, world_config).to(self.device)
        self.critic1_target = TransformerCritic(config, world_config).to(self.device)
        self.critic2_target = TransformerCritic(config, world_config).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.lr)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device)

    def select_action(self, state: dict, evaluate: bool = False) -> float:
        """Select action (normalized yaw change) based on the *last* state in the trajectory."""
        state_trajectory = state['full_trajectory']
        state_tensor = torch.FloatTensor(state_trajectory).to(self.device).unsqueeze(0) # (1, N, feat_dim)

        with torch.no_grad():
            if evaluate:
                _, _, action_mean_squashed = self.actor.sample(state_tensor)
                action_normalized = action_mean_squashed
            else:
                action_normalized, _, _ = self.actor.sample(state_tensor)

        action_normalized_float = action_normalized.detach().cpu().numpy()[0, 0]
        return action_normalized_float

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform a single T-SAC update step using a batch of experiences."""
        sampled_batch = memory.sample(batch_size)
        if sampled_batch is None:
            return None

        # state_batch shape: (b, N, feat_dim) -> Trajectory ending at s_{N-1}
        # action_batch shape: (b, 1) -> Action a_{N-1} taken after s_{N-1}
        # reward_batch shape: (b,) -> Reward r_{N-1} received after a_{N-1}
        # next_state_batch shape: (b, N, feat_dim) -> Trajectory ending at s_{N}
        # done_batch shape: (b,) -> Done flag for state s_{N}
        state_batch, action_batch_current, reward_batch, next_state_batch, done_batch = sampled_batch

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        # action_batch_current = torch.FloatTensor(action_batch_current).to(self.device) # Action a_{N-1}
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1) # (b, 1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1) # (b, 1)

        # Extract information needed for Transformer Critic
        # Initial state s_t is the *first* basic state in the state_batch trajectory
        initial_state_critic = state_batch[:, 0, :self.config.state_dim] # Shape: (b, state_dim)

        # Action sequence a_t...a_{t+N-1} needs to be reconstructed/extracted
        # state_batch contains (basic_state_k, action_{k-1}, reward_{k-1}) for k=t..t+N-1
        # We need actions a_t to a_{t+N-1}
        # action_{t+k-1} is stored at index k in the state_batch trajectory
        action_sequence = state_batch[:, 1:, self.config.state_dim:self.config.state_dim+self.config.action_dim] # (b, N-1, action_dim) actions a_t..a_{t+N-2}
        # The last action a_{t+N-1} is action_batch_current
        action_sequence = torch.cat([action_sequence, torch.FloatTensor(action_batch_current).to(self.device).unsqueeze(1)], dim=1) # (b, N, action_dim)

        # --- Critic Update ---
        with torch.no_grad():
            # Calculate N-step targets G(n+1) for n=0..N-1
            # Get policy actions and log probs for the *next* states
            # next_state_batch contains trajectories ending at s_{t+N}
            # We need states s_{t+1} to s_{t+N}
            target_q_values_list = []
            cumulative_rewards = torch.zeros_like(reward_batch[:, 0]).to(self.device) # (b,)
            discount_factor = torch.ones_like(reward_batch[:, 0]).to(self.device) # (b,)

            # rewards_in_state_batch are r_{t-1} to r_{t+N-2}
            # We need r_t to r_{t+N-1}
            # r_{t+k-1} is at index k in state_batch
            rewards_sequence = state_batch[:, 1:, -1] # (b, N-1) rewards r_t..r_{t+N-2}
            rewards_sequence = torch.cat([rewards_sequence, reward_batch], dim=1) # (b, N) rewards r_t..r_{t+N-1}

            # dones_in_state_batch correspond to states s_t..s_{t+N-1}
            # We need dones for s_{t+1}..s_{t+N}
            # Need to extract done flags corresponding to next_state_batch elements? No, buffer gives done for the state *after* the final action.

            # Let's recalculate target using Bellman expectation over the sequence length N
            # Target for Q(s_t, a_t...a_{t+N-1}) is E[ sum(gamma^k * r_{t+k}) + gamma^N * V(s_{t+N}) ]
            # V(s) = min Q_target(s, a') - alpha * log_pi(a'|s)

            # State s_{t+N} is the last basic state in the next_state_batch trajectory
            s_t_plus_N = next_state_batch[:, -1, :self.config.state_dim] # (b, state_dim)

            # Action and log_prob for s_{t+N}
            # Actor needs the full next_state_batch trajectory to compute action for its last state
            next_action_N, next_log_prob_N, _ = self.actor.sample(next_state_batch) # (b, 1), (b, 1)

            # Target Q for s_{t+N}, a'_{t+N}
            # Target critics need the *initial state* s_{t+N} and the single action a'_{t+N}
            # Reshape next_action_N for critic: (b, 1, action_dim)
            target_q1_N = self.critic1_target(s_t_plus_N, next_action_N.unsqueeze(1))[:, 0] # (b, 1)
            target_q2_N = self.critic2_target(s_t_plus_N, next_action_N.unsqueeze(1))[:, 0] # (b, 1)
            target_q_min_N = torch.min(target_q1_N, target_q2_N) # (b, 1)

            # Target Value V(s_{t+N})
            current_alpha = self.log_alpha.exp()
            target_V_N = target_q_min_N - current_alpha * next_log_prob_N # (b, 1)

            # Calculate cumulative discounted reward sum(gamma^k * r_{t+k}) for k=0..N-1
            discounted_rewards = torch.zeros_like(reward_batch).to(self.device) # (b, 1)
            gamma_pow = 1.0
            for k in range(self.sequence_length):
                discounted_rewards += gamma_pow * rewards_sequence[:, k].unsqueeze(1) # (b, 1)
                gamma_pow *= self.gamma

            # Final N-step target Y
            y = discounted_rewards + (gamma_pow * (1.0 - done_batch) * target_V_N) # (b, 1)

        # --- Calculate Current Q predictions ---
        # Get Q-value for the *full sequence* from current critics
        # We need the Q-value corresponding to the *last* action in the sequence, Q(s_t, a_t...a_{t+N-1})
        current_q1_preds_seq = self.critic1(initial_state_critic, action_sequence) # (b, N, 1)
        current_q2_preds_seq = self.critic2(initial_state_critic, action_sequence) # (b, N, 1)

        # We compare the target 'y' with the predicted Q-value for the full N-step horizon,
        # which should be the last element of the critic output sequence.
        current_q1_N = current_q1_preds_seq[:, -1, :] # (b, 1)
        current_q2_N = current_q2_preds_seq[:, -1, :] # (b, 1)

        # --- Calculate Critic Loss ---
        # Loss compares the N-step target 'y' with the critic's prediction for the N-step Q-value
        critic1_loss = F.mse_loss(current_q1_N, y)
        critic2_loss = F.mse_loss(current_q2_N, y)

        # Optimize Critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Actor Update ---
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # Get policy action and log prob for the initial state s_t (using the state_batch trajectory)
        action_pi_t, log_prob_pi_t, _ = self.actor.sample(state_batch) # (b, 1), (b, 1)

        # Get Q-value for s_t and the policy action a_t
        # We need the Q-value for the *first* step (n=0) from the critics.
        # Critic requires s_t and the sequence starting with a_t.
        # Reconstruct action sequence starting with a_t: [a_t, a_{t+1}, ..., a_{t+N-1}]
        action_sequence_pi = torch.cat([action_pi_t.unsqueeze(1), action_sequence[:, 1:, :]], dim=1) # (b, N, action_dim)

        q1_pi_seq = self.critic1(initial_state_critic, action_sequence_pi) # (b, N, 1)
        q2_pi_seq = self.critic2(initial_state_critic, action_sequence_pi) # (b, N, 1)

        # Use the Q-value for the first step (n=0)
        q1_pi = q1_pi_seq[:, 0, :] # (b, 1)
        q2_pi = q2_pi_seq[:, 0, :] # (b, 1)
        q_pi_min = torch.min(q1_pi, q2_pi) # (b, 1)

        current_alpha = self.log_alpha.exp()
        actor_loss = (current_alpha * log_prob_pi_t - q_pi_min).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # --- Alpha Update ---
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_pi_t.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Target Network Update ---
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }

    def save_model(self, path: str):
        print(f"Saving T-SAC model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'device_type': self.device.type
        }
        if self.auto_tune_alpha:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            print(f"Warn: T-SAC model not found: {path}. Skipping load.")
            return
        print(f"Loading T-SAC model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

        if self.auto_tune_alpha and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha'].to(self.device)
            if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            try: self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except ValueError as e: print(f"Warn: Could not load T-SAC alpha opt state: {e}. Reinit."); self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            self.alpha = self.log_alpha.exp().item()
        elif not self.auto_tune_alpha:
             self.alpha = self.log_alpha.exp().item()

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        print(f"T-SAC model loaded successfully from {path}")


# --- Training Loop ---

def train_tsac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    tsac_config = config.tsac
    train_config = config.training
    buffer_config = config.replay_buffer
    world_config = config.world
    cuda_device = config.cuda_device

    learning_starts = train_config.learning_starts
    gradient_steps = train_config.gradient_steps
    train_freq = train_config.train_freq
    batch_size = train_config.batch_size
    log_frequency_ep = train_config.log_frequency
    save_interval_ep = train_config.save_interval

    # Device Setup
    if torch.cuda.is_available():
        if use_multi_gpu: print(f"Warn: Multi-GPU not standard for T-SAC. Using: {cuda_device}")
        device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try: torch.cuda.set_device(device); print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e: print(f"Warn: Could not set CUDA device {cuda_device}. E: {e}"); device = torch.device("cuda:0")
    else:
        device = torch.device("cpu"); print("GPU not available, using CPU.")

    # --- Initialization ---
    log_dir = os.path.join("runs", f"tsac_traj_{int(time.time())}") # Suffix changed
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    agent = TSAC(config=tsac_config, world_config=world_config, device=device)
    memory = ReplayBuffer(config=buffer_config, world_config=world_config)
    os.makedirs(train_config.models_dir, exist_ok=True)

    # --- Load Checkpoint Logic (similar to SAC) ---
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("tsac_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming T-SAC training from: {latest_model_path}")
        agent.load_model(latest_model_path)
        try:
            step_str = latest_model_path.split('_step')[-1].split('.pt')[0]
            ep_str = latest_model_path.split('_ep')[-1].split('_')[0]
            total_steps = int(step_str)
            start_episode = int(ep_str) + 1
            print(f"Resuming at step {total_steps}, episode {start_episode}")
        except (IndexError, ValueError): print("Warn: Could not parse steps/episode from filename.")
    else:
        print("\nStarting T-SAC training from scratch.")

    # --- Training Loop ---
    episode_rewards = []
    all_losses = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': []}
    timing_metrics = { 'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100) }

    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training T-SAC", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    world = World(world_config=world_config) # Initialize once

    for episode in pbar:
        world.reset()
        state = world.encode_state() # Initial trajectory dict
        episode_reward = 0
        episode_steps = 0

        episode_losses_temp = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': []}
        updates_made_this_episode = 0

        for step_in_episode in range(train_config.max_steps):
            action_normalized = agent.select_action(state, evaluate=False)

            step_start_time = time.time()
            world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            step_time = time.time() - step_start_time
            timing_metrics['env_step_time'].append(step_time)

            reward = world.reward
            next_state = world.encode_state()
            done = world.done

            memory.push(state, action_normalized, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Perform Updates
            if total_steps >= learning_starts and total_steps % train_freq == 0:
                for _ in range(gradient_steps):
                    if len(memory) >= batch_size:
                        update_start_time = time.time()
                        losses = agent.update_parameters(memory, batch_size)
                        update_time = time.time() - update_start_time
                        if losses:
                            timing_metrics['parameter_update_time'].append(update_time)
                            episode_losses_temp['critic1_loss'].append(losses['critic1_loss'])
                            episode_losses_temp['critic2_loss'].append(losses['critic2_loss'])
                            episode_losses_temp['actor_loss'].append(losses['actor_loss'])
                            episode_losses_temp['alpha'].append(losses['alpha'])
                            updates_made_this_episode += 1
                        else: break
                    else: break

            if done: break

        # --- Logging ---
        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else 0 for k, v in episode_losses_temp.items()}
        if updates_made_this_episode > 0:
             all_losses['critic1_loss'].append(avg_losses['critic1_loss'])
             all_losses['critic2_loss'].append(avg_losses['critic2_loss'])
             all_losses['actor_loss'].append(avg_losses['actor_loss'])
             all_losses['alpha'].append(avg_losses['alpha'])

        if episode % log_frequency_ep == 0:
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Environment_Step_ms_Avg100', np.mean(timing_metrics['env_step_time']) * 1000, total_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Parameter_Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time']) * 1000, total_steps)
            elif total_steps >= learning_starts: writer.add_scalar('Time/Parameter_Update_ms_Avg100', 0, total_steps)

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Buffer/Size', len(memory), total_steps)
            writer.add_scalar('Error/Distance_EndEpisode', world.error_dist, total_steps)

            if updates_made_this_episode > 0:
                writer.add_scalar('Loss/Critic1_AvgEp', avg_losses['critic1_loss'], total_steps)
                writer.add_scalar('Loss/Critic2_AvgEp', avg_losses['critic2_loss'], total_steps)
                writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                writer.add_scalar('Alpha/Value', avg_losses['alpha'], total_steps)
            else: # Log current agent alpha
                writer.add_scalar('Loss/Critic1_AvgEp', 0, total_steps)
                writer.add_scalar('Loss/Critic2_AvgEp', 0, total_steps)
                writer.add_scalar('Loss/Actor_AvgEp', 0, total_steps)
                writer.add_scalar('Alpha/Value', agent.alpha, total_steps)


            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar_postfix = {'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps, 'alpha': f"{agent.alpha:.3f}"}
            if updates_made_this_episode: pbar_postfix['c1_loss'] = f"{avg_losses['critic1_loss']:.2f}"
            pbar.set_postfix(pbar_postfix)

        if episode % save_interval_ep == 0:
            save_path = os.path.join(train_config.models_dir, f"tsac_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

    pbar.close()
    writer.close()
    print(f"T-SAC Training finished. Total steps: {total_steps}")

    final_save_path = os.path.join(train_config.models_dir, f"tsac_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_tsac(agent=agent, config=config) # Pass full config

    return agent, episode_rewards


# --- Evaluation Loop ---

def evaluate_tsac(agent: TSAC, config: DefaultConfig): # Pass full config
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

    agent.actor.eval() # Set actor to eval mode

    print(f"\nRunning T-SAC Evaluation for {eval_config.num_episodes} episodes...")
    world = World(world_config=world_config) # Create world instance

    for episode in range(eval_config.num_episodes):
        world.reset()
        state = world.encode_state() # Initial state dict
        episode_reward = 0
        episode_frames = []

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            try:
                fname = f"tsac_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config, filename=fname, collect_for_gif=True)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warn: Vis failed init state. E: {e}")

        for step in range(eval_config.max_steps):
            action_normalized = agent.select_action(state, evaluate=True)

            world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward
            next_state = world.encode_state()
            done = world.done

            if eval_config.render and vis_available:
                try:
                    fname = f"tsac_eval_ep{episode+1}_frame_{step+1:03d}.png"
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

        if not done:
            success = world.error_dist <= world_config.success_threshold
            status = "Success!" if success else "Failure."
            if success: success_count +=1
            print(f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps}). Final Err: {world.error_dist:.2f}. {status}")

        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        # GIF Saving
        if eval_config.render and vis_available and episode_frames:
            gif_filename = f"tsac_eval_episode_{episode+1}.gif"
            try:
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"Warn: Failed GIF save ep {episode+1}. E: {e}")
            if vis_config.delete_frames_after_gif: # Clean up frames
                 cleaned_count = 0
                 for frame in episode_frames:
                     if os.path.exists(frame):
                         try: os.remove(frame); cleaned_count += 1
                         except OSError as ose: print(f"    Warn: Could not delete TSAC frame file {frame}: {ose}")


    agent.actor.train() # Set actor back to train mode

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0

    print("\n--- T-SAC Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Reward: {avg_eval_reward:.2f} +/- {std_eval_reward:.2f}")
    print(f"Success Rate (Error <= {world_config.success_threshold}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_episode_gif_paths:
        print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering enabled but libs not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End T-SAC Evaluation ---\n")

    return eval_rewards, success_rate