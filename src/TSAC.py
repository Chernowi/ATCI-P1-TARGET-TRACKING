# --- START OF FILE tsac.py ---

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
from world_objects import Velocity
from visualization import visualize_world, reset_trajectories, save_gif
from configs import DefaultConfig, ReplayBufferConfig, TSACConfig, TrainingConfig
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, List

# --- Replay Buffer (Adapted for Sequences) ---

class ReplayBuffer:
    """Experience replay buffer storing transitions and sampling sequences."""

    def __init__(self, config: ReplayBufferConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.gamma = config.gamma # Store gamma for potential N-step calculations if needed directly here

    def push(self, state, action, reward, next_state, done):
        # Store individual transitions
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, sequence_length: int = 1) -> Optional[Tuple]:
        """Sample a batch of sequences from the buffer."""
        if len(self.buffer) < sequence_length:
            return None # Not enough data even for one sequence

        states, actions, rewards, next_states, dones = [], [], [], [], []
        valid_start_indices = []

        # Identify valid starting points for sequences
        # A start index 'i' is valid if the sequence buffer[i:i+sequence_length] exists
        # and no 'done' flag is True within the first sequence_length-1 transitions.
        for i in range(len(self.buffer) - sequence_length + 1):
            is_valid = True
            # Check for premature 'done' flags within the sequence span (excluding the very last step)
            for k in range(sequence_length - 1):
                if self.buffer[i + k][4]: # Check done flag at index 4
                    is_valid = False
                    break
            if is_valid:
                valid_start_indices.append(i)

        if not valid_start_indices or len(valid_start_indices) < batch_size:
            # Not enough valid sequences to form a batch
            # print(f"Warning: Not enough valid sequences ({len(valid_start_indices)}) in buffer for batch size {batch_size}. Need {sequence_length} contiguous steps without early 'done'. Buffer size: {len(self.buffer)}")
            return None # Indicate failure to sample

        # Sample starting indices for the batch
        sampled_indices = random.sample(valid_start_indices, batch_size)

        # Extract sequences
        for idx in sampled_indices:
            seq_s, seq_a, seq_r, seq_ns, seq_d = [], [], [], [], []
            for i in range(sequence_length):
                s, a, r, ns, d = self.buffer[idx + i]
                seq_s.append(s['basic_state']) # Store only the basic state tuple
                seq_a.append(a)
                seq_r.append(r)
                seq_ns.append(ns['basic_state']) # Store only the basic state tuple
                seq_d.append(d)
            states.append(seq_s)
            actions.append(seq_a)
            rewards.append(seq_r)
            next_states.append(seq_ns)
            dones.append(seq_d)

        # Return sequences as numpy arrays
        # Shapes: (batch_size, sequence_length, feature_dim)
        return (np.array(states), np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

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
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (1, max_len, d_model)
        # Output shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


# --- T-SAC Actor (Modified SAC Actor) ---

class Actor(nn.Module):
    """Policy network (Actor) for T-SAC."""

    def __init__(self, config: TSACConfig):
        super(Actor, self).__init__()
        self.use_layer_norm = config.use_layer_norm_actor

        self.layers = nn.ModuleList()
        mlp_input_dim = config.state_dim
        for hidden_dim in config.hidden_dims:
            linear_layer = nn.Linear(mlp_input_dim, hidden_dim)
            # Initialize weights using Kaiming/He initialization
            nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
            # Initialize bias
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(linear_layer.bias, -bound, bound)
            
            self.layers.append(linear_layer)
            if self.use_layer_norm:
                self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            mlp_input_dim = hidden_dim

        # Initialize mean output layer with smaller values
        self.mean = nn.Linear(config.hidden_dims[-1], config.action_dim)
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)
        
        # Initialize log_std output layer
        self.log_std = nn.Linear(config.hidden_dims[-1], config.action_dim)
        nn.init.xavier_uniform_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, 0)

        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        x = state
        for layer in self.layers:
            x = layer(x)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from the policy distribution."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        action_normalized = torch.tanh(x_t) # Squash action to [-1, 1]

        # Calculate log probability with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1 - action_normalized.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # Return squashed action, log_prob, and squashed mean action
        return action_normalized, log_prob, torch.tanh(mean)


# --- T-SAC Critic (Transformer-based) ---

class TransformerCritic(nn.Module):
    """Q-function network (Critic) for T-SAC using a Transformer."""

    def __init__(self, config: TSACConfig):
        super(TransformerCritic, self).__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.embedding_dim = config.embedding_dim
        self.sequence_length = config.sequence_length

        # Embeddings
        self.state_embed = nn.Linear(config.state_dim, config.embedding_dim)
        self.action_embed = nn.Linear(config.action_dim, config.embedding_dim)
        self.pos_encoder = PositionalEncoding(config.embedding_dim, max_len=config.sequence_length + 1) # +1 if state prepended

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_n_heads,
            dim_feedforward=config.transformer_hidden_dim,
            batch_first=True, # Expect (batch, seq, feature)
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_n_layers)

        # Output head (predicts Q-value for each step in the sequence)
        self.q_head = nn.Linear(config.embedding_dim, 1)

        # Generate causal mask dynamically based on sequence length
        # Mask ensures prediction at step 'i' cannot attend to steps > 'i'
        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(config.sequence_length, device=None) # Device set later

    def _generate_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates the causal mask dynamically on the correct device."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask.to(device)


    def forward(self, state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer critic.

        Args:
            state (torch.Tensor): The initial state s_t. Shape: (batch, state_dim)
            action_sequence (torch.Tensor): Sequence of actions a_t to a_{t+N-1}.
                                            Shape: (batch, sequence_length, action_dim)

        Returns:
            torch.Tensor: Sequence of Q-value predictions for each step.
                          Shape: (batch, sequence_length, 1)
        """
        batch_size, seq_len, _ = action_sequence.shape
        device = state.device

        # 1. Embed state and actions
        state_emb = self.state_embed(state).unsqueeze(1) # Shape: (batch, 1, embedding_dim)
        action_emb = self.action_embed(action_sequence) # Shape: (batch, seq_len, embedding_dim)

        # 2. Combine state and action embeddings for Transformer input
        # Prepend state embedding to the action sequence embeddings
        transformer_input_emb = torch.cat([state_emb, action_emb], dim=1) # Shape: (batch, seq_len + 1, embedding_dim)

        # 3. Add positional encoding
        transformer_input_pos = self.pos_encoder(transformer_input_emb) # Shape: (batch, seq_len + 1, embedding_dim)

        # 4. Generate causal mask for the combined sequence length
        # Mask needs size seq_len + 1 because state is prepended
        mask = self._generate_mask(seq_len + 1, device)

        # 5. Pass through Transformer encoder
        transformer_output = self.transformer_encoder(transformer_input_pos, mask=mask) # Shape: (batch, seq_len + 1, embedding_dim)

        # 6. Extract outputs corresponding to action steps (ignore output for state step)
        # Output corresponding to action a_{t+i} is at index i+1 in the output sequence
        action_transformer_output = transformer_output[:, 1:, :] # Shape: (batch, seq_len, embedding_dim)

        # 7. Pass through Q-head
        q_predictions = self.q_head(action_transformer_output) # Shape: (batch, seq_len, 1)

        return q_predictions


# --- T-SAC Agent ---

class TSAC:
    """Transformer-based Soft Actor-Critic algorithm implementation."""

    def __init__(self, config: TSACConfig, device: torch.device = None):
        self.config = config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha # Initial alpha value
        self.action_scale = config.action_scale
        self.auto_tune_alpha = config.auto_tune_alpha
        self.sequence_length = config.sequence_length # N for N-step returns

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"T-SAC Agent using device: {self.device}")
        print(f"T-SAC using Sequence Length (N-step): {self.sequence_length}")

        # Initialize Actor
        self.actor = Actor(config).to(self.device)

        # Initialize Twin Transformer Critics and Target Critics
        self.critic1 = TransformerCritic(config).to(self.device)
        self.critic2 = TransformerCritic(config).to(self.device)
        self.critic1_target = TransformerCritic(config).to(self.device)
        self.critic2_target = TransformerCritic(config).to(self.device)

        # Initialize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.lr)

        # Alpha tuning
        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([config.action_dim]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device) # Initialize log_alpha
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device) # Fixed alpha

    def select_action(self, state: dict, evaluate: bool = False) -> np.ndarray:
        """Select action based on state (step-based evaluation)."""
        state_tuple = state['basic_state']
        state_tensor = torch.FloatTensor(state_tuple).to(self.device).unsqueeze(0) # Shape (1, state_dim)

        with torch.no_grad():
            if evaluate:
                # During evaluation, use the mean action
                _, _, action_mean_squashed = self.actor.sample(state_tensor)
                action_normalized = action_mean_squashed
            else:
                # During training, sample from the distribution
                action_normalized, _, _ = self.actor.sample(state_tensor)

        action_scaled = action_normalized.detach().cpu().numpy()[0] * self.action_scale
        return action_scaled

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform a single T-SAC update step using a batch of sequences."""

        # Sample sequences
        sampled_batch = memory.sample(batch_size, self.sequence_length)
        if sampled_batch is None:
            return None # Not enough data in buffer

        state_batch, action_batch_scaled, reward_batch, next_state_batch, done_batch = sampled_batch

        # Move batch to device
        state_batch = torch.FloatTensor(state_batch[:, 0, :]).to(self.device) # Initial state s_t (batch, state_dim)
        action_batch_normalized = torch.FloatTensor(action_batch_scaled / self.action_scale).to(self.device) # (batch, seq_len, action_dim)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device) # (batch, seq_len)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device) # (batch, seq_len, state_dim)
        done_batch = torch.FloatTensor(done_batch).to(self.device) # (batch, seq_len)

        # --- Critic Update ---
        with torch.no_grad():
            # Calculate N-step targets G(n) for n = 1 to sequence_length
            n_step_targets = []
            cumulative_rewards = torch.zeros_like(reward_batch[:, 0]).to(self.device) # (batch)
            discount_factor = torch.ones_like(reward_batch[:, 0]).to(self.device) # (batch)

            for n in range(self.sequence_length):
                # Accumulate discounted reward for step n
                cumulative_rewards += discount_factor * reward_batch[:, n]
                discount_factor *= self.gamma

                # Get next state s_{t+n+1} (which is next_state_batch[:, n, :])
                s_next_n = next_state_batch[:, n, :] # Shape: (batch, state_dim)
                done_n = done_batch[:, n] # Shape: (batch)

                # Get target Q-value for s_{t+n+1}
                next_action_n, next_log_prob_n, _ = self.actor.sample(s_next_n)

                # Use target critics - they need s_next_n and the *single* next_action_n
                # Target Transformer Critic expects sequence input, but here we need Q(s', a')
                # We need a way to get Q(s', a') from the Transformer Critic.
                # Simplification: Use 1-step action sequence for target Q calculation.
                # This is a deviation from pure N-step but feasible with Transformer architecture.
                # Feed s_next_n and next_action_n (as a sequence of length 1) to target critics.
                target_q1_n = self.critic1_target(s_next_n, next_action_n.unsqueeze(1))[:, 0] # Get Q value from first (only) step
                target_q2_n = self.critic2_target(s_next_n, next_action_n.unsqueeze(1))[:, 0]
                target_q_min_n = torch.min(target_q1_n, target_q2_n).squeeze(-1) # Shape: (batch)

                # Include entropy term
                target_q_entropy_n = target_q_min_n - self.log_alpha.exp() * next_log_prob_n.squeeze(-1) # Shape: (batch)

                # Calculate N-step target G(n+1) (indexed by n from 0 to seq_len-1)
                # G(n+1) = sum_{j=0}^{n} gamma^j r_{t+j} + gamma^{n+1} * (1-done_n) * target_Q_entropy(s_{t+n+1}, a'_{t+n+1})
                y_n = cumulative_rewards + (discount_factor * (1.0 - done_n) * target_q_entropy_n)
                n_step_targets.append(y_n)

            # Stack targets: Shape (batch, sequence_length)
            target_q_values = torch.stack(n_step_targets, dim=1)

        # --- Calculate Current Q predictions ---
        # Q-predictions from current critics for the action sequence
        current_q1_preds = self.critic1(state_batch, action_batch_normalized) # Shape: (batch, seq_len, 1)
        current_q2_preds = self.critic2(state_batch, action_batch_normalized) # Shape: (batch, seq_len, 1)

        # --- Calculate Critic Loss (Gradient Averaging via Summation) ---
        # Calculate MSE loss for each step prediction against its corresponding N-step target
        critic1_loss_total = 0.0
        critic2_loss_total = 0.0
        for n in range(self.sequence_length):
            # Loss for Q(s_t, a_t ... a_{t+n}) vs G(n+1) target
            critic1_loss_total += F.mse_loss(current_q1_preds[:, n, 0], target_q_values[:, n])
            critic2_loss_total += F.mse_loss(current_q2_preds[:, n, 0], target_q_values[:, n])

        # Average loss over sequence length implicitly by summing
        critic1_loss = critic1_loss_total / self.sequence_length
        critic2_loss = critic2_loss_total / self.sequence_length

        # Optimize Critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Optimize Critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Actor Update ---
        # Freeze critic parameters for actor update
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # Sample action from policy for the initial state s_t
        action_pi, log_prob_pi, _ = self.actor.sample(state_batch)

        # Get Q-value for the policy's action using the *first* step prediction of the critics
        q1_pi = self.critic1(state_batch, action_pi.unsqueeze(1))[:, 0] # (batch, 1)
        q2_pi = self.critic2(state_batch, action_pi.unsqueeze(1))[:, 0] # (batch, 1)
        q_pi_min = torch.min(q1_pi, q2_pi) # (batch, 1)

        # Calculate Actor loss
        actor_loss = (self.log_alpha.exp() * log_prob_pi - q_pi_min).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # --- Alpha Update ---
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item() # Update alpha value

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
            save_dict['log_alpha'] = self.log_alpha # Save the tensor directly
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            print(f"Warning: T-SAC model file not found at {path}. Skipping loading.")
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
            if not self.log_alpha.requires_grad:
                 self.log_alpha.requires_grad_(True)
            # Re-initialize alpha optimizer with the loaded log_alpha
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            try:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except ValueError as e:
                 print(f"Warning: Could not load T-SAC alpha optimizer state: {e}. Reinitializing.")
                 self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            self.alpha = self.log_alpha.exp().item()
        elif not self.auto_tune_alpha:
            self.alpha = self.log_alpha.exp().item() # Set alpha based on fixed log_alpha

        # Ensure target networks are synced and don't require grads
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        print(f"T-SAC model loaded successfully from {path}")

# --- Training Loop ---

def train_tsac(config: DefaultConfig, use_multi_gpu: bool = False):
    tsac_config = config.tsac # Use T-SAC specific config
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

    total_steps = 0

    log_dir = os.path.join("runs", f"tsac_training_{int(time.time())}") # Log dir specific to T-SAC
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Device Setup
    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Warning: Multi-GPU not explicitly handled for T-SAC parameter updates. Using single specified/default GPU.")
            device = torch.device(cuda_device) # Default to specified device even if multiple available
        else:
            device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try:
                torch.cuda.set_device(device) # Set the current device
                print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e:
                print(f"Warning: Could not set CUDA device {cuda_device}. Using default. Error: {e}")
                device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU for training")

    # Initialize Agent and Replay Buffer
    agent = TSAC(config=tsac_config, device=device)
    memory = ReplayBuffer(config=buffer_config)

    os.makedirs(train_config.models_dir, exist_ok=True)

    episode_rewards = []
    all_losses = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': []}

    timing_metrics = {
        'env_step_time': [],
        'parameter_update_time': []
    }

    pbar = tqdm(range(1, train_config.num_episodes + 1), desc="Training T-SAC", unit="episode")

    for episode in pbar:
        env = World(world_config=world_config)
        state = env.encode_state() # State includes basic_state dict field
        episode_reward = 0
        episode_steps = 0

        episode_step_times = []
        episode_param_update_times = []
        episode_losses = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': []}

        for step_in_episode in range(train_config.max_steps):
            action_scaled = agent.select_action(state, evaluate=False)
            action_obj = Velocity(x=action_scaled[0], y=action_scaled[1], z=0.0)

            step_start_time = time.time()
            env.step(action_obj, training=True, terminal_step=step_in_episode == train_config.max_steps - 1)
            step_time = time.time() - step_start_time
            episode_step_times.append(step_time)

            reward = env.reward
            next_state = env.encode_state()
            done = env.done

            # Push state dictionary, action, reward, next_state dictionary, done
            memory.push(state, action_scaled, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Perform Updates
            if total_steps >= learning_starts and total_steps % train_freq == 0:
                updates_this_step = 0
                for _ in range(gradient_steps):
                    # Check buffer size condition *before* sampling
                    if len(memory) >= batch_size * agent.sequence_length:
                        update_start_time = time.time()
                        losses = agent.update_parameters(memory, batch_size)
                        update_time = time.time() - update_start_time

                        if losses: # Check if update was successful
                            episode_param_update_times.append(update_time)
                            episode_losses['critic1_loss'].append(losses['critic1_loss'])
                            episode_losses['critic2_loss'].append(losses['critic2_loss'])
                            episode_losses['actor_loss'].append(losses['actor_loss'])
                            episode_losses['alpha'].append(losses['alpha'])
                            updates_this_step += 1
                        else:
                            # Break gradient step loop if sampling failed
                            break
                # print(f"Step {total_steps}: Performed {updates_this_step} gradient updates.")


            if done:
                break # End episode

        # --- Logging and Reporting (End of Episode) ---
        episode_rewards.append(episode_reward)

        # Average losses for this episode
        avg_losses = {k: np.mean(v) if v else 0 for k, v in episode_losses.items()}
        updates_made_this_episode = any(v for v in episode_losses.values() if len(v) > 0)
        if updates_made_this_episode:
             all_losses['critic1_loss'].append(avg_losses['critic1_loss'])
             all_losses['critic2_loss'].append(avg_losses['critic2_loss'])
             all_losses['actor_loss'].append(avg_losses['actor_loss'])
             all_losses['alpha'].append(avg_losses['alpha'])


        if episode % log_frequency_ep == 0:
            # Log timing
            if episode_step_times:
                avg_step_time = np.mean(episode_step_times)
                timing_metrics['env_step_time'].append(avg_step_time)
                writer.add_scalar('Time/Environment_Step_ms', avg_step_time * 1000, total_steps)
            if episode_param_update_times:
                avg_param_update_time = np.mean(episode_param_update_times)
                timing_metrics['parameter_update_time'].append(avg_param_update_time)
                writer.add_scalar('Time/Parameter_Update_ms', avg_param_update_time * 1000, total_steps)
            elif total_steps >= learning_starts:
                 writer.add_scalar('Time/Parameter_Update_ms', 0, total_steps) # Log 0 if learning started but no updates possible

            # Log core metrics
            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Buffer/Size', len(memory), total_steps)
            writer.add_scalar('Error/Distance_EndEpisode', env.error_dist, total_steps)

            # Log losses if updates were made
            if updates_made_this_episode:
                writer.add_scalar('Loss/Critic1_AvgEp', avg_losses['critic1_loss'], total_steps)
                writer.add_scalar('Loss/Critic2_AvgEp', avg_losses['critic2_loss'], total_steps)
                writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                writer.add_scalar('Alpha/Value', avg_losses['alpha'], total_steps)

            # Log average reward
            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        # Update progress bar
        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar_postfix = {'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps, 'alpha': f"{agent.alpha:.3f}"}
            if updates_made_this_episode:
                pbar_postfix['c1_loss'] = f"{avg_losses['critic1_loss']:.2f}"
                pbar_postfix['act_loss'] = f"{avg_losses['actor_loss']:.2f}"
            pbar.set_postfix(pbar_postfix)

        # Save model periodically
        if episode % save_interval_ep == 0:
            save_path = os.path.join(train_config.models_dir, f"tsac_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

    pbar.close()
    writer.close()
    print(f"T-SAC Training finished. Total steps: {total_steps}")

    # Save final model
    final_save_path = os.path.join(train_config.models_dir, f"tsac_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    return agent, episode_rewards


# --- Evaluation Loop ---

def evaluate_tsac(agent: TSAC, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    eval_rewards = []
    success_count = 0
    all_episode_gif_paths = []

    agent.actor.eval() # Set actor to evaluation mode

    print(f"\nRunning T-SAC Evaluation for {eval_config.num_episodes} episodes...")
    for episode in range(eval_config.num_episodes):
        env = World(world_config=world_config)
        state = env.encode_state()
        episode_reward = 0
        episode_frames = []

        # Render initial state
        if eval_config.render:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            try:
                initial_frame_file = visualize_world(
                    world=env, vis_config=vis_config,
                    filename=f"tsac_eval_ep{episode+1}_frame_000_initial.png",
                    collect_for_gif=True)
                if initial_frame_file: episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warning: Vis failed init state. E: {e}")

        # Run episode steps
        for step in range(eval_config.max_steps):
            action_scaled = agent.select_action(state, evaluate=True)
            action_obj = Velocity(x=action_scaled[0], y=action_scaled[1], z=0.0)
            env.step(action_obj, training=False) # Use training=False for eval
            reward = env.reward
            next_state = env.encode_state()
            done = env.done

            # Render step
            if eval_config.render:
                try:
                    frame_file = visualize_world(
                        world=env, vis_config=vis_config,
                        filename=f"tsac_eval_ep{episode+1}_frame_{step+1:03d}.png",
                        collect_for_gif=True)
                    if frame_file: episode_frames.append(frame_file)
                except Exception as e: print(f"Warning: Vis failed step {step+1}. E: {e}")

            state = next_state
            episode_reward += reward

            if done:
                if env.error_dist <= world_config.success_threshold:
                    success_count += 1
                    print(f"  Episode {episode+1}: Success! Step {step+1} (Err: {env.error_dist:.2f} <= Thresh {world_config.success_threshold})")
                else:
                    print(f"  Episode {episode+1}: Terminated early Step {step+1}. Final Err: {env.error_dist:.2f}")
                break

        # End of episode info
        if not done:
            print(f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps}). Final Err: {env.error_dist:.2f}")
        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        # Save GIF
        if eval_config.render and episode_frames:
            gif_filename = f"tsac_eval_episode_{episode+1}.gif"
            try:
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config,
                                    frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"Warning: Failed GIF save ep {episode+1}. E: {e}")
            if vis_config.delete_frames_after_gif and not gif_path: # Clean up frames if GIF failed
                 for frame in episode_frames:
                     if os.path.exists(frame): os.remove(frame)


    agent.actor.train() # Set actor back to training mode

    # Final evaluation summary
    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0
    print("\n--- T-SAC Evaluation Summary ---")
    print(f"Average Evaluation Reward: {avg_eval_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and all_episode_gif_paths:
        print(f"GIFs saved in: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render:
         print("Rendering enabled, but no GIFs successfully created.")
    print("--- End T-SAC Evaluation ---")

    return eval_rewards