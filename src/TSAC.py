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

# Local imports
from world import World
from configs import DefaultConfig, ReplayBufferConfig, TSACConfig, WorldConfig, CORE_STATE_DIM, CORE_ACTION_DIM, TRAJECTORY_REWARD_DIM
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, List
from utils import RunningMeanStd # Import the normalization module

# --- Replay Buffer (Same as in SAC.py) ---
class ReplayBuffer:
    """Experience replay buffer storing full state trajectories."""
    def __init__(self, config: ReplayBufferConfig, world_config: WorldConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim

    def push(self, state, action, reward, next_state, done):
        trajectory = state['full_trajectory']
        next_trajectory = next_state['full_trajectory']
        if isinstance(action, (np.ndarray, list)): action = action[0]
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (self.trajectory_length, self.feature_dim):
             return
        if not isinstance(next_trajectory, np.ndarray) or next_trajectory.shape != (self.trajectory_length, self.feature_dim):
             return
        if np.isnan(trajectory).any() or np.isnan(next_trajectory).any():
             return

        self.buffer.append((trajectory, float(action), float(reward), next_trajectory, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        if len(self.buffer) < batch_size:
            return None
        try:
            batch = random.sample(self.buffer, batch_size)
        except ValueError:
            print(f"Warning: TSAC ReplayBuffer sampling failed. len(buffer)={len(self.buffer)}, batch_size={batch_size}")
            return None

        trajectory, action, reward, next_trajectory, done = zip(*batch)

        try:
            trajectory_arr = np.array(trajectory, dtype=np.float32)
            action_arr = np.array(action, dtype=np.float32).reshape(-1, 1)
            reward_arr = np.array(reward, dtype=np.float32)
            next_trajectory_arr = np.array(next_trajectory, dtype=np.float32)
            done_arr = np.array(done, dtype=np.float32)

            if np.isnan(trajectory_arr).any() or np.isnan(next_trajectory_arr).any():
                print("Warning: TSAC Sampled batch contains NaN values. Returning None.")
                return None
        except Exception as e:
            print(f"Error converting TSAC sampled batch to numpy arrays: {e}")
            return None

        return (trajectory_arr, action_arr, reward_arr, next_trajectory_arr, done_arr)

    def __len__(self):
        return len(self.buffer)

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: Tensor, shape [batch_size, seq_len, embedding_dim] """
        pe_batch_first = self.pe.squeeze(1).transpose(0, 1) # (1, max_len, embed_dim)
        return x + pe_batch_first[:, :x.size(1), :]


# --- T-SAC Actor (Modified based on Paper Ablations) ---
class Actor(nn.Module):
    """Policy network (Actor) for T-SAC. Uses the normalized last basic_state."""
    def __init__(self, config: TSACConfig, world_config: WorldConfig):
        super(Actor, self).__init__()
        self.config = config
        self.world_config = world_config
        self.use_layer_norm = config.use_layer_norm_actor
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim

        self.layers = nn.ModuleList()
        mlp_input_dim = self.state_dim
        current_dim = mlp_input_dim

        # Match Box Pushing hyperparams: [512 * 4] = [2048] ? or [512, 512, 512, 512]? Using [512, 512] for now.
        # Let's default to config's hidden_dims unless explicitly overridden for BoxPush
        hidden_dims = config.hidden_dims if config.hidden_dims else [512, 512] # Fallback

        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(current_dim, hidden_dim)
            # Paper uses Fan-in initialization (similar to Kaiming Uniform default)
            nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
            if linear_layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(linear_layer.bias, -bound, bound)
            self.layers.append(linear_layer)
            if self.use_layer_norm:
                self.layers.append(nn.LayerNorm(hidden_dim))
            # Use Leaky ReLU based on Table 4
            self.layers.append(nn.LeakyReLU())
            current_dim = hidden_dim

        self.mean = nn.Linear(current_dim, self.action_dim)
        # Default Pytorch init is Kaiming Uniform for Linear
        # Paper uses Fan-in for actor head as well? Let's try default first.
        # nn.init.xavier_uniform_(self.mean.weight, gain=0.01) # Old init
        # nn.init.constant_(self.mean.bias, 0)

        self.log_std = nn.Linear(current_dim, self.action_dim)
        # nn.init.xavier_uniform_(self.log_std.weight, gain=0.01) # Old init
        # nn.init.constant_(self.log_std.bias, 0)

        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, normalized_last_basic_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using the normalized last basic state."""
        x = normalized_last_basic_state
        for layer in self.layers:
            x = layer(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, normalized_last_basic_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (normalized) from the policy distribution."""
        mean, log_std = self.forward(normalized_last_basic_state)
        std = log_std.exp()

        # Add stability checks
        if torch.isnan(std).any() or torch.isinf(std).any() or (std < 1e-6).any():
            # print(f"Warning: Actor std unstable: {std}. Clamping log_std.")
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max - 0.5) # Clamp harder
            std = log_std.exp()
            if torch.isnan(std).any() or torch.isinf(std).any() or (std < 1e-6).any():
                 print(f"ERROR: Actor std still invalid after clamp: {std}. Returning zero action.")
                 # Return a default, low-probability action
                 zero_action = torch.zeros_like(mean)
                 # Very low log prob to indicate error / avoid large gradients
                 low_log_prob = torch.full_like(mean.sum(dim=1, keepdim=True), -1e6)
                 return zero_action, low_log_prob, torch.tanh(mean) # Return tanh(mean) as mean action


        normal = Normal(mean, std)
        try:
             # Use rsample for reparameterization trick
             x_t = normal.rsample()
        except ValueError as e:
             print(f"ERROR: Normal distribution rsample failed: {e}. Mean: {mean}, Std: {std}")
             zero_action = torch.zeros_like(mean)
             low_log_prob = torch.full_like(mean.sum(dim=1, keepdim=True), -1e6)
             return zero_action, low_log_prob, torch.tanh(mean)


        action_normalized = torch.tanh(x_t)

        try:
            # Log prob with tanh correction (safer calculation)
            log_prob_unbounded = normal.log_prob(x_t)
            # Clamp before log_det_jacobian calculation
            clamped_tanh = action_normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7) # Add epsilon
            log_prob = log_prob_unbounded - log_det_jacobian
            log_prob = log_prob.sum(1, keepdim=True)

            if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                # print(f"Warning: NaN/Inf log_prob detected. Unbounded: {log_prob_unbounded}, Jacobian: {log_det_jacobian}")
                log_prob = torch.nan_to_num(log_prob, nan=-1e6, posinf=-1e6, neginf=-1e6)


        except Exception as e:
            print(f"ERROR calculating log_prob: {e}")
            log_prob = torch.full_like(mean.sum(dim=1, keepdim=True), -1e6)


        return action_normalized, log_prob, torch.tanh(mean)


# --- T-SAC Critic (Revised based on Paper Interpretation) ---
class TransformerCritic(nn.Module):
    """Q-function network (Critic) for T-SAC using a Transformer.
       Predicts n-step returns for action sequences of length 1 to N.
    """
    def __init__(self, config: TSACConfig, world_config: WorldConfig):
        super(TransformerCritic, self).__init__()
        self.config = config
        self.world_config = world_config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        # Embedding dim from config, e.g., 256 (4 heads * 64 dims/head)
        self.embedding_dim = config.embedding_dim
        self.sequence_length = world_config.trajectory_length # N (max sequence length)
        # Transformer input includes initial state + N actions = N+1 tokens
        self.transformer_seq_len = self.sequence_length + 1

        # Embeddings
        self.state_embed = nn.Linear(self.state_dim, self.embedding_dim)
        self.action_embed = nn.Linear(self.action_dim, self.embedding_dim)
        # Increased max_len slightly just in case
        self.pos_encoder = PositionalEncoding(self.embedding_dim, max_len=self.transformer_seq_len + 10)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=config.transformer_n_heads,
            dim_feedforward=config.transformer_hidden_dim, batch_first=True,
            activation='gelu', dropout=0.1, norm_first=True) # Use norm_first for potentially better stability
        encoder_norm = nn.LayerNorm(self.embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_n_layers, norm=encoder_norm)

        # Output Head - predicts Q-value for each step
        self.q_head = nn.Linear(self.embedding_dim, 1)

        # Causal mask generation
        self.mask = None

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a square causal mask for attention."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, normalized_initial_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer critic.
           Expects normalized_initial_state: (batch, state_dim)
           action_sequence: (batch, n, action_dim) where n <= N
           Returns Q-predictions: (batch, n, 1) - Q-value for each action subsequence
        """
        batch_size, current_seq_len, _ = action_sequence.shape # current_seq_len = n
        device = normalized_initial_state.device
        transformer_input_len = current_seq_len + 1 # n+1 tokens (s0, a1..an)

        # Embed initial state and actions
        state_emb = self.state_embed(normalized_initial_state).unsqueeze(1) # (batch, 1, embed_dim)
        action_emb = self.action_embed(action_sequence) # (batch, n, embed_dim)

        # Concatenate: [s0_emb, a1_emb, ..., an_emb]
        transformer_input_emb = torch.cat([state_emb, action_emb], dim=1) # (batch, n+1, embed_dim)

        # Add positional encoding
        transformer_input_pos = self.pos_encoder(transformer_input_emb) # (batch, n+1, embed_dim)

        # Generate causal mask if needed for the current input length
        if self.mask is None or self.mask.size(0) != transformer_input_len:
             self.mask = self._generate_square_subsequent_mask(transformer_input_len, device)

        # Pass through transformer encoder with causal mask
        # Shape: (batch, n+1, embed_dim)
        transformer_output = self.transformer_encoder(transformer_input_pos, mask=self.mask)

        # Extract outputs corresponding to *actions*. Output at index i corresponds to sequence (s0, a1...ai)
        # We need outputs from index 1 to n (inclusive) -> corresponds to a1 to an
        action_transformer_output = transformer_output[:, 1:, :] # (batch, n, embed_dim)

        # Pass through Q-head
        q_predictions = self.q_head(action_transformer_output) # (batch, n, 1)

        if torch.isnan(q_predictions).any():
             print("WARNING: TransformerCritic forward pass produced NaN q_predictions!")
             q_predictions = torch.nan_to_num(q_predictions, nan=0.0) # Replace NaNs with 0

        return q_predictions


# --- T-SAC Agent (Revised Update Logic) ---
class TSAC:
    """Transformer-based Soft Actor-Critic aligned with paper's N-step return and gradient averaging."""
    def __init__(self, config: TSACConfig, world_config: WorldConfig, device: torch.device = None):
        self.config = config
        self.world_config = world_config
        # Use paper's default gamma unless overridden (check table)
        self.gamma = config.gamma
        self.tau = config.tau # Use paper's polyak_weight if specified, else default
        self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha
        self.sequence_length = world_config.trajectory_length # N
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"T-SAC Agent using device: {self.device} (Paper Aligned - Gradient Averaging)")
        print(f"T-SAC using Max Sequence Length (N): {self.sequence_length}")

        # State Normalization
        self.state_normalizer = RunningMeanStd(shape=(self.state_dim,)).to(self.device)
        print(f"T-SAC Agent state normalization enabled for dim: {self.state_dim}")

        # Initialize networks
        self.actor = Actor(config, world_config).to(self.device)
        self.critic1 = TransformerCritic(config, world_config).to(self.device)
        self.critic2 = TransformerCritic(config, world_config).to(self.device)
        self.critic1_target = TransformerCritic(config, world_config).to(self.device)
        self.critic2_target = TransformerCritic(config, world_config).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        # Optimizers (Use separate LR for critic if specified, like BoxPush)
        actor_lr = config.lr
        critic_lr = config.lr # Default to same LR
        # Check if a specific critic LR is available (like in BoxPush table)
        if hasattr(config, 'critic_lr') and config.critic_lr is not None:
             critic_lr = config.critic_lr
             print(f"Using separate critic LR: {critic_lr}")


        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4) # Use AdamW
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), lr=critic_lr, weight_decay=1e-4)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), lr=critic_lr, weight_decay=1e-4)

        # Alpha Initialization
        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = nn.Parameter(torch.tensor(np.log(self.alpha), device=self.device, dtype=torch.float32))
            # Use actor_lr for alpha optimizer too? Or critic_lr? Let's use actor_lr
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=actor_lr, weight_decay=1e-4)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha), device=self.device, dtype=torch.float32)

        # Store discount factors powers
        self.gamma_powers = torch.pow(self.gamma, torch.arange(self.sequence_length + 1, device=self.device)).unsqueeze(0) # (1, N+1)

    def select_action(self, state: dict, evaluate: bool = False) -> float:
        """Select action (normalized) based on the normalized last state."""
        state_trajectory = state['full_trajectory']
        if np.isnan(state_trajectory).any():
             return 0.0

        # Extract and normalize the last basic state
        last_basic_state_raw = torch.FloatTensor(state_trajectory[-1, :self.state_dim]).to(self.device).unsqueeze(0) # (1, state_dim)
        with torch.no_grad():
             # Use state normalizer in eval mode for action selection
             self.state_normalizer.eval()
             normalized_last_basic_state = self.state_normalizer.normalize(last_basic_state_raw)
             self.state_normalizer.train() # Set back to train mode

        self.actor.eval()
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor.forward(normalized_last_basic_state)
                action_normalized = torch.tanh(mean)
            else:
                action_normalized, _, _ = self.actor.sample(normalized_last_basic_state)
        self.actor.train()

        if torch.isnan(action_normalized).any():
             print("WARNING: NaN detected in final action_normalized in select_action! Returning 0.")
             return 0.0

        return action_normalized.detach().cpu().numpy()[0, 0]

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform T-SAC update using gradient averaging over N-step returns."""
        sampled_batch = memory.sample(batch_size)
        if sampled_batch is None: return None
        # state/next_state: (b, N, feat), action: (b, 1), reward/done: (b,)
        state_batch, action_batch_current, reward_batch, next_state_batch, done_batch = sampled_batch

        state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)
        # action_batch_current represents a_N
        action_batch_current_tensor = torch.FloatTensor(action_batch_current).to(self.device) # (b, 1)
        # reward_batch represents r_N
        reward_batch_tensor = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1) # (b, 1)
        next_state_batch_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        # done_batch represents d_N
        done_batch_tensor = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1) # (b, 1)

        # --- Update Running State Statistics ---
        raw_basic_states_batch = state_batch_tensor[:, :, :self.state_dim] # (b, N, state_dim)
        self.state_normalizer.update(raw_basic_states_batch.reshape(-1, self.state_dim))
        # --- End Update ---

        # --- Prepare Normalized States and Sequences ---
        # Initial state s_0 (for critic input)
        initial_state_raw = raw_basic_states_batch[:, 0, :] # (b, state_dim)
        normalized_initial_state_critic = self.state_normalizer.normalize(initial_state_raw) # (b, state_dim)

        # Action sequence a_1...a_N (for critic input)
        # Actions a_1..a_{N-1} are from state_batch history
        action_sequence_prefix = state_batch_tensor[:, 1:, self.state_dim:(self.state_dim + self.action_dim)] # (b, N-1, act_dim)
        # Action a_N is the current action taken
        action_sequence = torch.cat([action_sequence_prefix, action_batch_current_tensor.unsqueeze(1)], dim=1) # (b, N, act_dim)

        # Reward sequence r_1...r_N (for target calculation)
        # Rewards r_1..r_{N-1} are from state_batch history
        rewards_sequence_prefix = state_batch_tensor[:, 1:, -1] # (b, N-1)
        # Reward r_N is the current reward received
        rewards_sequence = torch.cat([rewards_sequence_prefix, reward_batch_tensor], dim=1) # (b, N)

        # Target states s_1...s_N (for V_tar calculation)
        # States s_1..s_{N-1} are from state_batch
        intermediate_states_raw = raw_basic_states_batch[:, 1:, :] # (b, N-1, state_dim)
        # State s_N is the final state from next_state_batch
        final_state_raw = next_state_batch_tensor[:, -1, :self.state_dim] # (b, state_dim)
        target_states_raw = torch.cat([intermediate_states_raw, final_state_raw.unsqueeze(1)], dim=1) # (b, N, state_dim)

        # Normalize target states
        batch_norm, seq_len_norm, state_dim_norm = target_states_raw.shape
        normalized_target_states = self.state_normalizer.normalize(
            target_states_raw.reshape(-1, state_dim_norm)
        ).reshape(batch_norm, seq_len_norm, state_dim_norm) # (b, N, state_dim)
        # --- End Prepare ---

        current_alpha = self.log_alpha.exp().detach() # Use detached alpha for target V

        # --- Calculate ALL n-step return targets G^(n) first ---
        targets_G_n = []
        self.actor.eval() # Actor in eval mode for target calculation
        self.critic1_target.eval()
        self.critic2_target.eval()

        with torch.no_grad():
            # Get actions and log_probs for *all* target states s_1...s_N
            # Input shape: (b*N, state_dim)
            flat_normalized_target_states = normalized_target_states.reshape(-1, self.state_dim)
            # Output shapes: (b*N, act_dim), (b*N, 1)
            next_actions_flat, next_log_probs_flat, _ = self.actor.sample(flat_normalized_target_states)

            # Reshape outputs back: (b, N, act_dim), (b, N, 1)
            next_actions_target = next_actions_flat.reshape(batch_norm, seq_len_norm, self.action_dim)
            next_log_probs_target = next_log_probs_flat.reshape(batch_norm, seq_len_norm, 1)

            # Get target Q values for all target states s_1...s_N
            # ** This is the tricky part. How to evaluate target Q(s_n, a'_n)? **
            # ** Assumption (Option 1 - requires robust critic): Feed sequence of length 1 **
            Q1_target_values = []
            Q2_target_values = []
            for n in range(self.sequence_length): # n = 0 to N-1, corresponds to s_1 to s_N
                state_sn = normalized_target_states[:, n, :] # (b, state_dim)
                action_an_prime = next_actions_target[:, n, :] # (b, act_dim)
                # Critic expects (b, state_dim), (b, 1, act_dim) -> gives (b, 1, 1)
                q1_sn_an_prime = self.critic1_target(state_sn, action_an_prime.unsqueeze(1)).squeeze(1) # (b, 1)
                q2_sn_an_prime = self.critic2_target(state_sn, action_an_prime.unsqueeze(1)).squeeze(1) # (b, 1)
                Q1_target_values.append(q1_sn_an_prime)
                Q2_target_values.append(q2_sn_an_prime)

            Q1_target_all = torch.stack(Q1_target_values, dim=1) # (b, N, 1)
            Q2_target_all = torch.stack(Q2_target_values, dim=1) # (b, N, 1)

            Q_target_min_all = torch.min(Q1_target_all, Q2_target_all) # (b, N, 1)
            V_target_all = Q_target_min_all - current_alpha * next_log_probs_target # (b, N, 1)

            # Calculate G^(n) for n=1 to N using rewards r_1...r_N and V_tar(s_1)...V_tar(s_N)
            cumulative_rewards = torch.zeros_like(reward_batch_tensor).to(self.device) # (b, 1)
            # gamma_powers shape (1, N+1), select relevant powers
            for n in range(1, self.sequence_length + 1): # n = 1..N
                 # Sum rewards r_1 to r_n
                 reward_sum = torch.sum(rewards_sequence[:, :n] * self.gamma_powers[:, :n], dim=1, keepdim=True) # (b, 1)

                 # Get V_tar(s_n) - corresponds to index n-1 in V_target_all
                 V_target_sn = V_target_all[:, n-1, :] # (b, 1)

                 # Done mask only applies if n == N (i.e., we bootstrap from s_N)
                 # If intermediate step k < N leads to termination, the rewards r_{k+1}..r_N should be 0
                 # and V_tar(s_k) should also be 0? Standard n-step doesn't usually handle this explicitly,
                 # relies on V(terminal)=0. Let's assume episode doesn't terminate mid-trajectory segment.
                 # The done_batch_tensor applies only to the final bootstrap term from s_N.
                 done_mask_n = done_batch_tensor if n == self.sequence_length else torch.zeros_like(done_batch_tensor)

                 # G^(n) = sum_{k=1..n} gamma^(k-1) * r_k + gamma^n * V_tar(s_n) * (1-d_n)
                 # Our gamma_powers[0, n] = gamma^n
                 target_G = reward_sum + self.gamma_powers[:, n] * V_target_sn * (1.0 - done_mask_n)
                 targets_G_n.append(target_G)

        # --- Critic Update using Gradient Averaging ---
        self.critic1.train(); self.critic2.train() # Ensure critics are in train mode
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        # Get Q predictions for the actual sequence (s0, a1...aN)
        # Q_pred_seq shape: (b, N, 1)
        Q1_pred_seq = self.critic1(normalized_initial_state_critic, action_sequence)
        Q2_pred_seq = self.critic2(normalized_initial_state_critic, action_sequence)

        total_critic1_loss = 0.0
        total_critic2_loss = 0.0

        # Loop n=1 to N, calculate loss L_n, backpropagate gradient L_n / N
        for n in range(1, self.sequence_length + 1):
            q1_pred_n = Q1_pred_seq[:, n-1, :] # Prediction corresponding to (s0, a1..an)
            q2_pred_n = Q2_pred_seq[:, n-1, :]
            target_G_n = targets_G_n[n-1].detach() # Target G^(n)

            loss1_n = F.mse_loss(q1_pred_n, target_G_n)
            loss2_n = F.mse_loss(q2_pred_n, target_G_n)

            # Accumulate total loss for reporting
            total_critic1_loss += loss1_n.item()
            total_critic2_loss += loss2_n.item()

            # Average gradient over N steps
            retain_graph = (n < self.sequence_length) # Keep graph for all but last step
            (loss1_n / self.sequence_length).backward(retain_graph=retain_graph)
            (loss2_n / self.sequence_length).backward(retain_graph=retain_graph)

        # Clip gradients and step optimizers
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        avg_critic1_loss = total_critic1_loss / self.sequence_length
        avg_critic2_loss = total_critic2_loss / self.sequence_length

        # --- Actor and Alpha Update ---
        # Freeze critics to avoid computing gradients for them during actor update
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        self.actor.train()
        # Get policy action and log_prob for the initial state s_0
        action_pi_t0, log_prob_pi_t0, _ = self.actor.sample(normalized_initial_state_critic)

        # We need Q(s_0, a_pi_0). Evaluate critics with a_pi_0 as the *first* action.
        # Need to construct the action sequence [a_pi_0, a_2, ..., a_N] ?
        # No, standard SAC actor loss uses Q(s, a_pi(s)). For T-SAC, this likely means Q(s0, a_pi(s0)),
        # which corresponds to the *first* output of the critic when fed the sequence starting with a_pi_0.
        action_sequence_for_actor_loss = torch.cat([action_pi_t0.unsqueeze(1), action_sequence[:, 1:, :]], dim=1)

        q1_pi_seq = self.critic1(normalized_initial_state_critic, action_sequence_for_actor_loss)
        q2_pi_seq = self.critic2(normalized_initial_state_critic, action_sequence_for_actor_loss)
        q1_pi_t0 = q1_pi_seq[:, 0, :] # Get the Q-value corresponding to the first action a_pi_t0
        q2_pi_t0 = q2_pi_seq[:, 0, :]
        q_pi_min_t0 = torch.min(q1_pi_t0, q2_pi_t0)

        current_alpha_val = self.log_alpha.exp() # Get current alpha value (might be updated)
        actor_loss = (current_alpha_val * log_prob_pi_t0 - q_pi_min_t0).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Unfreeze critics
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # Alpha update
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            # Use the log_prob from the actor update step
            alpha_loss = -(self.log_alpha * (log_prob_pi_t0.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item() # Update internal alpha value
            alpha_loss_val = alpha_loss.item()

        # --- Target Network Update ---
        with torch.no_grad():
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic1_loss': avg_critic1_loss,
            'critic2_loss': avg_critic2_loss,
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss_val,
            'alpha': self.alpha
        }

    def save_model(self, path: str):
        """Saves the model state including normalizer."""
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
            'state_normalizer_state_dict': self.state_normalizer.state_dict(),
            'device_type': self.device.type
        }
        if self.auto_tune_alpha:
            save_dict['log_alpha_value'] = self.log_alpha.data
            if hasattr(self, 'alpha_optimizer'):
                save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        """Loads the model state including normalizer."""
        if not os.path.exists(path):
            print(f"Warn: T-SAC model not found: {path}. Skipping load.")
            return
        print(f"Loading T-SAC model from {path}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

            if 'state_normalizer_state_dict' in checkpoint:
                self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
                print("Loaded T-SAC state normalizer statistics.")
            else:
                print("Warning: T-SAC state normalizer statistics not found in checkpoint.")

            if self.auto_tune_alpha:
                if 'log_alpha_value' in checkpoint:
                    # Ensure log_alpha is Parameter before loading data
                    if not isinstance(self.log_alpha, nn.Parameter):
                         self.log_alpha = nn.Parameter(torch.zeros_like(checkpoint['log_alpha_value']))
                    self.log_alpha.data.copy_(checkpoint['log_alpha_value'])
                else:
                    print("Warn: log_alpha_value not found in checkpoint. Using initial alpha.")
                # Re-create optimizer AFTER loading log_alpha
                self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=self.config.lr, weight_decay=1e-4) # Use actor LR default
                if 'alpha_optimizer_state_dict' in checkpoint:
                    try:
                        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                    except Exception as e:
                        print(f"Warning: Could not load alpha optimizer state_dict: {e}. Optimizer state reset.")
                else:
                    print("Warn: Alpha optimizer state dict not found in checkpoint.")
                self.alpha = self.log_alpha.exp().item()
            elif 'log_alpha_value' in checkpoint:
                 self.log_alpha = checkpoint['log_alpha_value'].to(self.device)
                 self.alpha = self.log_alpha.exp().item()

            # Sync targets AFTER loading main critics
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())
            for p in self.critic1_target.parameters(): p.requires_grad = False
            for p in self.critic2_target.parameters(): p.requires_grad = False

            print(f"T-SAC model loaded successfully from {path}")
            self.actor.train()
            self.critic1.train()
            self.critic2.train()
            self.critic1_target.eval()
            self.critic2_target.eval()

        except KeyError as e:
            print(f"Error loading T-SAC model: Missing key {e}. Model may be partially loaded.")
        except Exception as e:
            print(f"Error loading T-SAC model from {path}: {e}")


# --- Training Loop (train_tsac - Requires config adjustments) ---
def train_tsac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    # --- Adjust Configs based on Paper (BoxPushing Sparse T-SAC column) ---
    print("Applying Paper Hyperparameters for T-SAC Training...")
    config.tsac.lr = 2.5e-4  # Actor LR
    config.tsac.critic_lr = 3e-5 # Specific Critic LR
    config.tsac.hidden_dims = [2048] # Actor MLP layers [512*4]
    # config.tsac.hidden_dims = [512, 512] # Alternative smaller actor MLP
    config.tsac.use_layer_norm_actor = True # Enable LayerNorm in Actor (per paper ablation)
    config.tsac.embedding_dim = 256 # 4 heads * 64 dim/head
    config.tsac.transformer_n_heads = 4
    config.tsac.transformer_n_layers = 2
    config.tsac.transformer_hidden_dim = 512 # Check paper's default? Using previous value for now.
    config.tsac.alpha = 0.1 # Starting alpha
    config.tsac.auto_tune_alpha = False # Paper uses fixed alpha for BoxPush sparse
    config.tsac.tau = 2e-3 # Polyak weight

    config.training.batch_size = 256
    config.training.learning_starts = 5000
    config.training.train_freq = 1 # Update every step
    config.training.gradient_steps = 1 # 1 update per step (UTD=1)
    config.replay_buffer.capacity = 20000

    config.world.trajectory_length = 4 # Paper uses N=4 for BoxPush

    # Ensure world_config reflects new trajectory length for buffer init etc.
    world_config = config.world
    world_config.trajectory_length = config.world.trajectory_length
    world_config.trajectory_feature_dim = CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM

    tsac_config = config.tsac
    train_config = config.training
    buffer_config = config.replay_buffer
    cuda_device = config.cuda_device
    # --- End Config Adjustments ---


    learning_starts = train_config.learning_starts
    gradient_steps = train_config.gradient_steps
    train_freq = train_config.train_freq
    batch_size = train_config.batch_size
    log_frequency_ep = train_config.log_frequency
    save_interval_ep = train_config.save_interval

    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Warn: Multi-GPU not standard for T-SAC. Using: {cuda_device}")
            device = torch.device(cuda_device)
        else:
            device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try: torch.cuda.set_device(device); print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e: print(f"Warn: Could not set CUDA device {cuda_device}. E: {e}"); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu"); print("GPU not available, using CPU.")

    log_dir = os.path.join("runs", f"tsac_gradavg_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    agent = TSAC(config=tsac_config, world_config=world_config, device=device)
    memory = ReplayBuffer(config=buffer_config, world_config=world_config)
    os.makedirs(train_config.models_dir, exist_ok=True)

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
            parts = os.path.basename(latest_model_path).split('_')
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            step_part = next((p for p in parts if p.startswith('step')), None)
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            if step_part: total_steps = int(step_part.replace('step', '').split('.')[0])
            print(f"Resuming at step {total_steps}, episode {start_episode}")
        except (IndexError, ValueError, StopIteration) as e:
            print(f"Warn: Could not parse steps/episode from filename ({latest_model_path}): {e}. Starting from scratch.")
            total_steps = 0; start_episode = 1
    else:
        print("\nStarting T-SAC training from scratch.")

    episode_rewards = []
    all_losses = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1), desc="Training T-SAC", unit="episode", initial=start_episode-1, total=train_config.num_episodes)
    world = World(world_config=world_config) # Use updated world_config

    for episode in pbar:
        world.reset()
        state = world.encode_state()
        episode_reward = 0
        episode_steps = 0
        episode_losses_temp = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
        updates_made_this_episode = 0

        for step_in_episode in range(train_config.max_steps):
            action_normalized = agent.select_action(state, evaluate=False)
            step_start_time = time.time()
            world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            timing_metrics['env_step_time'].append(time.time() - step_start_time)
            reward, next_state, done = world.reward, world.encode_state(), world.done
            memory.push(state, action_normalized, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if total_steps >= learning_starts and total_steps % train_freq == 0:
                for _ in range(gradient_steps):
                    if len(memory) >= batch_size:
                        update_start_time = time.time()
                        losses = agent.update_parameters(memory, batch_size)
                        timing_metrics['parameter_update_time'].append(time.time() - update_start_time)
                        if losses:
                            # Check for NaN in losses before appending
                            if not any(np.isnan(v) for k, v in losses.items() if k != 'alpha'): # Exclude alpha itself
                                for key, val in losses.items():
                                     if isinstance(val, (float, np.float64)):
                                        episode_losses_temp[key].append(val)
                                updates_made_this_episode += 1
                            else:
                                print(f"INFO: Skipping T-SAC loss logging at step {total_steps} due to NaN.")
                        else:
                            break # Stop gradient steps if update fails
                    else:
                        break # Stop gradient steps if buffer not full enough
            if done: break

        # --- Logging ---
        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else float('nan') for k, v in episode_losses_temp.items()} # Use NaN if no updates
        if updates_made_this_episode > 0:
             if not np.isnan(avg_losses['critic1_loss']): all_losses['critic1_loss'].append(avg_losses['critic1_loss'])
             if not np.isnan(avg_losses['critic2_loss']): all_losses['critic2_loss'].append(avg_losses['critic2_loss'])
             if not np.isnan(avg_losses['actor_loss']): all_losses['actor_loss'].append(avg_losses['actor_loss'])
             if not np.isnan(avg_losses['alpha']): all_losses['alpha'].append(avg_losses['alpha'])
             if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): all_losses['alpha_loss'].append(avg_losses['alpha_loss'])

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
                if not np.isnan(avg_losses['critic1_loss']): writer.add_scalar('Loss/Critic1_AvgEp', avg_losses['critic1_loss'], total_steps)
                if not np.isnan(avg_losses['critic2_loss']): writer.add_scalar('Loss/Critic2_AvgEp', avg_losses['critic2_loss'], total_steps)
                if not np.isnan(avg_losses['actor_loss']): writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): writer.add_scalar('Loss/Alpha_AvgEp', avg_losses['alpha_loss'], total_steps)
                if not np.isnan(avg_losses['alpha']): writer.add_scalar('Alpha/Value', avg_losses['alpha'], total_steps)
            else:
                 writer.add_scalar('Alpha/Value', agent.alpha, total_steps) # Log current alpha if no updates

            if agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/TSAC_Normalizer_Count', agent.state_normalizer.count.item(), total_steps)
                writer.add_scalar('Stats/TSAC_Normalizer_Mean_Feat0', agent.state_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/TSAC_Normalizer_Std_Feat0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), total_steps)

            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar_postfix = {'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps, 'alpha': f"{agent.alpha:.3f}"}
            if updates_made_this_episode and not np.isnan(avg_losses['critic1_loss']): pbar_postfix['c1_loss'] = f"{avg_losses['critic1_loss']:.2f}"
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
         evaluate_tsac(agent=agent, config=config) # Pass the possibly modified config

    return agent, episode_rewards


# --- Evaluation Loop (evaluate_tsac - No changes needed from previous SAC version) ---
def evaluate_tsac(agent: TSAC, config: DefaultConfig):
    """Evaluates the trained T-SAC agent."""
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

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

    agent.actor.eval(); agent.critic1.eval(); agent.critic2.eval()
    agent.state_normalizer.eval() # Set normalizer to eval mode

    print(f"\nRunning T-SAC Evaluation for {eval_config.num_episodes} episodes...")
    # Make sure world uses the same config (esp. trajectory_length) as agent
    world = World(world_config=world_config)

    for episode in range(eval_config.num_episodes):
        world.reset()
        state = world.encode_state()
        episode_reward = 0
        episode_frames = []

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            try:
                fname = f"tsac_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config, filename=fname, collect_for_gif=True)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warn: Vis failed init state ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            # select_action uses normalizer in eval mode internally
            action_normalized = agent.select_action(state, evaluate=True)
            world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward, next_state, done = world.reward, world.encode_state(), world.done

            if eval_config.render and vis_available:
                try:
                    fname = f"tsac_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config, filename=fname, collect_for_gif=True)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")

            state = next_state
            episode_reward += reward
            if done:
                break

        success = world.error_dist <= world_config.success_threshold
        status = "Success!" if success else "Failure."
        if done:
            if success: success_count += 1
            print(f"  Episode {episode+1}: Terminated Step {step+1}. Final Err: {world.error_dist:.2f}. {status}")
        else: # Max steps reached
             if success: success_count +=1
             print(f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps}). Final Err: {world.error_dist:.2f}. {status}")

        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        if eval_config.render and vis_available and episode_frames:
            gif_filename = f"tsac_eval_episode_{episode+1}.gif"
            try:
                print(f"  Saving GIF for T-SAC episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"  Warn: Failed GIF save ep {episode+1}. E: {e}")
            if vis_config.delete_frames_after_gif:
                 cleaned_count = 0
                 for frame in episode_frames:
                     if os.path.exists(frame):
                         try: os.remove(frame); cleaned_count += 1
                         except OSError as ose: print(f"    Warn: Could not delete TSAC frame file {frame}: {ose}")

    agent.actor.train(); agent.critic1.train(); agent.critic2.train()
    agent.state_normalizer.train() # Set normalizer back to train mode

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0
    print("\n--- T-SAC Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Reward: {avg_eval_reward:.2f} +/- {std_eval_reward:.2f}")
    print(f"Success Rate (Error <= {world_config.success_threshold}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_episode_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering was enabled but visualization libraries were not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End T-SAC Evaluation ---\n")
    return eval_rewards, success_rate