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
from configs import DefaultConfig, ReplayBufferConfig, TSACConfig, WorldConfig, CORE_STATE_DIM
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, List

# --- Replay Buffer ---
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
             # print(f"Warn: Push traj shape {trajectory.shape if isinstance(trajectory, np.ndarray) else type(trajectory)} != ({self.trajectory_length}, {self.feature_dim})")
             return # Avoid adding malformed data
        if not isinstance(next_trajectory, np.ndarray) or next_trajectory.shape != (self.trajectory_length, self.feature_dim):
             # print(f"Warn: Push next_traj shape {next_trajectory.shape if isinstance(next_trajectory, np.ndarray) else type(next_trajectory)} != ({self.trajectory_length}, {self.feature_dim})")
             return

        self.buffer.append((trajectory, float(action), float(reward), next_trajectory, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        if len(self.buffer) < batch_size:
            return None
        try:
            batch = random.sample(self.buffer, batch_size)
        except ValueError: # Handle case where batch_size > len(self.buffer) after check (race condition?)
            print(f"Warning: Sampling failed. len(buffer)={len(self.buffer)}, batch_size={batch_size}")
            return None

        trajectory, action, reward, next_trajectory, done = zip(*batch)

        try:
            trajectory_arr = np.array(trajectory, dtype=np.float32)
            action_arr = np.array(action, dtype=np.float32).reshape(-1, 1)
            reward_arr = np.array(reward, dtype=np.float32)
            next_trajectory_arr = np.array(next_trajectory, dtype=np.float32)
            done_arr = np.array(done, dtype=np.float32)
        except Exception as e:
            print(f"Error converting sampled batch to numpy arrays: {e}")
            return None

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
        self.feature_dim = world_config.trajectory_feature_dim
        self.sequence_length = world_config.trajectory_length

        self.layers = nn.ModuleList()
        mlp_input_dim = self.state_dim # Input is the last basic state
        current_dim = mlp_input_dim
        for i, hidden_dim in enumerate(config.hidden_dims):
            linear_layer = nn.Linear(current_dim, hidden_dim)
            nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
            if linear_layer.bias is not None:
                 fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
                 bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                 nn.init.uniform_(linear_layer.bias, -bound, bound)
            self.layers.append(linear_layer)
            if self.use_layer_norm:
                 self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.mean = nn.Linear(current_dim, self.action_dim)
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)
        self.log_std = nn.Linear(current_dim, self.action_dim)
        nn.init.xavier_uniform_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, 0)
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, state_trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using only the last basic state."""
        last_basic_state = state_trajectory[:, -1, :self.state_dim] # (batch, state_dim)
        x = last_basic_state
        for layer in self.layers:
            x = layer(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        if torch.isnan(mean).any() or torch.isnan(log_std).any():
            print("WARNING: Actor forward pass produced NaN mean or log_std!")
        return mean, log_std

    def sample(self, state_trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (normalized) from the policy distribution with safer logprob."""
        mean, log_std = self.forward(state_trajectory)
        std = log_std.exp()
        if torch.isnan(std).any() or (std <= 1e-8).any(): # Use small epsilon for std check
            print(f"WARNING: Actor produced invalid std (NaN or ~0): {std.detach().cpu().numpy()}. Clamping log_std.")
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max - 0.1)
            std = log_std.exp()
            if torch.isnan(std).any() or (std <= 1e-8).any():
                 print("ERROR: Actor std still invalid after clamping log_std. Returning default action.")
                 dummy_action = torch.zeros_like(mean)
                 dummy_log_prob = torch.full((mean.shape[0], 1), -100.0, device=mean.device)
                 dummy_mean_action = torch.tanh(torch.zeros_like(mean))
                 return dummy_action, dummy_log_prob, dummy_mean_action

        normal = Normal(mean, std)
        try:
            x_t = normal.rsample() # Represents action in unbounded space
        except ValueError as e:
            print(f"ERROR: Normal distribution rsample failed: {e}")
            print(f"Mean: {mean.detach().cpu().numpy()}, Std: {std.detach().cpu().numpy()}")
            dummy_action = torch.zeros_like(mean)
            dummy_log_prob = torch.full((mean.shape[0], 1), -100.0, device=mean.device)
            dummy_mean_action = torch.tanh(torch.zeros_like(mean))
            return dummy_action, dummy_log_prob, dummy_mean_action

        action_normalized = torch.tanh(x_t) # Squashed action [-1, 1]

        # --- Safer Log Prob Calculation ---
        log_prob_unbounded = normal.log_prob(x_t)
        clamped_tanh = action_normalized.clamp(-0.999999, 0.999999)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)
        # --- End Safer Log Prob ---

        if torch.isnan(log_prob).any():
            print(f"WARNING: Actor sample produced NaN log_prob! mean={mean.detach().cpu().numpy()}, std={std.detach().cpu().numpy()}, x_t={x_t.detach().cpu().numpy()}")
            # log_prob = torch.where(torch.isnan(log_prob), torch.full_like(log_prob, -100.0), log_prob)

        return action_normalized, log_prob, torch.tanh(mean)

    def sample_logprob_for_state(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
         """ Helper to sample action and logprob for a specific basic state."""
         batch_size = state_tensor.shape[0]
         device = state_tensor.device
         dummy_trajectory = torch.zeros(batch_size, self.sequence_length, self.feature_dim, device=device)
         if torch.isnan(state_tensor).any():
              print(f"WARNING: Input state_tensor to sample_logprob_for_state contains NaN!")
              return torch.zeros(batch_size, self.action_dim, device=device), torch.full((batch_size, 1), -100.0, device=device)
         dummy_trajectory[:, -1, :self.state_dim] = state_tensor
         action_normalized, log_prob, _ = self.sample(dummy_trajectory)
         return action_normalized, log_prob


# --- T-SAC Critic (Transformer-based) ---
class TransformerCritic(nn.Module):
    """Q-function network (Critic) for T-SAC using a Transformer."""
    def __init__(self, config: TSACConfig, world_config: WorldConfig):
        super(TransformerCritic, self).__init__()
        self.config = config
        self.world_config = world_config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.embedding_dim = config.embedding_dim
        self.sequence_length = world_config.trajectory_length
        self.transformer_seq_len = self.sequence_length + 1

        self.state_embed = nn.Linear(self.state_dim, self.embedding_dim)
        self.action_embed = nn.Linear(self.action_dim, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(self.embedding_dim, max_len=self.transformer_seq_len + 5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=config.transformer_n_heads,
            dim_feedforward=config.transformer_hidden_dim, batch_first=True, activation='relu', dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_n_layers, norm=nn.LayerNorm(self.embedding_dim))
        self.q_head = nn.Linear(self.embedding_dim, 1)

    def _generate_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """ Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, initial_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer critic."""
        batch_size, seq_len, _ = action_sequence.shape
        device = initial_state.device
        state_emb = self.state_embed(initial_state).unsqueeze(1)
        action_emb = self.action_embed(action_sequence)
        transformer_input_emb = torch.cat([state_emb, action_emb], dim=1)
        transformer_input_pos = self.pos_encoder(transformer_input_emb)
        current_transformer_seq_len = seq_len + 1
        mask = self._generate_mask(current_transformer_seq_len, device)
        transformer_output = self.transformer_encoder(transformer_input_pos, mask=mask, is_causal=False)
        action_transformer_output = transformer_output[:, 1:, :]
        q_predictions = self.q_head(action_transformer_output)
        if torch.isnan(q_predictions).any():
             print("WARNING: TransformerCritic forward pass produced NaN q_predictions!")
        return q_predictions


# --- T-SAC Agent ---
class TSAC:
    """Transformer-based Soft Actor-Critic aligned with paper's multi-step approach."""
    def __init__(self, config: TSACConfig, world_config: WorldConfig, device: torch.device = None):
        self.config = config
        self.world_config = world_config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha
        self.sequence_length = world_config.trajectory_length # N
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim # Basic state dim

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"T-SAC Agent using device: {self.device} (Paper Aligned Version)")
        print(f"T-SAC using Sequence Length (N): {self.sequence_length}")

        # Initialize networks
        self.actor = Actor(config, world_config).to(self.device)
        self.critic1 = TransformerCritic(config, world_config).to(self.device)
        self.critic2 = TransformerCritic(config, world_config).to(self.device)
        self.critic1_target = TransformerCritic(config, world_config).to(self.device)
        self.critic2_target = TransformerCritic(config, world_config).to(self.device)

        # Initialize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.lr)

        # Initialize alpha
        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = nn.Parameter(torch.tensor(np.log(self.alpha), device=self.device))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device)


    def select_action(self, state: dict, evaluate: bool = False) -> float:
        """Select action (normalized yaw change) based on the *last* state in the trajectory."""
        state_trajectory = state['full_trajectory']
        if np.isnan(state_trajectory).any():
             print("WARNING: NaN detected in state_trajectory input to select_action!")
             return 0.0

        state_tensor = torch.FloatTensor(state_trajectory).to(self.device).unsqueeze(0)

        self.actor.eval() # Ensure actor is in eval mode for action selection
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor.forward(state_tensor)
                action_normalized = torch.tanh(mean)
            else:
                action_normalized, _, _ = self.actor.sample(state_tensor)
        self.actor.train() # Set back to train mode

        if torch.isnan(action_normalized).any():
             print("WARNING: NaN detected in final action_normalized in select_action!")
             return 0.0

        return action_normalized.detach().cpu().numpy()[0, 0]


    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform T-SAC update aligned with paper's multi-step gradient averaging."""
        sampled_batch = memory.sample(batch_size)
        if sampled_batch is None: return None
        state_batch, action_batch_current, reward_batch, next_state_batch, done_batch = sampled_batch

        if np.isnan(state_batch).any() or np.isnan(action_batch_current).any() or \
           np.isnan(reward_batch).any() or np.isnan(next_state_batch).any() or np.isnan(done_batch).any():
            print("WARNING: NaN detected in sampled batch from replay buffer! Skipping update.")
            return None

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch_current_tensor = torch.FloatTensor(action_batch_current).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        initial_state_critic = state_batch[:, 0, :self.state_dim]
        action_sequence_prefix = state_batch[:, 1:, self.state_dim:(self.state_dim + self.action_dim)]
        action_sequence = torch.cat([action_sequence_prefix, action_batch_current_tensor.unsqueeze(1)], dim=1)
        rewards_sequence_prefix = state_batch[:, 1:, -1]
        rewards_sequence = torch.cat([rewards_sequence_prefix, reward_batch], dim=1)

        # --- Critic Update ---
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        current_q1_preds_seq = self.critic1(initial_state_critic, action_sequence)
        current_q2_preds_seq = self.critic2(initial_state_critic, action_sequence)

        if torch.isnan(current_q1_preds_seq).any() or torch.isnan(current_q2_preds_seq).any():
             print(f"WARNING: NaN detected in current critic predictions! Skipping update.")
             return None

        current_alpha_val = self.log_alpha.exp().item() # Scalar value

        cumulative_rewards = torch.zeros_like(reward_batch).to(self.device)
        gamma_pow = 1.0
        critic_total_loss1 = 0.0
        critic_total_loss2 = 0.0
        target_G_n_list = []

        # --- Target Calculation and Loss Accumulation ---
        self.actor.eval() # Use eval mode for target calculation stability
        self.critic1_target.eval()
        self.critic2_target.eval()
        with torch.no_grad(): # Target calculations should not have gradients
            for n in range(1, self.sequence_length + 1):
                current_reward = rewards_sequence[:, n-1].unsqueeze(1)
                cumulative_rewards += gamma_pow * current_reward

                if n < self.sequence_length:
                    s_t_plus_n = state_batch[:, n, :self.state_dim]
                    is_done_n = torch.zeros_like(done_batch)
                else:
                    s_t_plus_n = next_state_batch[:, -1, :self.state_dim]
                    is_done_n = done_batch

                next_action_n, next_log_prob_n = self.actor.sample_logprob_for_state(s_t_plus_n)

                if torch.isnan(next_action_n).any() or torch.isnan(next_log_prob_n).any():
                     print(f"WARNING: NaN detected in actor output for target V(s_{n})! Skipping update.")
                     self.actor.train() # Revert mode
                     return None

                q1_target_n = self.critic1_target(s_t_plus_n, next_action_n.unsqueeze(1))[:, 0]
                q2_target_n = self.critic2_target(s_t_plus_n, next_action_n.unsqueeze(1))[:, 0]

                if torch.isnan(q1_target_n).any() or torch.isnan(q2_target_n).any():
                     print(f"WARNING: NaN detected in target critic output for target V(s_{n})! Skipping update.")
                     self.actor.train()
                     return None

                q_target_min_n = torch.min(q1_target_n, q2_target_n)
                v_target_n = q_target_min_n - current_alpha_val * next_log_prob_n
                target_G_n = cumulative_rewards + (gamma_pow * self.gamma) * (1.0 - is_done_n) * v_target_n
                target_G_n_list.append(target_G_n)

                if torch.isnan(target_G_n).any():
                     print(f"WARNING: NaN detected in final target G({n})! Skipping update.")
                     self.actor.train()
                     return None

                gamma_pow *= self.gamma # Update gamma_pow for the *next* iteration's reward

        self.actor.train() # Revert actor to train mode

        # --- Gradient Averaging Backward Pass ---
        # Must calculate losses outside no_grad block if using preds from trainable critics
        for n in range(1, self.sequence_length + 1):
            q1_pred_n = current_q1_preds_seq[:, n-1, :]
            q2_pred_n = current_q2_preds_seq[:, n-1, :]
            target_G_n = target_G_n_list[n-1] # Retrieve stored target

            loss1_n = F.mse_loss(q1_pred_n, target_G_n.detach()) # Use stored target
            loss2_n = F.mse_loss(q2_pred_n, target_G_n.detach())

            critic_total_loss1 += loss1_n
            critic_total_loss2 += loss2_n

            is_last_step = (n == self.sequence_length)
            # Need retain_graph=True because Q-preds come from same network,
            # and the graph might be needed later for actor update if computed before critic step.
            # Safer to retain graph until all backward passes for critics are done.
            (loss1_n / self.sequence_length).backward(retain_graph=True)
            (loss2_n / self.sequence_length).backward(retain_graph=not is_last_step) # Can release graph on last step if actor loss computed AFTER critic step

        # Check accumulated loss before stepping
        if torch.isnan(critic_total_loss1).any() or torch.isnan(critic_total_loss2).any():
             print(f"WARNING: NaN detected in accumulated critic loss before step! Skipping update.")
             self.critic1_optimizer.zero_grad()
             self.critic2_optimizer.zero_grad()
             return None

        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        critic1_loss_item = (critic_total_loss1 / self.sequence_length).item()
        critic2_loss_item = (critic_total_loss2 / self.sequence_length).item()

        # --- Actor Update ---
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        action_pi_t, log_prob_pi_t = self.actor.sample_logprob_for_state(initial_state_critic)

        if torch.isnan(action_pi_t).any() or torch.isnan(log_prob_pi_t).any():
            print(f"WARNING: NaN detected in actor output for actor loss! Skipping update.")
            for p in self.critic1.parameters(): p.requires_grad = True
            for p in self.critic2.parameters(): p.requires_grad = True
            return None

        action_sequence_pi = torch.cat([action_pi_t.unsqueeze(1), action_sequence[:, 1:, :]], dim=1)
        # Recompute Q-values with updated critics for actor loss (as done in SAC)
        # This requires critics to be in train mode and graph retention in critic backward pass.
        # Alternatively, use the Q-values computed before the critic step (more like TD3)? Let's stick to SAC style.
        q1_pi_seq = self.critic1(initial_state_critic, action_sequence_pi)
        q2_pi_seq = self.critic2(initial_state_critic, action_sequence_pi)


        if torch.isnan(q1_pi_seq).any() or torch.isnan(q2_pi_seq).any():
            print(f"WARNING: NaN detected in critic output for actor loss! Skipping update.")
            for p in self.critic1.parameters(): p.requires_grad = True
            for p in self.critic2.parameters(): p.requires_grad = True
            return None

        q1_pi = q1_pi_seq[:, 0, :]
        q2_pi = q2_pi_seq[:, 0, :]
        q_pi_min = torch.min(q1_pi, q2_pi)

        actor_loss = (current_alpha_val * log_prob_pi_t - q_pi_min).mean()

        actor_loss_item = float('nan')
        if torch.isnan(actor_loss):
            print(f"WARNING: NaN detected in actor loss value! Skipping actor update step.")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            actor_loss_item = actor_loss.item()

        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # --- Alpha Update ---
        alpha_loss_item = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_pi_t.detach() + self.target_entropy)).mean()
            if torch.isnan(alpha_loss):
                 print(f"WARNING: NaN detected in alpha loss value! Skipping alpha update step.")
                 self.alpha_optimizer.zero_grad()
            else:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            alpha_loss_item = alpha_loss.item() if not torch.isnan(alpha_loss) else float('nan')

        # --- Target Network Update ---
        with torch.no_grad():
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic1_loss': critic1_loss_item,
            'critic2_loss': critic2_loss_item,
            'actor_loss': actor_loss_item,
            'alpha_loss': alpha_loss_item,
            'alpha': self.alpha
        }

    def save_model(self, path: str):
        """Saves the model state."""
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
            save_dict['log_alpha_value'] = self.log_alpha.data # Save the tensor value
            if hasattr(self, 'alpha_optimizer'):
                save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        """Loads the model state."""
        if not os.path.exists(path):
            print(f"Warn: T-SAC model not found: {path}. Skipping load.")
            return
        print(f"Loading T-SAC model from {path}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
            self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

            if self.auto_tune_alpha:
                if 'log_alpha_value' in checkpoint:
                    # Ensure log_alpha is a Parameter before loading data
                    if not isinstance(self.log_alpha, nn.Parameter):
                         self.log_alpha = nn.Parameter(self.log_alpha) # Convert if necessary
                    self.log_alpha.data.copy_(checkpoint['log_alpha_value'])
                else:
                    print("Warn: log_alpha_value not found in checkpoint. Using initial alpha.")

                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
                if 'alpha_optimizer_state_dict' in checkpoint:
                    try: self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                    except Exception as e: print(f"Warning: Could not load alpha optimizer state_dict: {e}. Optimizer state reset.")
                else: print("Warn: Alpha optimizer state dict not found in checkpoint.")
                self.alpha = self.log_alpha.exp().item()
            elif 'log_alpha_value' in checkpoint:
                 # Still load if not auto-tuning now but value exists in checkpoint
                 self.log_alpha = checkpoint['log_alpha_value'] # Load as tensor
                 if isinstance(self.log_alpha, nn.Parameter): # Detach if it became parameter
                     self.log_alpha = self.log_alpha.data
                 self.alpha = self.log_alpha.exp().item()


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
            print(f"Error loading model: Missing key {e} in checkpoint file {path}. Model may be partially loaded.")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")


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

    log_dir = os.path.join("runs", f"tsac_paper_aligned_{int(time.time())}")
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
    all_losses = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': []}
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1), desc="Training T-SAC (Paper Aligned)", unit="episode", initial=start_episode-1, total=train_config.num_episodes)
    world = World(world_config=world_config)

    for episode in pbar:
        world.reset()
        state = world.encode_state()
        episode_reward = 0
        episode_steps = 0
        episode_losses_temp = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
        updates_made_this_episode = 0

        for step_in_episode in range(train_config.max_steps):
            agent.actor.train(); agent.critic1.train(); agent.critic2.train() # Ensure train mode
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
                             # Use np.float64 for NumPy 2.0 compatibility
                            if not any(np.isnan(v) for v in losses.values() if isinstance(v, (float, np.float64))):
                                for key, val in losses.items():
                                    if isinstance(val, (float, np.float64)):
                                        episode_losses_temp[key].append(val)
                                updates_made_this_episode += 1
                            else:
                                print("INFO: Skipping loss logging due to NaN in update results.")
                        else:
                            print("INFO: agent.update_parameters returned None, skipping gradient step.")
                            break
                    else:
                        break
            if done: break

        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else 0 for k, v in episode_losses_temp.items()}
        if updates_made_this_episode > 0:
             if not np.isnan(avg_losses['critic1_loss']): all_losses['critic1_loss'].append(avg_losses['critic1_loss'])
             if not np.isnan(avg_losses['critic2_loss']): all_losses['critic2_loss'].append(avg_losses['critic2_loss'])
             if not np.isnan(avg_losses['actor_loss']): all_losses['actor_loss'].append(avg_losses['actor_loss'])
             if not np.isnan(avg_losses['alpha']): all_losses['alpha'].append(avg_losses['alpha'])

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
                writer.add_scalar('Alpha/Value', agent.alpha, total_steps)
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
    print(f"T-SAC (Paper Aligned) Training finished. Total steps: {total_steps}")
    final_save_path = os.path.join(train_config.models_dir, f"tsac_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_tsac(agent=agent, config=config)

    return agent, episode_rewards


# --- Evaluation Loop ---
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
    agent.actor.eval(); agent.critic1.eval(); agent.critic2.eval() # Set all relevant models to eval
    print(f"\nRunning T-SAC Evaluation for {eval_config.num_episodes} episodes...")
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
                elif initial_frame_file: print(f"Warn: Initial frame path returned but not found: {initial_frame_file}")
            except Exception as e: print(f"Warn: Vis failed init state ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            action_normalized = agent.select_action(state, evaluate=True)
            world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward, next_state, done = world.reward, world.encode_state(), world.done

            if eval_config.render and vis_available:
                try:
                    fname = f"tsac_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config, filename=fname, collect_for_gif=True)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                    elif frame_file: print(f"Warn: Frame path returned but not found: {frame_file} step {step+1}")
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")

            state = next_state
            episode_reward += reward
            if done:
                if world.error_dist <= world_config.success_threshold: success_count += 1
                break
        # End step loop

        success = world.error_dist <= world_config.success_threshold
        status = "Success!" if success else "Failure."
        if done and not success:
            true_dist = world._calculate_range_measurement(world.agent.location, world.true_landmark.location)
            termination_reason = "Collision" if true_dist < world.world_config.collision_threshold else "Unknown"
            print(f"  Episode {episode+1}: Terminated early ({termination_reason}) Step {step+1}. Final Err: {world.error_dist:.2f}. {status}")
        elif not done: # Max steps reached
             if success: success_count +=1
             print(f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps}). Final Err: {world.error_dist:.2f}. {status}")
        else: # Done and Success
            print(f"  Episode {episode+1}: Success! Step {step+1} (Err: {world.error_dist:.2f} <= Thresh {world_config.success_threshold})")

        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        if eval_config.render and vis_available and episode_frames:
            gif_filename = f"tsac_eval_episode_{episode+1}.gif"
            try:
                print(f"  Saving GIF for episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"  Warn: Failed GIF save ep {episode+1}. E: {e}")
            if vis_config.delete_frames_after_gif:
                 cleaned_count = 0
                 for frame in episode_frames:
                     if os.path.exists(frame):
                         try: os.remove(frame); cleaned_count += 1
                         except OSError as ose: print(f"    Warn: Could not delete TSAC frame file {frame}: {ose}")
    # End episode loop

    agent.actor.train(); agent.critic1.train(); agent.critic2.train() # Revert to train mode
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