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
# from world_objects import Velocity # No longer needed here
from configs import DefaultConfig, ReplayBufferConfig, SACConfig, TrainingConfig
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions, potentially sequences."""

    def __init__(self, config: ReplayBufferConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config # Store config for sequence length default

    def push(self, state, action, reward, next_state, done):
        # state and next_state are dicts, action is float
        self.buffer.append((state['basic_state'], action, reward, next_state['basic_state'], done))

    def sample(self, batch_size: int, sequence_length: int = 1) -> Tuple:
        """Sample a batch of transitions or sequences from the buffer."""
        buffer_len = len(self.buffer)

        if sequence_length <= 1:
            # Sample individual transitions
            if buffer_len < batch_size:
                 raise ValueError(f"Not enough single transitions ({buffer_len}) in buffer to sample batch size ({batch_size}).")
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*batch)
            # Action shape (batch_size,) -> (batch_size, 1)
            return (np.array(state), np.array(action).reshape(-1, 1),
                    np.array(reward, dtype=np.float32),
                    np.array(next_state), np.array(done, dtype=np.float32))
        else:
            # Sample sequences using rejection sampling for efficiency
            if buffer_len < sequence_length:
                raise ValueError(f"Not enough transitions ({buffer_len}) in buffer for sequence length ({sequence_length}).")

            sampled_indices = []
            max_start_idx = buffer_len - sequence_length
            max_attempts = batch_size * 100
            attempts = 0

            while len(sampled_indices) < batch_size and attempts < max_attempts:
                attempts += 1
                idx = random.randint(0, max_start_idx)
                is_valid = not any(self.buffer[idx + k][4] for k in range(sequence_length - 1))

                if is_valid:
                    sampled_indices.append(idx)

            if len(sampled_indices) < batch_size:
                 raise ValueError(f"Could not sample {batch_size} valid sequences after {attempts} attempts. "
                                  f"Found {len(sampled_indices)}. Buffer size: {buffer_len}, Sequence length: {sequence_length}. ")

            states, actions, rewards, next_states, dones = [], [], [], [], []
            for idx in sampled_indices:
                seq_s, seq_a, seq_r, seq_ns, seq_d = [], [], [], [], []
                for i in range(sequence_length):
                    s, a, r, ns, d = self.buffer[idx + i]
                    seq_s.append(s) # s is basic_state tuple
                    seq_a.append(a) # a is float
                    seq_r.append(r)
                    seq_ns.append(ns) # ns is basic_state tuple
                    seq_d.append(d)
                states.append(seq_s)
                actions.append(seq_a)
                rewards.append(seq_r)
                next_states.append(seq_ns)
                dones.append(seq_d)

            # Shapes: state/next_state (batch, seq, state_dim), action (batch, seq, 1), reward/done (batch, seq)
            return (np.array(states), np.array(actions).reshape(batch_size, sequence_length, 1),
                    np.array(rewards, dtype=np.float32),
                    np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Policy network (Actor) for SAC, optionally with RNN."""

    def __init__(self, config: SACConfig):
        super(Actor, self).__init__()
        self.use_rnn = config.use_rnn
        self.rnn_hidden_size = config.rnn_hidden_size
        self.rnn_num_layers = config.rnn_num_layers
        self.rnn_type = config.rnn_type
        self.action_dim = config.action_dim # Should be 1

        input_dim = config.state_dim
        if self.use_rnn:
            if self.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size=config.state_dim, hidden_size=config.rnn_hidden_size,
                                   num_layers=config.rnn_num_layers, batch_first=True)
            elif self.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size=config.state_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            else:
                raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
            input_dim = config.rnn_hidden_size

        self.layers = nn.ModuleList()
        mlp_input_dim = input_dim
        for hidden_dim in config.hidden_dims:
            self.layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            mlp_input_dim = hidden_dim

        self.mean = nn.Linear(config.hidden_dims[-1], self.action_dim)
        self.log_std = nn.Linear(config.hidden_dims[-1], self.action_dim)

        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, state: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network, handling RNN if enabled."""
        if self.use_rnn:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            x, next_hidden_state = self.rnn(state, hidden_state)
            x = x[:, -1, :]
        else:
            x = state
            next_hidden_state = None

        for layer in self.layers:
            x = F.relu(layer(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, next_hidden_state

    def sample(self, state: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Sample action from the policy distribution, handling RNN."""
        mean, log_std, next_hidden_state = self.forward(state, hidden_state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        action_normalized = torch.tanh(x_t) # Output is [-1, 1]

        log_prob = normal.log_prob(x_t) - torch.log(1 - action_normalized.pow(2) + 1e-6)
        # Sum over action dimension (dim=1), keeping batch dim (dim=0)
        log_prob = log_prob.sum(1, keepdim=True)

        # Return normalized action [-1, 1]
        return action_normalized, log_prob, torch.tanh(mean), next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return initial hidden state for RNN."""
        if not self.use_rnn:
            return None
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device),
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device))
        elif self.rnn_type == 'gru':
            return torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)


class Critic(nn.Module):
    """Q-function network (Critic) for SAC, optionally with RNN."""

    def __init__(self, config: SACConfig):
        super(Critic, self).__init__()
        self.use_rnn = config.use_rnn
        self.rnn_hidden_size = config.rnn_hidden_size
        self.rnn_num_layers = config.rnn_num_layers
        self.rnn_type = config.rnn_type
        self.action_dim = config.action_dim # Should be 1

        rnn_input_dim = config.state_dim
        mlp_input_dim = config.state_dim + self.action_dim # Correct input dim

        if self.use_rnn:
            rnn_cell = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
            self.rnn1 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            self.rnn2 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            mlp_input_dim = config.rnn_hidden_size + self.action_dim
        else:
             self.rnn1, self.rnn2 = None, None

        # Q1 architecture
        self.q1_layers = nn.ModuleList()
        q1_mlp_input = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.q1_layers.append(nn.Linear(q1_mlp_input, hidden_dim))
            q1_mlp_input = hidden_dim
        self.q1_out = nn.Linear(config.hidden_dims[-1], 1)

        # Q2 architecture
        self.q2_layers = nn.ModuleList()
        q2_mlp_input = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.q2_layers.append(nn.Linear(q2_mlp_input, hidden_dim))
            q2_mlp_input = hidden_dim
        self.q2_out = nn.Linear(config.hidden_dims[-1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass returning both Q-values, handling RNN if enabled."""
        if self.use_rnn:
            if state.ndim == 2: state = state.unsqueeze(1)
            # Action shape is (batch, seq_len, 1)
            if action.ndim == 2: action = action.unsqueeze(1) # Ensure seq dim if single step

            h1_in, h2_in = None, None
            if isinstance(hidden_state, (list, tuple)) and len(hidden_state) == 2:
                h1_in, h2_in = hidden_state
            elif hidden_state is not None:
                print(f"Warning: Unexpected hidden_state format in Critic forward: {type(hidden_state)}. Expected tuple of two states.")
                h1_in, h2_in = None, None

            rnn_out1, next_h1 = self.rnn1(state, h1_in)
            rnn_out2, next_h2 = self.rnn2(state, h2_in)
            next_hidden_state = (next_h1, next_h2)

            rnn_out1 = rnn_out1[:, -1, :]
            rnn_out2 = rnn_out2[:, -1, :]
            # Action corresponds to the last time step
            action_last_step = action[:, -1, :] # Shape (batch, 1)

            x1 = torch.cat([rnn_out1, action_last_step], 1)
            x2 = torch.cat([rnn_out2, action_last_step], 1)
        else:
            # State (batch, state_dim), action (batch, 1)
            sa = torch.cat([state, action], 1)
            x1 = sa
            x2 = sa
            next_hidden_state = None

        for layer in self.q1_layers:
            x1 = F.relu(layer(x1))
        q1 = self.q1_out(x1)

        for layer in self.q2_layers:
            x2 = F.relu(layer(x2))
        q2 = self.q2_out(x2)

        return q1, q2, next_hidden_state

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward pass returning only Q1 value, handling RNN if enabled."""
        if self.use_rnn:
            if state.ndim == 2: state = state.unsqueeze(1)
            if action.ndim == 2: action = action.unsqueeze(1)

            h1_in = None
            if isinstance(hidden_state, (list, tuple)) and len(hidden_state) > 0:
                 h1_in = hidden_state[0]
            elif hidden_state is not None:
                 print(f"Warning: Unexpected hidden_state format in Critic q1_forward: {type(hidden_state)}. Assuming it's for Q1.")
                 h1_in = hidden_state

            rnn_out1, next_h1 = self.rnn1(state, h1_in)
            next_hidden_state = next_h1
            rnn_out1 = rnn_out1[:, -1, :]
            action_last_step = action[:, -1, :]
            x1 = torch.cat([rnn_out1, action_last_step], 1)
        else:
            sa = torch.cat([state, action], 1)
            x1 = sa
            next_hidden_state = None

        for layer in self.q1_layers:
            x1 = F.relu(layer(x1))
        q1 = self.q1_out(x1)
        return q1, next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
         """Return initial hidden state tuple (h1, h2) for RNNs."""
         if not self.use_rnn:
             return None
         if self.rnn_type == 'lstm':
             h1 = (torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device),
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device))
             h2 = (torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device),
                   torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device))
             return (h1, h2)
         elif self.rnn_type == 'gru':
             h1 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
             h2 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
             return (h1, h2)


class SAC:
    """Soft Actor-Critic algorithm implementation, optionally with RNN."""

    def __init__(self, config: SACConfig, device: torch.device = None):
        self.config = config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        # self.action_scale = config.action_scale # No longer used for scaling output here
        self.auto_tune_alpha = config.auto_tune_alpha
        self.use_rnn = config.use_rnn
        self.sequence_length = config.sequence_length if self.use_rnn else 1
        self.action_dim = config.action_dim # Should be 1

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"SAC Agent using device: {self.device}")
        if self.use_rnn:
            print(f"SAC Agent using RNN: Type={config.rnn_type}, Hidden={config.rnn_hidden_size}, Layers={config.rnn_num_layers}, SeqLen={self.sequence_length}")

        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config).to(self.device)
        self.critic_target = Critic(config).to(self.device)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.tensor([np.log(self.alpha)], requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)

    def select_action(self, state: dict, actor_hidden_state: Optional[Tuple] = None, evaluate: bool = False) -> Tuple[float, Optional[Tuple]]:
        """Select action (normalized yaw change [-1, 1]) based on state."""
        state_tuple = state['basic_state']
        state_tensor = torch.FloatTensor(state_tuple).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                _, _, action_mean_squashed, next_actor_hidden_state = self.actor.sample(state_tensor, actor_hidden_state)
                action_normalized = action_mean_squashed
            else:
                action_normalized, _, _, next_actor_hidden_state = self.actor.sample(state_tensor, actor_hidden_state)

        # Return the normalized action [-1, 1] as a float
        action_normalized_float = action_normalized.detach().cpu().numpy()[0, 0]
        return action_normalized_float, next_actor_hidden_state

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform a single SAC update step using a batch from memory."""
        min_buffer_size = batch_size if not self.use_rnn else self.sequence_length
        if len(memory) < min_buffer_size:
            return None

        try:
            state_batch, action_batch_normalized, reward_batch, next_state_batch, done_batch = memory.sample(
                batch_size, self.sequence_length)
        except ValueError as e:
            print(f"Skipping update due to sampling error: {e}")
            return None

        # Move batch to device
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        # Action is already normalized [-1, 1], shape (batch, [seq,] 1)
        action_batch_normalized = torch.FloatTensor(action_batch_normalized).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        initial_actor_hidden = self.actor.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_hidden = self.critic.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_target_hidden = self.critic_target.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None

        if self.use_rnn:
            # Reward/done for target calculation (last step): shape (batch, 1)
            reward_target = reward_batch[:, -1].unsqueeze(-1)
            done_target = done_batch[:, -1].unsqueeze(-1)
            # Action batch shape (batch, seq_len, 1)
        else:
            # Reward/done needs unsqueezing: shape (batch, 1)
            reward_target = reward_batch.unsqueeze(1)
            done_target = done_batch.unsqueeze(1)
            # Action batch shape (batch, 1)


        # --- Critic Update ---
        with torch.no_grad():
            if self.use_rnn:
                 next_actions_normalized_seq = []
                 next_log_probs_seq = []
                 hidden_state_actor_for_target = initial_actor_hidden
                 next_state_seq = next_state_batch # Shape (batch, seq_len, state_dim)
                 for t in range(self.sequence_length):
                     next_action_t, next_log_prob_t, _, hidden_state_actor_for_target = self.actor.sample(
                         next_state_seq[:, t, :], hidden_state_actor_for_target
                     )
                     next_actions_normalized_seq.append(next_action_t)
                     next_log_probs_seq.append(next_log_prob_t)

                 next_actions_normalized_seq_tensor = torch.stack(next_actions_normalized_seq, dim=1) # (batch, seq_len, 1)
                 next_log_prob_last = next_log_probs_seq[-1] # (batch, 1)

                 target_q1_last, target_q2_last, _ = self.critic_target(
                     next_state_batch, # (batch, seq_len, state_dim)
                     next_actions_normalized_seq_tensor, # (batch, seq_len, 1)
                     initial_critic_target_hidden
                 ) # Output shapes (batch, 1)
                 target_q_min = torch.min(target_q1_last, target_q2_last)

            else:
                 next_action_normalized, next_log_prob, _, _ = self.actor.sample(next_state_batch) # action shape (batch, 1)
                 target_q1, target_q2, _ = self.critic_target(next_state_batch, next_action_normalized) # Outputs (batch, 1)
                 target_q_min = torch.min(target_q1, target_q2)
                 next_log_prob_last = next_log_prob # Shape (batch, 1)

            current_alpha = self.alpha if not self.auto_tune_alpha else self.log_alpha.exp().item()
            target_q_entropy = target_q_min - current_alpha * next_log_prob_last
            y = reward_target + (1 - done_target) * self.gamma * target_q_entropy


        # --- Calculate Current Q values ---
        if self.use_rnn:
             current_q1_last, current_q2_last, _ = self.critic(
                 state_batch, # (batch, seq_len, state_dim)
                 action_batch_normalized, # (batch, seq_len, 1)
                 initial_critic_hidden
             ) # Outputs (batch, 1)
        else:
            current_q1, current_q2, _ = self.critic(state_batch, action_batch_normalized) # Outputs (batch, 1)
            current_q1_last = current_q1
            current_q2_last = current_q2

        critic_loss = F.mse_loss(current_q1_last, y) + F.mse_loss(current_q2_last, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # --- Actor Update ---
        for param in self.critic.parameters():
            param.requires_grad = False

        if self.use_rnn:
             action_pi_normalized_seq = []
             log_prob_pi_seq = []
             hidden_state_actor_for_update = initial_actor_hidden
             state_seq = state_batch # Shape (batch, seq_len, state_dim)
             for t in range(self.sequence_length):
                 action_t, log_prob_t, _, hidden_state_actor_for_update = self.actor.sample(
                      state_seq[:, t, :], hidden_state_actor_for_update
                 )
                 action_pi_normalized_seq.append(action_t)
                 log_prob_pi_seq.append(log_prob_t)

             action_pi_normalized_seq_tensor = torch.stack(action_pi_normalized_seq, dim=1) # (batch, seq_len, 1)
             log_prob_pi_last = log_prob_pi_seq[-1] # Shape (batch, 1)

             q1_pi_last, q2_pi_last, _ = self.critic(
                 state_batch.detach(),
                 action_pi_normalized_seq_tensor, # (batch, seq_len, 1)
                 initial_critic_hidden
             ) # Outputs (batch, 1)
             q_pi_min = torch.min(q1_pi_last, q2_pi_last)

        else:
            action_pi_normalized, log_prob_pi, _, _ = self.actor.sample(state_batch) # action shape (batch, 1)
            q1_pi, q2_pi, _ = self.critic(state_batch.detach(), action_pi_normalized) # Outputs (batch, 1)
            q_pi_min = torch.min(q1_pi, q2_pi)
            log_prob_pi_last = log_prob_pi # Shape (batch, 1)

        current_alpha = self.alpha if not self.auto_tune_alpha else self.log_alpha.exp().item()
        actor_loss = (current_alpha * log_prob_pi_last - q_pi_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad = True

        # --- Alpha Update ---
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_pi_last.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Target Network Update (Soft Update) ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }

    def save_model(self, path: str):
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
            save_dict['log_alpha_tensor'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
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

        if self.auto_tune_alpha and 'log_alpha_tensor' in checkpoint:
            loaded_log_alpha = checkpoint['log_alpha_tensor']
            self.log_alpha = loaded_log_alpha.to(self.device)
            if not self.log_alpha.requires_grad:
                 self.log_alpha.requires_grad_(True)

            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            try:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except ValueError as e:
                 print(f"Warning: Could not load alpha optimizer state: {e}. Reinitializing.")
                 self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            self.alpha = self.log_alpha.exp().item()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor.train()
        self.critic.train()
        self.critic_target.train()

        print(f"Model loaded successfully from {path}")


def train_sac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    sac_config = config.sac
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

    log_dir = os.path.join("runs", f"sac_training_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Device Setup
    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Warning: Multi-GPU not optimized for SAC updates. Using single specified/default GPU: {cuda_device}")
            device = torch.device(cuda_device)
        else:
            device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try:
                torch.cuda.set_device(device)
                print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e:
                print(f"Warning: Could not set CUDA device {cuda_device}. Error: {e}")
                device = torch.device("cuda:0") # Fallback
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU for training")


    agent = SAC(config=sac_config, device=device)
    memory = ReplayBuffer(config=buffer_config)

    os.makedirs(train_config.models_dir, exist_ok=True)
    # Load latest model if exists (logic remains same)
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("sac_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try:
            latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e:
            print(f"Could not find latest model: {e}")

    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming training from checkpoint: {latest_model_path}")
        agent.load_model(latest_model_path)
        try:
            total_steps = int(latest_model_path.split('_step')[1].split('.pt')[0])
            start_episode = int(latest_model_path.split('_ep')[1].split('_')[0]) + 1
            print(f"Resuming total_steps at: {total_steps}, episode: {start_episode}")
        except Exception:
             print("Warning: Could not infer steps/episode from model filename.")
             total_steps = 0
             start_episode = 1
    else:
        print("\nStarting training from scratch.")
        total_steps = 0
        start_episode = 1


    episode_rewards = []
    all_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': []}
    timing_metrics = {
        'env_step_time': deque(maxlen=100),
        'parameter_update_time': deque(maxlen=100)
    }

    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training SAC", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    for episode in pbar:
        env = World(world_config=world_config)
        state = env.encode_state() # state is dict
        episode_reward = 0
        episode_steps = 0

        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=device) if agent.use_rnn else None

        episode_step_times = []
        episode_param_update_times = []
        episode_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': []}
        updates_made_this_episode = 0

        for step_in_episode in range(train_config.max_steps):
            # Action is now a float (normalized yaw change)
            action_normalized, next_actor_hidden_state = agent.select_action(state, actor_hidden_state=actor_hidden_state, evaluate=False)

            step_start_time = time.time()
            # Pass the float action directly to env.step
            env.step(action_normalized, training=True, terminal_step=step_in_episode==train_config.max_steps-1)
            step_time = time.time() - step_start_time
            episode_step_times.append(step_time)
            timing_metrics['env_step_time'].append(step_time)

            reward = env.reward
            next_state = env.encode_state() # next_state is dict
            done = env.done

            # Push state dict, float action, reward, next_state dict, done
            memory.push(state, action_normalized, reward, next_state, done)

            state = next_state
            if agent.use_rnn:
                actor_hidden_state = next_actor_hidden_state

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Perform Updates
            if total_steps >= learning_starts and total_steps % train_freq == 0:
                for _ in range(gradient_steps):
                    min_buffer_size = batch_size if not agent.use_rnn else agent.sequence_length
                    # Check if enough data, considering batch_size and sequence_length
                    can_sample_batch = (not agent.use_rnn and len(memory) >= batch_size) or \
                                     (agent.use_rnn and len(memory) >= batch_size * agent.sequence_length)

                    if len(memory) >= min_buffer_size and can_sample_batch :
                        update_start_time = time.time()
                        losses = agent.update_parameters(memory, batch_size)
                        update_time = time.time() - update_start_time
                        if losses: # update might return None if sampling fails
                            episode_param_update_times.append(update_time)
                            timing_metrics['parameter_update_time'].append(update_time)
                            episode_losses['critic_loss'].append(losses['critic_loss'])
                            episode_losses['actor_loss'].append(losses['actor_loss'])
                            episode_losses['alpha'].append(losses['alpha'])
                            updates_made_this_episode += 1
                        else:
                             break # Stop gradient steps if sampling failed

            if done:
                break

        # --- Logging and Reporting (End of Episode) ---
        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else 0 for k, v in episode_losses.items()}
        if updates_made_this_episode > 0 :
             all_losses['critic_loss'].append(avg_losses['critic_loss'])
             all_losses['actor_loss'].append(avg_losses['actor_loss'])
             all_losses['alpha'].append(avg_losses['alpha'])

        if episode % log_frequency_ep == 0:
            if timing_metrics['env_step_time']:
                 avg_step_time = np.mean(timing_metrics['env_step_time'])
                 writer.add_scalar('Time/Environment_Step_ms_Avg100', avg_step_time * 1000, total_steps)
            if timing_metrics['parameter_update_time']:
                 avg_param_update_time = np.mean(timing_metrics['parameter_update_time'])
                 writer.add_scalar('Time/Parameter_Update_ms_Avg100', avg_param_update_time * 1000, total_steps)
            elif total_steps >= learning_starts:
                 writer.add_scalar('Time/Parameter_Update_ms_Avg100', 0, total_steps)

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Progress/Buffer_Size', len(memory), total_steps)
            writer.add_scalar('Error/Distance_EndEpisode', env.error_dist, total_steps)

            if hasattr(env, 'pf_update_time') and isinstance(env.pf_update_time, (int, float)):
                 writer.add_scalar('Time/Estimator_Update_ms_LastStep', env.pf_update_time * 1000, total_steps)

            if updates_made_this_episode > 0:
                writer.add_scalar('Loss/Critic_AvgEp', avg_losses['critic_loss'], total_steps)
                writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                writer.add_scalar('Alpha/Value_AvgEp', avg_losses['alpha'], total_steps)
            else:
                 writer.add_scalar('Loss/Critic_AvgEp', 0, total_steps)
                 writer.add_scalar('Loss/Actor_AvgEp', 0, total_steps)
                 writer.add_scalar('Alpha/Value_AvgEp', agent.alpha, total_steps)

            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar_postfix = {'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps}
            if updates_made_this_episode > 0:
                pbar_postfix['crit_loss'] = f"{avg_losses['critic_loss']:.3f}"
                pbar_postfix['alpha'] = f"{avg_losses['alpha']:.3f}"
            elif len(all_losses['critic_loss']) > 0:
                 pbar_postfix['crit_loss'] = f"{all_losses['critic_loss'][-1]:.3f}"
                 pbar_postfix['alpha'] = f"{all_losses['alpha'][-1]:.3f}"
            pbar.set_postfix(pbar_postfix)

        if episode % save_interval_ep == 0:
            save_path = os.path.join(
                train_config.models_dir, f"sac_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

    pbar.close()
    writer.close()
    print(f"Training finished. Total steps: {total_steps}")
    final_save_path = os.path.join(
        train_config.models_dir, f"sac_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_sac(agent=agent, config=config)

    return agent, episode_rewards


def evaluate_sac(agent: SAC, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    # --- Conditional import for visualization ---
    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            import imageio.v2 as imageio # Needed by save_gif
            vis_available = True
            print("Visualization enabled.")
        except ImportError:
            print("Visualization libraries not found (matplotlib, imageio). Rendering disabled.")
            vis_available = False
            eval_config.render = False # Force disable rendering
    else:
        vis_available = False
        print("Rendering disabled by config.")

    eval_rewards = []
    success_count = 0
    all_episode_gif_paths = []

    agent.actor.eval()
    agent.critic.eval()

    print(f"\nRunning Evaluation for {eval_config.num_episodes} episodes...")
    for episode in range(eval_config.num_episodes):
        env = World(world_config=world_config)
        state = env.encode_state() # state is dict
        episode_reward = 0
        episode_frames = []

        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=agent.device) if agent.use_rnn else None

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            reset_trajectories()
            try:
                initial_frame_file = visualize_world(
                    world=env,
                    vis_config=vis_config,
                    filename=f"eval_ep{episode+1}_frame_000_initial.png",
                    collect_for_gif=True
                )
                if initial_frame_file and os.path.exists(initial_frame_file):
                    episode_frames.append(initial_frame_file)
                elif initial_frame_file:
                     print(f"Warning: Initial frame file path returned but not found: {initial_frame_file}")
            except Exception as e:
                print(f"Warning: Visualization failed for initial state. Error: {e}")

        for step in range(eval_config.max_steps):
            # Action is float (normalized yaw change)
            action_normalized, next_actor_hidden_state = agent.select_action(state, actor_hidden_state=actor_hidden_state, evaluate=True)

            # Step environment with float action
            env.step(action_normalized, training=False, terminal_step=step==eval_config.max_steps-1)
            reward = env.reward
            next_state = env.encode_state() # next_state is dict
            done = env.done

            if eval_config.render and vis_available:
                try:
                    frame_file = visualize_world(
                        world=env,
                        vis_config=vis_config,
                        filename=f"eval_ep{episode+1}_frame_{step+1:03d}.png",
                        collect_for_gif=True
                    )
                    if frame_file and os.path.exists(frame_file):
                        episode_frames.append(frame_file)
                    elif frame_file:
                         print(f"Warning: Frame file path returned but not found: {frame_file} for step {step+1}")
                except Exception as e:
                     print(f"Warning: Visualization failed for step {step+1}. Error: {e}")

            state = next_state
            if agent.use_rnn:
                actor_hidden_state = next_actor_hidden_state

            episode_reward += reward

            if done:
                if env.error_dist <= world_config.success_threshold:
                    success_count += 1
                    print(f"  Episode {episode+1}: Success! Found landmark at step {step+1} (Error: {env.error_dist:.4f} <= threshold {world_config.success_threshold})")
                else:
                    print(f"  Episode {episode+1}: Terminated early at step {step+1} (Not success). Final Error: {env.error_dist:.4f}")
                break

        if not done:
             success = env.error_dist <= world_config.success_threshold
             status = "Success!" if success else "Failure."
             if success: success_count +=1
             print(f"  Episode {episode+1}: Finished (Max steps {eval_config.max_steps} reached). Final Error: {env.error_dist:.4f}. {status}")

        eval_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}")

        if eval_config.render and vis_available and episode_frames:
            gif_filename = f"eval_episode_{episode+1}.gif"
            try:
                print(f"  Attempting to save GIF for episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(
                    output_filename=gif_filename,
                    vis_config=vis_config,
                    frame_paths=episode_frames,
                    delete_frames=vis_config.delete_frames_after_gif
                )
                if gif_path:
                    print(f"  GIF saved: {gif_path}")
                    all_episode_gif_paths.append(gif_path)
                else:
                     print(f"  Warning: GIF saving function returned None for episode {episode+1}.")
            except Exception as e:
                print(f"  Warning: Failed to create or save GIF for episode {episode+1}. Error: {e}")
            if vis_config.delete_frames_after_gif: # Clean up frames regardless of GIF success if requested
                cleaned_count = 0
                for frame in episode_frames:
                    if os.path.exists(frame):
                        try: os.remove(frame); cleaned_count += 1
                        except OSError as ose: print(f"    Warning: Could not delete frame file {frame}: {ose}")
                # print(f"    Cleaned up {cleaned_count}/{len(episode_frames)} frame files.")

    agent.actor.train()
    agent.critic.train()

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0

    print("\n--- Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Reward: {avg_eval_reward:.2f} +/- {std_eval_reward:.2f}")
    print(f"Success Rate (Error <= {world_config.success_threshold}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_episode_gif_paths:
        print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available:
        print("Rendering was enabled in config, but visualization libraries were not found.")
    elif not eval_config.render:
        print("Rendering was disabled.")
    print("--- End Evaluation ---\n")

    return eval_rewards, success_rate