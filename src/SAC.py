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
from visualization import visualize_world, reset_trajectories, save_gif
from configs import DefaultConfig, ReplayBufferConfig, SACConfig, TrainingConfig
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions, potentially sequences."""

    def __init__(self, config: ReplayBufferConfig):
        self.buffer = deque(maxlen=config.capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, sequence_length: int = 1) -> Tuple:
        """Sample a batch of transitions or sequences from the buffer."""
        if sequence_length <= 1:
            # Sample individual transitions
            if len(self.buffer) < batch_size:
                 raise ValueError("Not enough transitions in buffer to sample.")
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*batch)
            return (np.array(state), np.array(action),
                    np.array(reward, dtype=np.float32),
                    np.array(next_state), np.array(done, dtype=np.float32))
        else:
            # Sample sequences
            states, actions, rewards, next_states, dones = [], [], [], [], []
            # Ensure we have enough space for sequences and sample valid start indices
            # A sequence is valid if it fits in the buffer and does not contain 'done=True' except possibly at the very end
            valid_indices = [i for i in range(len(self.buffer) - sequence_length + 1)
                             if not any(self.buffer[i+k][4] for k in range(sequence_length - 1))]

            if len(valid_indices) < batch_size:
                 # Fallback or raise error, here we sample with replacement from valid indices if needed
                 if not valid_indices:
                     raise ValueError("Not enough valid sequences in buffer to sample.")
                 sampled_indices = random.choices(valid_indices, k=batch_size)
            else:
                 sampled_indices = random.sample(valid_indices, batch_size)


            for idx in sampled_indices:
                seq_s, seq_a, seq_r, seq_ns, seq_d = [], [], [], [], []
                for i in range(sequence_length):
                    s, a, r, ns, d = self.buffer[idx + i]
                    seq_s.append(s)
                    seq_a.append(a)
                    seq_r.append(r)
                    seq_ns.append(ns)
                    seq_d.append(d)
                states.append(seq_s)
                actions.append(seq_a)
                rewards.append(seq_r)
                next_states.append(seq_ns)
                dones.append(seq_d)

            # Shape: (batch_size, sequence_length, feature_dim)
            return (np.array(states), np.array(actions),
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

        self.mean = nn.Linear(config.hidden_dims[-1], config.action_dim)
        self.log_std = nn.Linear(config.hidden_dims[-1], config.action_dim)

        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, state: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network, handling RNN if enabled."""
        if self.use_rnn:
            # Expect state shape (batch, seq_len, features)
            if state.ndim == 2:
                state = state.unsqueeze(1) # Add sequence dimension if missing for single step inference
            x, next_hidden_state = self.rnn(state, hidden_state)
            # Use the output of the last time step if sequence, or the only time step if single step
            x = x[:, -1, :]
        else:
            # Expect state shape (batch, features)
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
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean), next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return initial hidden state for RNN."""
        if not self.use_rnn:
            return None
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device),
                    torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device))
        elif self.rnn_type == 'gru':
            # GRU only has one hidden state tensor
            return torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)


class Critic(nn.Module):
    """Q-function network (Critic) for SAC, optionally with RNN."""

    def __init__(self, config: SACConfig):
        super(Critic, self).__init__()
        self.use_rnn = config.use_rnn
        self.rnn_hidden_size = config.rnn_hidden_size
        self.rnn_num_layers = config.rnn_num_layers
        self.rnn_type = config.rnn_type

        rnn_input_dim = config.state_dim # RNN takes only state
        mlp_input_dim = config.state_dim + config.action_dim # Default MLP input

        if self.use_rnn:
            rnn_cell = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
            self.rnn1 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            self.rnn2 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            mlp_input_dim = config.rnn_hidden_size + config.action_dim # MLP uses RNN output + action
        else:
             self.rnn1, self.rnn2 = None, None # Explicitly set to None if not used

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
             # Expect state shape (batch, seq_len, features), action (batch, seq_len, action_dim)
            if state.ndim == 2: state = state.unsqueeze(1)
            if action.ndim == 2: action = action.unsqueeze(1)

            h1_in, h2_in = None, None
            if hidden_state is not None:
                h1_in, h2_in = hidden_state # Unpack tuple of hidden states for Q1/Q2 RNNs

            rnn_out1, next_h1 = self.rnn1(state, h1_in)
            rnn_out2, next_h2 = self.rnn2(state, h2_in)
            next_hidden_state = (next_h1, next_h2)

            # Use the output of the last time step if sequence
            rnn_out1 = rnn_out1[:, -1, :]
            rnn_out2 = rnn_out2[:, -1, :]
            # Action corresponds to the last time step
            action_last_step = action[:, -1, :]

            x1 = torch.cat([rnn_out1, action_last_step], 1)
            x2 = torch.cat([rnn_out2, action_last_step], 1)
        else:
            # Expect state (batch, features), action (batch, action_dim)
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
            if hidden_state is not None: h1_in = hidden_state # Q1 uses first element if tuple passed

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
        self.action_scale = config.action_scale
        self.auto_tune_alpha = config.auto_tune_alpha
        self.use_rnn = config.use_rnn
        # sequence_length is 1 if RNN is not used
        self.sequence_length = config.sequence_length if self.use_rnn else 1

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
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

    def select_action(self, state: dict, actor_hidden_state: Optional[Tuple] = None, evaluate: bool = False) -> Tuple[np.ndarray, Optional[Tuple]]:
        """Select action based on state and current actor hidden state."""
        state_tuple = state['basic_state']
        state_tensor = torch.FloatTensor(state_tuple).to(self.device).unsqueeze(0) # Shape (1, state_dim)

        with torch.no_grad():
            if evaluate:
                _, _, action_mean_squashed, next_actor_hidden_state = self.actor.sample(state_tensor, actor_hidden_state)
                action_normalized = action_mean_squashed
            else:
                action_normalized, _, _, next_actor_hidden_state = self.actor.sample(state_tensor, actor_hidden_state)

        action_scaled = action_normalized.detach().cpu().numpy()[0] * self.action_scale
        return action_scaled, next_actor_hidden_state

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform a single SAC update step using a batch from memory."""
        # Ensure enough samples for the batch (considering sequence length)
        if len(memory) < batch_size * self.sequence_length:
            return None

        # Sample sequences or transitions based on use_rnn
        # sequence_length will be 1 if self.use_rnn is False
        state_batch, action_batch_scaled, reward_batch, next_state_batch, done_batch = memory.sample(
            batch_size, self.sequence_length)

        # Move batch to device
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch_normalized = torch.FloatTensor(action_batch_scaled / self.action_scale).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # --- Prepare inputs and initial hidden states ---
        initial_actor_hidden = None
        initial_critic_hidden = None
        initial_critic_target_hidden = None

        if self.use_rnn:
            # Data shape: (batch, seq_len, dim)
            # Target calculation needs last reward/done: (batch, 1)
            reward_target = reward_batch[:, -1].unsqueeze(-1)
            done_target = done_batch[:, -1].unsqueeze(-1)
            # Get initial hidden states
            initial_actor_hidden = self.actor.get_initial_hidden_state(batch_size, self.device)
            initial_critic_hidden = self.critic.get_initial_hidden_state(batch_size, self.device)
            initial_critic_target_hidden = self.critic_target.get_initial_hidden_state(batch_size, self.device)
        else:
            # Data shape: (batch, dim)
            # Target calculation needs unsqueezed reward/done: (batch, 1)
            reward_target = reward_batch.unsqueeze(1)
            done_target = done_batch.unsqueeze(1)
            # No hidden states needed

        # --- Critic Update ---
        with torch.no_grad():
            if self.use_rnn:
                 # --- RNN Target Calculation ---
                 next_actions_normalized_seq, next_log_probs_seq = [], []
                 current_actor_hidden = initial_actor_hidden
                 for t in range(self.sequence_length):
                     next_action_t, next_log_prob_t, _, current_actor_hidden = self.actor.sample(
                         next_state_batch[:, t, :], current_actor_hidden
                     )
                     next_actions_normalized_seq.append(next_action_t)
                     next_log_probs_seq.append(next_log_prob_t)
                 # Note: We only need the last step's results for Bellman target
                 next_action_last = next_actions_normalized_seq[-1]
                 next_log_prob_last = next_log_probs_seq[-1]

                 # Pass the whole sequence to target critic
                 # We need the hidden state from processing the sequence up to the second-to-last step
                 # to get the Q value corresponding to the *last* state transition.
                 target_q1_seq, target_q2_seq = [], []
                 current_critic_target_hidden = initial_critic_target_hidden
                 for t in range(self.sequence_length):
                      target_q1_t, target_q2_t, current_critic_target_hidden = self.critic_target(
                          next_state_batch[:, t, :],
                          torch.stack(next_actions_normalized_seq, dim=1)[:, t, :], # Pass corresponding action
                          current_critic_target_hidden
                      )
                      target_q1_seq.append(target_q1_t)
                      target_q2_seq.append(target_q2_t)

                 # Use Q-value from the *last* time step for Bellman target
                 target_q1_last = target_q1_seq[-1]
                 target_q2_last = target_q2_seq[-1]
                 target_q_min = torch.min(target_q1_last, target_q2_last)
            else:
                 # --- MLP Target Calculation ---
                 next_action_normalized, next_log_prob, _, _ = self.actor.sample(next_state_batch) # No hidden state
                 target_q1, target_q2, _ = self.critic_target(next_state_batch, next_action_normalized) # No hidden state
                 target_q_min = torch.min(target_q1, target_q2)
                 next_log_prob_last = next_log_prob # Only one step

            # Calculate Bellman target 'y'
            target_q_entropy = target_q_min - self.alpha * next_log_prob_last
            y = reward_target + (1 - done_target) * self.gamma * target_q_entropy

        # --- Calculate Current Q values ---
        if self.use_rnn:
             # --- RNN Current Q Calculation ---
             # Pass the whole sequence to the critic
             current_q1_seq, current_q2_seq = [], []
             current_critic_hidden_prop = initial_critic_hidden
             for t in range(self.sequence_length):
                 q1_t, q2_t, current_critic_hidden_prop = self.critic(
                     state_batch[:, t, :], action_batch_normalized[:, t, :], current_critic_hidden_prop
                 )
                 current_q1_seq.append(q1_t)
                 current_q2_seq.append(q2_t)

             # Use Q-value from the last time step for loss calculation
             current_q1_last = current_q1_seq[-1]
             current_q2_last = current_q2_seq[-1]
        else:
            # --- MLP Current Q Calculation ---
            current_q1, current_q2, _ = self.critic(state_batch, action_batch_normalized) # No hidden state
            current_q1_last = current_q1
            current_q2_last = current_q2

        # Calculate Critic Loss using Q values corresponding to the target 'y'
        critic_loss = F.mse_loss(current_q1_last, y) + F.mse_loss(current_q2_last, y)

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Freeze critic parameters
        for param in self.critic.parameters():
            param.requires_grad = False

        if self.use_rnn:
             # --- RNN Actor Update ---
             # Propagate actor over sequence to get actions and log_probs
             action_pi_normalized_seq, log_prob_pi_seq = [], []
             current_actor_hidden_update = initial_actor_hidden # Use fresh initial hidden state
             for t in range(self.sequence_length):
                 action_t, log_prob_t, _, current_actor_hidden_update = self.actor.sample(
                      state_batch[:, t, :], current_actor_hidden_update
                 )
                 action_pi_normalized_seq.append(action_t)
                 log_prob_pi_seq.append(log_prob_t)
             # Need last step's log_prob for loss and alpha update
             log_prob_pi_last = log_prob_pi_seq[-1]

             # Propagate critic over sequence with *new* actor actions
             q1_pi_seq, q2_pi_seq = [], []
             current_critic_hidden_actor_update = initial_critic_hidden # Use fresh initial hidden state
             for t in range(self.sequence_length):
                 q1_pi_t, q2_pi_t, current_critic_hidden_actor_update = self.critic(
                     state_batch[:, t, :].detach(), # Detach state batch
                     action_pi_normalized_seq[t], # Use action from actor pass
                     current_critic_hidden_actor_update
                 )
                 q1_pi_seq.append(q1_pi_t)
                 q2_pi_seq.append(q2_pi_t)
             # Use last step's Q value for actor loss
             q1_pi_last = q1_pi_seq[-1]
             q2_pi_last = q2_pi_seq[-1]
             q_pi_min = torch.min(q1_pi_last, q2_pi_last)

        else:
            # --- MLP Actor Update ---
            action_pi_normalized, log_prob_pi, _, _ = self.actor.sample(state_batch) # No hidden state
            q1_pi, q2_pi, _ = self.critic(state_batch, action_pi_normalized) # No hidden state
            q_pi_min = torch.min(q1_pi, q2_pi)
            log_prob_pi_last = log_prob_pi # Only one step

        # Calculate Actor loss
        actor_loss = (self.alpha * log_prob_pi_last - q_pi_min).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for param in self.critic.parameters():
            param.requires_grad = True

        # --- Alpha Update ---
        if self.auto_tune_alpha:
            # Use log_prob from the last time step (detached)
            alpha_loss = -(self.log_alpha * (log_prob_pi_last.detach() + self.target_entropy)).mean()
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
            # Ensure log_alpha is loaded correctly and requires grad
            self.log_alpha = checkpoint['log_alpha_state_dict'].to(self.device)
            if not self.log_alpha.requires_grad:
                 self.log_alpha.requires_grad_(True)
            # Re-initialize alpha optimizer with the potentially new log_alpha parameter
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            try: # Add try-except block for robustness if optimizer state is incompatible
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except ValueError as e:
                 print(f"Warning: Could not load alpha optimizer state, possibly due to parameter mismatch. Reinitializing alpha optimizer. Error: {e}")
                 self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr) # Reinitialize if loading fails

            self.alpha = self.log_alpha.exp().item()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        print(f"Model loaded successfully from {path}")


def train_sac(config: DefaultConfig, use_multi_gpu: bool = False):
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

    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training (Note: SAC updates are often sequential)")
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

    agent = SAC(config=sac_config, device=device)
    memory = ReplayBuffer(config=buffer_config)

    os.makedirs(train_config.models_dir, exist_ok=True)

    episode_rewards = []
    all_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': []}

    timing_metrics = {
        'env_step_time': [],
        'parameter_update_time': []
    }

    pbar = tqdm(range(1, train_config.num_episodes + 1),
                desc="Training", unit="episode")

    for episode in pbar:
        env = World(world_config=world_config)
        state = env.encode_state()
        episode_reward = 0
        episode_steps = 0

        # Initialize actor hidden state at the start of the episode if using RNN
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=device) if agent.use_rnn else None

        episode_step_times = []
        episode_param_update_times = []
        episode_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': []}

        for step_in_episode in range(train_config.max_steps):
            action_scaled, next_actor_hidden_state = agent.select_action(state, actor_hidden_state=actor_hidden_state, evaluate=False)
            action_obj = Velocity(x=action_scaled[0], y=action_scaled[1], z=0.0)

            step_start_time = time.time()
            env.step(action_obj, training=True, terminal_step=step_in_episode==train_config.max_steps-1)
            step_time = time.time() - step_start_time
            episode_step_times.append(step_time)

            reward = env.reward
            next_state = env.encode_state()
            done = env.done

            memory.push(state['basic_state'], action_scaled, reward, next_state['basic_state'], done)

            state = next_state
            # Update actor hidden state for the next step only if using RNN
            if agent.use_rnn:
                actor_hidden_state = next_actor_hidden_state
                # Reset hidden state if episode ended (done=True)
                # The reset actually happens at the start of the *next* episode's loop
                # If done is True here, the hidden state is passed one last time,
                # but won't be used to start the next step in *this* episode.
                # For training updates using sequences ending in 'done', the target calculation handles it.

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Perform Updates Here
            if total_steps >= learning_starts and total_steps % train_freq == 0:
                for _ in range(gradient_steps):
                     # Check buffer size condition based on sequence length
                    if len(memory) >= batch_size * agent.sequence_length:
                        update_start_time = time.time()
                        losses = agent.update_parameters(memory, batch_size)
                        update_time = time.time() - update_start_time
                        episode_param_update_times.append(update_time)

                        if losses:
                            episode_losses['critic_loss'].append(losses['critic_loss'])
                            episode_losses['actor_loss'].append(losses['actor_loss'])
                            episode_losses['alpha'].append(losses['alpha'])

            if done:
                break # End episode

        # --- Logging and Reporting (End of Episode) ---
        episode_rewards.append(episode_reward)

        avg_losses = {k: np.mean(v) if v else 0 for k, v in episode_losses.items()}
        updates_made_this_episode = any(v for v in episode_losses.values() if len(v) > 0) # Check if any lists are non-empty
        if updates_made_this_episode:
             all_losses['critic_loss'].append(avg_losses['critic_loss'])
             all_losses['actor_loss'].append(avg_losses['actor_loss'])
             all_losses['alpha'].append(avg_losses['alpha'])

        if episode % log_frequency_ep == 0:
            if episode_step_times:
                avg_step_time = np.mean(episode_step_times)
                timing_metrics['env_step_time'].append(avg_step_time)
                writer.add_scalar('Time/Environment_Step_ms', avg_step_time * 1000, total_steps)

            if episode_param_update_times:
                avg_param_update_time = np.mean(episode_param_update_times)
                timing_metrics['parameter_update_time'].append(avg_param_update_time)
                writer.add_scalar('Time/Parameter_Update_ms', avg_param_update_time * 1000, total_steps)
            elif total_steps >= learning_starts:
                 writer.add_scalar('Time/Parameter_Update_ms', 0, total_steps)

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Error/Distance_EndEpisode', env.error_dist, total_steps)
            # Ensure pf_update_time exists and is a number before logging
            if hasattr(env, 'pf_update_time') and isinstance(env.pf_update_time, (int, float)):
                 writer.add_scalar('Time/Estimator_Update_ms', env.pf_update_time * 1000, total_steps)

            if updates_made_this_episode:
                writer.add_scalar('Loss/Critic_AvgEp', avg_losses['critic_loss'], total_steps)
                writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                writer.add_scalar('Alpha/Value', avg_losses['alpha'], total_steps)

            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar_postfix = {'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps}
            if updates_made_this_episode:
                pbar_postfix['crit_loss'] = f"{avg_losses['critic_loss']:.3f}"
                pbar_postfix['act_loss'] = f"{avg_losses['actor_loss']:.3f}"
                pbar_postfix['alpha'] = f"{avg_losses['alpha']:.3f}"
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

    return agent, episode_rewards


def evaluate_sac(agent: SAC, config: DefaultConfig):
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

        # Initialize actor hidden state for evaluation if using RNN
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=agent.device) if agent.use_rnn else None

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
            action_scaled, next_actor_hidden_state = agent.select_action(state, actor_hidden_state=actor_hidden_state, evaluate=True)
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
            # Update actor hidden state for the next evaluation step
            if agent.use_rnn:
                actor_hidden_state = next_actor_hidden_state
                if done: # Reset hidden state if episode finished
                     actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=agent.device)

            episode_reward += reward

            if done:
                if env.error_dist <= world_config.success_threshold:
                    success_count += 1
                    print(
                        f"  Episode {episode+1}: Success! Found landmark at step {step+1} (Error: {env.error_dist:.2f} <= threshold {world_config.success_threshold})")
                else:
                    print(
                        f"  Episode {episode+1}: Terminated early at step {step+1} (Not success). Final Error: {env.error_dist:.2f}"
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
    success_rate = success_count / \
        eval_config.num_episodes if eval_config.num_episodes > 0 else 0
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
