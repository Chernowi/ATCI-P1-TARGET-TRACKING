import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import time
from collections import deque
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


# Local imports
from world import World
from configs import DefaultConfig, PPOConfig, TrainingConfig, WorldConfig, CORE_STATE_DIM
from torch.utils.tensorboard import SummaryWriter
import math
from utils import RunningMeanStd 
from typing import Tuple, Optional, List, Any, Dict


def detach_hidden_state(hidden_state: Optional[Tuple]) -> Optional[Tuple]:
    """Detaches a hidden state (or a tuple of hidden states for LSTM) from the computation graph."""
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):  # LSTM hidden state (h, c)
        return tuple(h.detach() for h in hidden_state)
    else:  # GRU hidden state (h)
        return hidden_state.detach()


class RecurrentPPOMemory:
    """Memory buffer for Recurrent PPO. Stores full rollouts of experiences."""
    def __init__(self, ppo_config: PPOConfig, world_config: WorldConfig):
        self.ppo_config = ppo_config
        self.world_config = world_config
        self.gamma = ppo_config.gamma
        self.gae_lambda = ppo_config.gae_lambda
        self.state_dim = ppo_config.state_dim # CORE_STATE_DIM

        # Buffers for the current rollout being collected
        self.current_rollout_states: List[Tuple] = [] # List of normalized basic_state tuples
        self.current_rollout_actions: List[float] = []
        self.current_rollout_log_probs: List[float] = []
        self.current_rollout_values: List[float] = []
        self.current_rollout_rewards_raw: List[float] = []
        self.current_rollout_dones: List[bool] = []
        self.current_rollout_actor_hiddens: List[Optional[Tuple]] = [] # Store actor hidden states AFTER state
        self.current_rollout_critic_hiddens: List[Optional[Tuple]] = [] # Store critic hidden states AFTER state

        # Main storage for processed rollouts ready for batching
        self.rollouts_states: List[torch.Tensor] = [] # Each item is a tensor of states for a rollout
        self.rollouts_actions: List[torch.Tensor] = []
        self.rollouts_old_log_probs: List[torch.Tensor] = []
        self.rollouts_advantages: List[torch.Tensor] = []
        self.rollouts_returns: List[torch.Tensor] = []
        self.rollouts_seq_lengths: List[int] = []
        # Storing initial hidden states for each rollout can be complex if BPTT spans updates.
        # For simplicity, we'll re-initialize to zeros for each batch in update_parameters.
        # If we were to store them:
        # self.rollouts_initial_actor_hiddens: List[Optional[Tuple]] = []
        # self.rollouts_initial_critic_hiddens: List[Optional[Tuple]] = []


    def store_step(self, norm_basic_state: Tuple, action: float, log_prob: float, value: float,
                   raw_reward: float, done: bool,
                   actor_hidden_state_after: Optional[Tuple], critic_hidden_state_after: Optional[Tuple]):
        """Store data for a single step of the current rollout."""
        self.current_rollout_states.append(norm_basic_state)
        self.current_rollout_actions.append(action)
        self.current_rollout_log_probs.append(log_prob)
        self.current_rollout_values.append(value)
        self.current_rollout_rewards_raw.append(raw_reward)
        self.current_rollout_dones.append(done)
        if self.ppo_config.use_rnn:
            self.current_rollout_actor_hiddens.append(detach_hidden_state(actor_hidden_state_after))
            self.current_rollout_critic_hiddens.append(detach_hidden_state(critic_hidden_state_after))

    def finalize_rollout(self, last_value: float, reward_normalizer: Optional[RunningMeanStd] = None):
        """
        Process the completed current rollout: calculate GAE and returns, then store it.
        `last_value` is V(S_T) or V(S_{t+rollout_len}) used for GAE if the rollout didn't end terminally.
        """
        if not self.current_rollout_states:
            return # Nothing to finalize

        n_steps = len(self.current_rollout_states)
        states_arr = np.array(self.current_rollout_states, dtype=np.float32)
        actions_arr = np.array(self.current_rollout_actions, dtype=np.float32)
        log_probs_arr = np.array(self.current_rollout_log_probs, dtype=np.float32)
        values_arr = np.array(self.current_rollout_values, dtype=np.float32)
        rewards_raw_arr = np.array(self.current_rollout_rewards_raw, dtype=np.float32)
        dones_arr = np.array(self.current_rollout_dones, dtype=np.float32)

        # Normalize rewards for GAE calculation if normalizer is provided
        if reward_normalizer:
            rewards_tensor_raw = torch.from_numpy(rewards_raw_arr).unsqueeze(1).to(reward_normalizer.mean.device)
            # Note: GAE benefits from stable reward stats. Updating normalizer here might be too frequent.
            # Usually, normalizer is updated with *all* rewards collected before an update cycle.
            # For simplicity, we'll assume it's been updated or use its current stats.
            rewards_for_gae = reward_normalizer.normalize(rewards_tensor_raw).squeeze(1).cpu().numpy()
        else:
            rewards_for_gae = rewards_raw_arr

        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)
        gae = 0.0
        
        # If the last step of current_rollout was not a terminal 'done', last_value is V(S_{t+1})
        # If it was a terminal 'done', last_value should be 0 (or V(S_terminal)=0).
        # The `last_value` passed in should already account for this.
        
        for t in reversed(range(n_steps)):
            delta = rewards_for_gae[t] + self.gamma * (last_value if t == n_steps - 1 else values_arr[t+1]) * (1.0 - dones_arr[t]) - values_arr[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones_arr[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values_arr[t] # GAE definition of return
        
        self.rollouts_states.append(torch.from_numpy(states_arr))
        self.rollouts_actions.append(torch.from_numpy(actions_arr).unsqueeze(1))
        self.rollouts_old_log_probs.append(torch.from_numpy(log_probs_arr).unsqueeze(1))
        self.rollouts_advantages.append(torch.from_numpy(advantages))
        self.rollouts_returns.append(torch.from_numpy(returns).unsqueeze(1))
        self.rollouts_seq_lengths.append(n_steps)
        
        # Clear current rollout buffers
        self.current_rollout_states.clear()
        self.current_rollout_actions.clear()
        self.current_rollout_log_probs.clear()
        self.current_rollout_values.clear()
        self.current_rollout_rewards_raw.clear()
        self.current_rollout_dones.clear()
        if self.ppo_config.use_rnn:
            self.current_rollout_actor_hiddens.clear()
            self.current_rollout_critic_hiddens.clear()

    def _pad_sequences(self, sequences: List[torch.Tensor], batch_first: bool = True, padding_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads a list of variable length sequences and returns their original lengths."""
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        padded_seqs = pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
        return padded_seqs, lengths

    def generate_batches(self) -> Optional[List[Dict[str, Any]]]:
        if not self.rollouts_states:
            return None

        num_rollouts = len(self.rollouts_states)
        indices = np.arange(num_rollouts)
        np.random.shuffle(indices)
        
        # Determine number of sequences per batch (ppo_config.batch_size refers to this)
        batch_size_rollouts = self.ppo_config.batch_size 

        all_batches_data = []

        for i in range(0, num_rollouts, batch_size_rollouts):
            batch_indices = indices[i : i + batch_size_rollouts]
            
            batch_states_list = [self.rollouts_states[j] for j in batch_indices]
            batch_actions_list = [self.rollouts_actions[j] for j in batch_indices]
            batch_log_probs_list = [self.rollouts_old_log_probs[j] for j in batch_indices]
            batch_advantages_list = [self.rollouts_advantages[j] for j in batch_indices]
            batch_returns_list = [self.rollouts_returns[j] for j in batch_indices]
            batch_seq_lengths_list = [self.rollouts_seq_lengths[j] for j in batch_indices]

            # Pad sequences in the batch
            states_b, seq_lengths_b = self._pad_sequences(batch_states_list)
            actions_b, _ = self._pad_sequences(batch_actions_list)
            old_log_probs_b, _ = self._pad_sequences(batch_log_probs_list)
            advantages_b_unnorm, _ = self._pad_sequences(batch_advantages_list, padding_value=0.0) # Pad advantages with 0
            returns_b, _ = self._pad_sequences(batch_returns_list)
            
            # Create mask from sequence lengths
            max_len = states_b.size(1)
            mask_b = torch.arange(max_len).unsqueeze(0) < seq_lengths_b.unsqueeze(1) # (batch_size, max_len)
            
            # Normalize advantages using only valid (masked) steps
            valid_advantages = advantages_b_unnorm[mask_b]
            if len(valid_advantages) > 1:
                advantages_b_norm = (advantages_b_unnorm - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
                advantages_b_norm[~mask_b] = 0.0 # Zero out padded advantages after normalization
            else: # Avoid std=0 if only one valid advantage or all same
                 advantages_b_norm = advantages_b_unnorm 
                 advantages_b_norm[~mask_b] = 0.0


            batch_data = {
                "states": states_b, # (batch_size, max_seq_len, state_dim)
                "actions": actions_b, # (batch_size, max_seq_len, action_dim)
                "old_log_probs": old_log_probs_b, # (batch_size, max_seq_len, 1)
                "advantages": advantages_b_norm.unsqueeze(-1), # (batch_size, max_seq_len, 1)
                "returns": returns_b, # (batch_size, max_seq_len, 1)
                "seq_lengths": seq_lengths_b, # (batch_size,)
                "mask": mask_b # (batch_size, max_seq_len)
            }
            all_batches_data.append(batch_data)

        # Clear main rollout storage after generating all batches for an update cycle
        self.rollouts_states.clear()
        self.rollouts_actions.clear()
        self.rollouts_old_log_probs.clear()
        self.rollouts_advantages.clear()
        self.rollouts_returns.clear()
        self.rollouts_seq_lengths.clear()
        
        return all_batches_data


    def __len__(self):
        """Returns the number of full rollouts stored."""
        return len(self.rollouts_states)


class PolicyNetwork(nn.Module):
    """Actor network for PPO. Takes pre-normalized basic_state (MLP) or sequence of basic_states (RNN) as input."""
    def __init__(self, ppo_config: PPOConfig, world_config: WorldConfig):
        super(PolicyNetwork, self).__init__()
        self.ppo_config = ppo_config
        self.world_config = world_config
        self.use_rnn = ppo_config.use_rnn
        self.state_dim = ppo_config.state_dim 
        self.action_dim = ppo_config.action_dim
        
        if self.use_rnn:
            self.rnn_hidden_size = ppo_config.rnn_hidden_size
            self.rnn_num_layers = ppo_config.rnn_num_layers
            rnn_input_dim = self.state_dim 
            if ppo_config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=self.rnn_hidden_size,
                                   num_layers=self.rnn_num_layers, batch_first=True)
            elif ppo_config.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=self.rnn_hidden_size,
                                  num_layers=self.rnn_num_layers, batch_first=True)
            else: raise ValueError(f"Unsupported RNN type: {ppo_config.rnn_type}")
            mlp_input_dim = self.rnn_hidden_size
        else:
            self.rnn = None
            mlp_input_dim = self.state_dim 

        self.fc1 = nn.Linear(mlp_input_dim, ppo_config.hidden_dim)
        self.fc2 = nn.Linear(ppo_config.hidden_dim, ppo_config.hidden_dim)
        self.mean_layer = nn.Linear(ppo_config.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))
        self.log_std_min = ppo_config.log_std_min
        self.log_std_max = ppo_config.log_std_max

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        next_hidden_state = None
        if self.use_rnn:
            if lengths is not None: # Training with packed sequences
                packed_input = pack_padded_sequence(network_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, next_hidden_state = self.rnn(packed_input, hidden_state)
                rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=network_input.size(1))
                # rnn_output shape: (batch, max_seq_len, rnn_hidden_size) - used by MLP for all timesteps
                mlp_features = rnn_output 
            else: # Action selection (single step or fixed N-length history without padding)
                rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
                mlp_features = rnn_output[:, -1, :] # Use last output for single action decision
        else:
            mlp_features = network_input
        
        x = F.relu(self.fc1(mlp_features))
        x = F.relu(self.fc2(x))
        action_mean = self.mean_layer(x)
        
        # If lengths were provided, action_mean is (batch, max_seq_len, action_dim)
        # log_std needs to be broadcastable: (1, 1, action_dim) or (1, action_dim)
        # Current self.log_std is (1, action_dim). Needs to align for broadcasting.
        current_log_std = self.log_std
        if self.use_rnn and lengths is not None: # For sequence output
             current_log_std = self.log_std.unsqueeze(0) # (1, 1, action_dim)

        action_log_std_clamped = torch.clamp(current_log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_log_std_clamped)

        # Ensure action_std is broadcastable with action_mean
        if self.use_rnn and lengths is not None and action_mean.dim() == 3 and action_std.dim() == 3:
             action_std = action_std.expand_as(action_mean)
        elif action_mean.dim() == 2 and action_std.dim() == 2: # MLP case or RNN last step output
             action_std = action_std.expand_as(action_mean)


        return action_mean, action_std, next_hidden_state


    def sample(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        # For sample (action selection), lengths is None, forward will use rnn_output[:, -1, :]
        mean, std, next_hidden_state = self.forward(network_input, hidden_state, lengths=None)
        distribution = Normal(mean, std)
        x_t = distribution.sample()
        action_normalized = torch.tanh(x_t) 
        log_prob_unbounded = distribution.log_prob(x_t)
        clamped_tanh = action_normalized.clamp(-0.999999, 0.999999)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(-1, keepdim=True) # Sum over action_dim
        return action_normalized, log_prob, next_hidden_state

    def evaluate(self, network_input: torch.Tensor, action_normalized: torch.Tensor, 
                 hidden_state: Optional[Tuple] = None, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        # For evaluate (training update), lengths can be provided for packed sequences
        mean, std, next_hidden_state = self.forward(network_input, hidden_state, lengths=lengths)
        distribution = Normal(mean, std)
        
        # Ensure action_normalized has same sequence dim as mean/std if lengths were used
        # action_normalized from memory is (batch, max_seq_len, action_dim)
        
        action_tanh = torch.clamp(action_normalized, -0.99999, 0.99999) 
        action_original_space = torch.atanh(action_tanh) 
        
        log_prob_unbounded = distribution.log_prob(action_original_space)
        log_det_jacobian = torch.log(1.0 - action_tanh.pow(2) + 1e-7) 
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(-1, keepdim=True) # Sum over action_dim
        entropy = distribution.entropy().sum(-1, keepdim=True) # Sum over action_dim
        return log_prob, entropy, next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.ppo_config.rnn_num_layers, batch_size, self.ppo_config.rnn_hidden_size).to(device)
        if self.ppo_config.rnn_type == 'lstm':
            c_zeros = torch.zeros(self.ppo_config.rnn_num_layers, batch_size, self.ppo_config.rnn_hidden_size).to(device)
            return (h_zeros, c_zeros)
        return h_zeros 


class ValueNetwork(nn.Module):
    """Critic network for PPO. Takes pre-normalized basic_state (MLP) or sequence (RNN) as input."""
    def __init__(self, ppo_config: PPOConfig, world_config: WorldConfig):
        super(ValueNetwork, self).__init__()
        self.ppo_config = ppo_config
        self.world_config = world_config
        self.use_rnn = ppo_config.use_rnn
        self.state_dim = ppo_config.state_dim 

        if self.use_rnn:
            self.rnn_hidden_size = ppo_config.rnn_hidden_size
            self.rnn_num_layers = ppo_config.rnn_num_layers
            rnn_input_dim = self.state_dim
            if ppo_config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=self.rnn_hidden_size,
                                   num_layers=self.rnn_num_layers, batch_first=True)
            elif ppo_config.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=self.rnn_hidden_size,
                                  num_layers=self.rnn_num_layers, batch_first=True)
            else: raise ValueError(f"Unsupported RNN type: {ppo_config.rnn_type}")
            mlp_input_dim = self.rnn_hidden_size
        else:
            self.rnn = None
            mlp_input_dim = self.state_dim
            
        self.fc1 = nn.Linear(mlp_input_dim, ppo_config.hidden_dim)
        self.fc2 = nn.Linear(ppo_config.hidden_dim, ppo_config.hidden_dim)
        self.value_layer = nn.Linear(ppo_config.hidden_dim, 1)

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        next_hidden_state = None
        if self.use_rnn:
            if lengths is not None: # Training with packed sequences
                packed_input = pack_padded_sequence(network_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, next_hidden_state = self.rnn(packed_input, hidden_state)
                rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=network_input.size(1))
                mlp_features = rnn_output # Value function output per time step
            else: # Action selection or fixed N-length history without padding
                rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
                mlp_features = rnn_output[:, -1, :] # Use last output for single value V(s_t)
        else:
            mlp_features = network_input
            
        x = F.relu(self.fc1(mlp_features))
        x = F.relu(self.fc2(x))
        return self.value_layer(x), next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.ppo_config.rnn_num_layers, batch_size, self.ppo_config.rnn_hidden_size).to(device)
        if self.ppo_config.rnn_type == 'lstm':
            c_zeros = torch.zeros(self.ppo_config.rnn_num_layers, batch_size, self.ppo_config.rnn_hidden_size).to(device)
            return (h_zeros, c_zeros)
        return h_zeros

class PPO:
    """PPO. Assumes states from World are already normalized if world_config.normalize_state=True."""
    def __init__(self, config: PPOConfig, training_config: TrainingConfig, world_config: WorldConfig, device: torch.device = None):
        self.config = config
        self.training_config = training_config 
        self.world_config = world_config
        self.use_rnn = config.use_rnn
        self.gamma = config.gamma; self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip; self.n_epochs = config.n_epochs
        self.entropy_coef = config.entropy_coef; self.value_coef = config.value_coef
        self.state_dim = config.state_dim # CORE_STATE_DIM
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")
        print(f"PPO Agent expects states to be normalized by World: {self.world_config.normalize_state}")
        if self.use_rnn: print(f"PPO Agent using RNN: Type={config.rnn_type}, Hidden={config.rnn_hidden_size}, Layers={config.rnn_num_layers}")
        else: print(f"PPO Agent using MLP")

        if self.training_config.normalize_rewards:
            self.reward_normalizer = RunningMeanStd(shape=(1,)).to(self.device)
            print("PPO Agent reward normalization enabled.")
        else:
            self.reward_normalizer = None

        self.actor = PolicyNetwork(config, world_config).to(self.device)
        self.critic = ValueNetwork(config, world_config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.memory = RecurrentPPOMemory(ppo_config=config, world_config=world_config)

    def select_action(self, norm_basic_state: Tuple, 
                      actor_hidden_state: Optional[Tuple]=None, 
                      critic_hidden_state: Optional[Tuple]=None, 
                      evaluate=False) -> Tuple[float, Optional[Tuple], Optional[Tuple], float, float]:
        """
        Select action. norm_basic_state is a pre-normalized tuple.
        Returns: action, next_actor_h (detached), next_critic_h (detached), log_prob, value
        """
        # For RNN, input is (1, 1, state_dim) - current step only, hidden state carries history
        # For MLP, input is (1, state_dim)
        if self.use_rnn:
            network_input_tensor = torch.FloatTensor(norm_basic_state).to(self.device).unsqueeze(0).unsqueeze(0) # (1, 1, state_dim)
        else:
            network_input_tensor = torch.FloatTensor(norm_basic_state).to(self.device).unsqueeze(0) # (1, state_dim)

        log_prob_np, value_np = 0.0, 0.0
        next_actor_h_detached, next_critic_h_detached = None, None

        with torch.no_grad():
            self.actor.eval(); self.critic.eval()
            
            if evaluate: # For evaluation, only actor mean
                action_mean, _, next_actor_h = self.actor.forward(network_input_tensor, actor_hidden_state, lengths=None)
                action_normalized = torch.tanh(action_mean)
            else: # For training data collection
                action_normalized, log_prob, next_actor_h = self.actor.sample(network_input_tensor, actor_hidden_state)
                log_prob_np = log_prob.cpu().numpy()[0, 0]
            
            # Get value V(s_t)
            value_tensor, next_critic_h = self.critic(network_input_tensor, critic_hidden_state, lengths=None)
            value_np = value_tensor.cpu().numpy()[0,0] # If MLP, value_tensor is (1,1). If RNN (1,1,1). Squeeze appropriately.
            if value_np.ndim > 0 : value_np = value_np.item()


            if self.use_rnn: # Detach hidden states for next step in rollout
                next_actor_h_detached = detach_hidden_state(next_actor_h)
                next_critic_h_detached = detach_hidden_state(next_critic_h)
            
            self.actor.train(); self.critic.train()
            
        return action_normalized.detach().cpu().numpy().item(), \
               next_actor_h_detached, next_critic_h_detached, \
               log_prob_np, value_np


    def update_parameters(self):
        # Generates batches of (padded_sequences, masks, lengths, etc.)
        batched_rollout_data = self.memory.generate_batches()
        if not batched_rollout_data: 
            return None

        actor_losses_all_batches, critic_losses_all_batches, entropies_all_batches = [], [], []

        for batch_data in batched_rollout_data:
            states_b = batch_data["states"].to(self.device)
            actions_b = batch_data["actions"].to(self.device)
            old_log_probs_b = batch_data["old_log_probs"].to(self.device)
            advantages_b = batch_data["advantages"].to(self.device) # Already (batch, max_len, 1)
            returns_b = batch_data["returns"].to(self.device)
            seq_lengths_b = batch_data["seq_lengths"].to(self.device) # For pack_padded_sequence
            mask_b = batch_data["mask"].to(self.device) # (batch, max_len) for masking losses
            mask_b_expanded = mask_b.unsqueeze(-1) # (batch, max_len, 1) for element-wise ops with (batch, max_len, 1) tensors
            
            current_batch_actual_size = states_b.size(0) # Actual number of sequences in this batch

            # RNN hidden states for batch processing during updates - re-initialize to zeros for each batch
            initial_actor_h_batch = None
            initial_critic_h_batch = None
            if self.use_rnn:
                initial_actor_h_batch = self.actor.get_initial_hidden_state(current_batch_actual_size, self.device)
                initial_critic_h_batch = self.critic.get_initial_hidden_state(current_batch_actual_size, self.device)
            
            for _ in range(self.n_epochs):
                # Pass seq_lengths_b to actor.evaluate and critic forward for pack_padded_sequence
                new_log_probs, entropy, _ = self.actor.evaluate(states_b, actions_b, initial_actor_h_batch, lengths=seq_lengths_b if self.use_rnn else None)
                new_values, _ = self.critic(states_b, initial_critic_h_batch, lengths=seq_lengths_b if self.use_rnn else None)
                
                # Ensure dimensions are (batch, max_seq_len, 1)
                new_log_probs = new_log_probs.view_as(old_log_probs_b)
                entropy = entropy.view_as(old_log_probs_b)
                new_values = new_values.view_as(returns_b)

                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantages_b
                
                # Apply mask to losses
                actor_loss_unmasked = -torch.min(surr1, surr2)
                critic_loss_unmasked = F.mse_loss(new_values, returns_b, reduction='none') # Get element-wise loss
                entropy_loss_unmasked = -entropy

                # Masked mean
                actor_loss = (actor_loss_unmasked * mask_b_expanded).sum() / mask_b_expanded.sum()
                critic_loss = (critic_loss_unmasked * mask_b_expanded).sum() / mask_b_expanded.sum()
                entropy_loss = (entropy_loss_unmasked * mask_b_expanded).sum() / mask_b_expanded.sum()
                
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.actor_optimizer.zero_grad(); self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step(); self.critic_optimizer.step()

            actor_losses_all_batches.append(actor_loss.item())
            critic_losses_all_batches.append(critic_loss.item())
            entropies_all_batches.append(entropy_loss.item() / (-self.entropy_coef if self.entropy_coef !=0 else 1)) # Average entropy, not entropy loss term

        return {
            'actor_loss': np.mean(actor_losses_all_batches), 
            'critic_loss': np.mean(critic_losses_all_batches), 
            'entropy': np.mean(entropies_all_batches)
        }

    def save_model(self, path: str):
        print(f"Saving PPO model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'device_type': self.device.type
        }
        if self.reward_normalizer: 
            save_dict['reward_normalizer_state_dict'] = self.reward_normalizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path): print(f"Warn: PPO model file not found: {path}. Skip loading."); return
        print(f"Loading PPO model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.reward_normalizer and 'reward_normalizer_state_dict' in checkpoint:
            self.reward_normalizer.load_state_dict(checkpoint['reward_normalizer_state_dict'])
            print("Loaded PPO reward normalizer statistics.")
        elif self.reward_normalizer:
            print("Warning: PPO reward normalizer statistics not found in checkpoint (reward_normalizer is active).")
        print(f"PPO model loaded successfully from {path}")


# --- Training Loop (train_ppo) ---
def train_ppo(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True,
              models_save_path: str = None, tensorboard_log_path: str = None):
    ppo_config = config.ppo
    train_config = config.training
    world_config = config.world 
    cuda_device = config.cuda_device

    tb_log_path = tensorboard_log_path or os.path.join("runs", f"ppo_recurrent_{ppo_config.use_rnn}_{int(time.time())}")
    model_path_base = models_save_path or train_config.models_dir 
    os.makedirs(tb_log_path, exist_ok=True); writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logs: {tb_log_path}")

    if torch.cuda.is_available():
        device = torch.device(cuda_device)
        if 'cuda' in cuda_device:
            try: torch.cuda.set_device(device); print(f"GPU: {torch.cuda.get_device_name(device)}")
            except: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else: device = torch.device("cpu"); print("GPU not available, using CPU.")

    agent = PPO(config=ppo_config, training_config=train_config, world_config=world_config, device=device)
    os.makedirs(model_path_base, exist_ok=True)

    latest_model_file = None
    if os.path.exists(model_path_base) and os.path.isdir(model_path_base):
        model_files = [f for f in os.listdir(model_path_base) if f.startswith("ppo_") and f.endswith(".pt")]
        if model_files:
            try: latest_model_file = max([os.path.join(model_path_base, f) for f in model_files], key=os.path.getmtime)
            except: pass 

    total_env_steps, start_episode = 0, 1
    if latest_model_file and os.path.exists(latest_model_file):
        print(f"\nResuming PPO training from: {latest_model_file}")
        agent.load_model(latest_model_file)
        try:
            parts = os.path.basename(latest_model_file).split('_')
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            step_part = next((p for p in parts if p.startswith('stepenv')), None) # Look for 'stepenv'
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            if step_part: total_env_steps = int(step_part.replace('stepenv', '').split('.')[0])
            print(f"Resuming at env_step {total_env_steps}, episode {start_episode}")
        except: print(f"Warn: Could not parse from {latest_model_file}. New counts."); total_env_steps=0;start_episode=1
    else: print(f"\nStarting PPO training from scratch. Models saved in {model_path_base}")

    episode_rewards, timing_metrics = [], {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1), desc="Training PPO", unit="episode", initial=start_episode-1, total=train_config.num_episodes)
    world = World(world_config=world_config)
    
    steps_since_last_update = 0

    for episode in pbar:
        world.reset()
        # state_dict = world.encode_state() -> state_dict['basic_state'] is the normalized basic state tuple
        current_norm_basic_state = world.encode_state()['basic_state'] 
        ep_reward_raw, ep_steps = 0, 0
        
        actor_h, critic_h = None, None
        if agent.use_rnn:
            actor_h = agent.actor.get_initial_hidden_state(1, device)
            critic_h = agent.critic.get_initial_hidden_state(1, device)

        for step_in_episode in range(train_config.max_steps):
            action_norm, next_actor_h, next_critic_h, log_prob, value = \
                agent.select_action(current_norm_basic_state, actor_h, critic_h, evaluate=False)
            
            step_start_time = time.time()
            # World step uses action_norm, training=True, terminal_step is for GAE calculation if episode ends by max_steps
            world.step(action_norm, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            timing_metrics['env_step_time'].append(time.time() - step_start_time)

            raw_reward_this_step = world.reward 
            next_norm_basic_state = world.encode_state()['basic_state']
            done_bool = world.done # Actual done from environment
            
            # Store step data in RecurrentPPOMemory's current rollout buffer
            agent.memory.store_step(current_norm_basic_state, action_norm, log_prob, value,
                                    raw_reward_this_step, done_bool,
                                    next_actor_h, next_critic_h) # Store hiddens AFTER state s_t

            current_norm_basic_state = next_norm_basic_state
            if agent.use_rnn:
                actor_h, critic_h = next_actor_h, next_critic_h
            
            ep_reward_raw += raw_reward_this_step
            ep_steps += 1; total_env_steps += 1; steps_since_last_update += 1

            if done_bool or steps_since_last_update >= ppo_config.steps_per_update:
                # Calculate last_value for GAE
                last_val = 0.0
                if not done_bool: # If rollout truncated by steps_per_update
                    # Get V(S_{t+rollout_len})
                    if agent.use_rnn:
                        temp_input = torch.FloatTensor(next_norm_basic_state).to(device).unsqueeze(0).unsqueeze(0)
                        temp_h_critic = agent.critic.get_initial_hidden_state(1, device) # Use fresh hidden for this single eval
                                                                                       # or pass critic_h if want to use current episode context
                        with torch.no_grad():
                            last_val_tensor, _ = agent.critic(temp_input, critic_h, lengths=None) # Use current critic_h
                            last_val = last_val_tensor.cpu().numpy().item()
                    else:
                        temp_input = torch.FloatTensor(next_norm_basic_state).to(device).unsqueeze(0)
                        with torch.no_grad():
                            last_val_tensor, _ = agent.critic(temp_input, None, lengths=None)
                            last_val = last_val_tensor.cpu().numpy().item()
                
                agent.memory.finalize_rollout(last_val, agent.reward_normalizer)
                steps_since_last_update = 0 # Reset counter

                if done_bool and agent.use_rnn: # Reset hidden states for new episode
                    actor_h = agent.actor.get_initial_hidden_state(1, device)
                    critic_h = agent.critic.get_initial_hidden_state(1, device)
            
            if done_bool: break # End episode
        
        # After collecting enough rollouts (implicit by steps_per_update leading to finalizations)
        # PPO updates typically happen after a certain number of environment steps,
        # which here means after enough rollouts are finalized to form batches.
        # The generate_batches call will return batches if memory has enough rollouts.
        if len(agent.memory) >= ppo_config.batch_size : # Check if enough rollouts are stored to form at least one batch
            update_start_time = time.time()
            losses = agent.update_parameters() 
            if losses: 
                timing_metrics['parameter_update_time'].append(time.time() - update_start_time)
                writer.add_scalar('Loss/Actor', losses['actor_loss'], total_env_steps)
                writer.add_scalar('Loss/Critic', losses['critic_loss'], total_env_steps)
                writer.add_scalar('Policy/Entropy', losses['entropy'], total_env_steps)
        
        episode_rewards.append(ep_reward_raw)
        if episode % train_config.log_frequency == 0:
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Env_Step_ms_Avg100', np.mean(timing_metrics['env_step_time'])*1000, total_env_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time'])*1000, total_env_steps)
            
            writer.add_scalar('Reward/Episode_Raw', ep_reward_raw, total_env_steps)
            writer.add_scalar('Steps/Episode', ep_steps, total_env_steps)
            writer.add_scalar('Progress/Total_Env_Steps', total_env_steps, episode)
            writer.add_scalar('Error/Distance_EndEp', world.error_dist, total_env_steps)
            writer.add_scalar('Buffer/PPO_Num_Rollouts_Stored', len(agent.memory), total_env_steps)

            if agent.reward_normalizer and agent.reward_normalizer.count > agent.reward_normalizer.epsilon:
                writer.add_scalar('Stats/PPO_RewardNorm_Mean', agent.reward_normalizer.mean[0].item(), total_env_steps)
                writer.add_scalar('Stats/PPO_RewardNorm_Std', torch.sqrt(agent.reward_normalizer.var[0].clamp(min=0.0)).item(), total_env_steps)
            
            avg_rew100 = np.mean(episode_rewards[-min(100,len(episode_rewards)):]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100_Raw', avg_rew100, total_env_steps)
            pbar.set_postfix({'avg_rew10_raw': f'{np.mean(episode_rewards[-min(10,len(episode_rewards)):]) if episode_rewards else 0:.2f}', 'steps': total_env_steps})

        if episode % train_config.save_interval == 0:
            agent.save_model(os.path.join(model_path_base, f"ppo_ep{episode}_stepenv{total_env_steps}.pt"))

    pbar.close(); writer.close()
    print(f"PPO Training finished. Total env steps: {total_env_steps}")
    agent.save_model(os.path.join(model_path_base, f"ppo_final_ep{train_config.num_episodes}_stepenv{total_env_steps}.pt"))
    if run_evaluation: print("\nStarting evaluation..."); evaluate_ppo(agent, config)
    return agent, episode_rewards


# --- Evaluation Loop (evaluate_ppo) ---
def evaluate_ppo(agent: PPO, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world 
    vis_config = config.visualization
    ppo_config = config.ppo 

    vis_save_dir_actual = vis_config.save_dir
    if not os.path.isabs(vis_save_dir_actual) and hasattr(config, '_experiment_path_for_vis'):
        vis_save_dir_actual = os.path.join(config._experiment_path_for_vis, vis_save_dir_actual)

    vis_available = False
    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            vis_available = True; print("Visualization enabled.")
            if vis_save_dir_actual: os.makedirs(vis_save_dir_actual, exist_ok=True)
        except ImportError: print("Vis libs not found. Rendering disabled."); eval_config.render = False
    else: print("Rendering disabled by config.")

    eval_rewards_raw, success_count, all_gif_paths = [], 0, []
    agent.actor.eval(); agent.critic.eval() 
    if agent.reward_normalizer: agent.reward_normalizer.eval() 

    print(f"\nRunning PPO Evaluation for {eval_config.num_episodes} episodes...")
    world = World(world_config=world_config)

    for episode in range(eval_config.num_episodes):
        world.reset()
        current_norm_basic_state = world.encode_state()['basic_state']
        ep_reward_raw, episode_frames = 0, []
        
        actor_h, critic_h = None, None
        if ppo_config.use_rnn:
            actor_h = agent.actor.get_initial_hidden_state(1, agent.device)
            critic_h = agent.critic.get_initial_hidden_state(1, agent.device)


        if eval_config.render and vis_available:
            current_vis_config = vis_config.model_copy(); current_vis_config.save_dir = vis_save_dir_actual
            reset_trajectories()
            try:
                fname = f"ppo_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame = visualize_world(world, current_vis_config, fname, True)
                if initial_frame and os.path.exists(initial_frame): episode_frames.append(initial_frame)
            except Exception as e: print(f"Warn: Vis failed init ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            action_normalized, next_actor_h, next_critic_h, _, _ = \
                agent.select_action(current_norm_basic_state, actor_h, critic_h, evaluate=True)
            
            world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            raw_reward = world.reward 
            next_norm_basic_state = world.encode_state()['basic_state']
            done = world.done

            if eval_config.render and vis_available:
                current_vis_config = vis_config.model_copy(); current_vis_config.save_dir = vis_save_dir_actual
                try:
                    fname = f"ppo_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, current_vis_config, fname, True)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")
            
            current_norm_basic_state = next_norm_basic_state
            if ppo_config.use_rnn:
                actor_h, critic_h = next_actor_h, next_critic_h
            ep_reward_raw += raw_reward
            if done: break
        
        success = world.error_dist <= world_config.success_threshold
        status_msg = "Success!" if success else "Failure."
        if success: success_count += 1
        term_reason = f"Terminated Step {step+1}" if done else f"Finished (Max steps {eval_config.max_steps})"
        print(f"  Episode {episode+1}: {term_reason}. Final Err: {world.error_dist:.2f}. {status_msg}")
        eval_rewards_raw.append(ep_reward_raw)
        print(f"  Episode {episode+1}: Total Raw Reward: {ep_reward_raw:.2f}")

        if eval_config.render and vis_available and episode_frames:
            current_vis_config = vis_config.model_copy(); current_vis_config.save_dir = vis_save_dir_actual
            gif_fname = f"ppo_eval_episode_{episode+1}.gif"
            try:
                gif_p = save_gif(gif_fname, current_vis_config, episode_frames, current_vis_config.delete_frames_after_gif)
                if gif_p: all_gif_paths.append(gif_p)
            except Exception as e: print(f"Warn: Failed GIF save ep {episode+1}. E: {e}")

    agent.actor.train(); agent.critic.train() 
    if agent.reward_normalizer: agent.reward_normalizer.train()

    avg_eval_rew = np.mean(eval_rewards_raw) if eval_rewards_raw else 0
    std_eval_rew = np.std(eval_rewards_raw) if eval_rewards_raw else 0
    success_rt = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0
    print("\n--- PPO Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}, Avg Raw Reward: {avg_eval_rew:.2f} +/- {std_eval_rew:.2f}")
    print(f"Success Rate (Err <= {world_config.success_threshold}): {success_rt:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_save_dir_actual)}'")
    print("--- End PPO Evaluation ---\n")
    return eval_rewards_raw, success_rt
