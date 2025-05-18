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

# Local imports
from world import World
from configs import DefaultConfig, ReplayBufferConfig, SACConfig, WorldConfig, TrainingConfig, CORE_STATE_DIM
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional
from utils import RunningMeanStd

class ReplayBuffer:
    """Experience replay buffer storing full state trajectories.
       Assumes trajectories in (state, next_state) are already normalized by the World.
       Rewards stored are raw.
    """

    def __init__(self, config: ReplayBufferConfig, world_config: WorldConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim # norm_state_dim + action_dim + reward_dim

    def push(self, state: dict, action, reward, next_state: dict, done):
        """ Add a new experience to memory. `state` and `next_state` contain normalized trajectories. Stores raw reward. """
        # 'full_trajectory' from world.encode_state() is (normalized_s, action, raw_r)
        # For SAC, we need (s_norm_traj, action_taken_after_s_norm_traj, raw_reward_for_that_action, next_s_norm_traj, done)
        # The state dict from world.encode_state() contains 'full_trajectory' which IS the s_norm_traj.
        
        current_normalized_trajectory = state['full_trajectory'] # This is S_t (sequence of N normalized states, actions, raw_rewards)
        # action is a_t (single float)
        # reward is r_t (single float, raw)
        next_normalized_trajectory = next_state['full_trajectory'] # This is S_{t+1} (sequence for next step)
        
        if isinstance(action, (np.ndarray, list)): action = action[0]
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        if not isinstance(current_normalized_trajectory, np.ndarray) or current_normalized_trajectory.shape != (self.trajectory_length, self.feature_dim):
            return
        if not isinstance(next_normalized_trajectory, np.ndarray) or next_normalized_trajectory.shape != (self.trajectory_length, self.feature_dim):
            return
        if np.isnan(current_normalized_trajectory).any() or np.isnan(next_normalized_trajectory).any():
             return 

        # Store (current_normalized_full_trajectory, current_action, raw_reward, next_normalized_full_trajectory, done)
        self.buffer.append((current_normalized_trajectory, float(action), float(reward), next_normalized_trajectory, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        """Sample a batch of experiences from memory."""
        if len(self.buffer) < batch_size:
            return None
        try:
            batch = random.sample(self.buffer, batch_size)
        except ValueError:
            print(f"Warning: ReplayBuffer sampling failed. len(buffer)={len(self.buffer)}, batch_size={batch_size}")
            return None

        # state_norm_traj_batch, action_batch, raw_reward_batch, next_state_norm_traj_batch, done_batch
        trajectory_arr, action_arr_intermediate, reward_arr, next_trajectory_arr, done_arr = zip(*batch)

        try:
            # trajectory_arr and next_trajectory_arr are already normalized trajectories from World
            # (batch_size, trajectory_length, feature_dim) where feature_dim includes norm_state, action, raw_reward
            state_norm_traj_batch = np.array(trajectory_arr, dtype=np.float32)
            action_batch = np.array(action_arr_intermediate, dtype=np.float32).reshape(-1, 1) 
            raw_reward_batch = np.array(reward_arr, dtype=np.float32) # Shape (batch,) - raw rewards
            next_state_norm_traj_batch = np.array(next_trajectory_arr, dtype=np.float32)
            done_batch = np.array(done_arr, dtype=np.float32) 

            if np.isnan(state_norm_traj_batch).any() or np.isnan(next_state_norm_traj_batch).any() or np.isnan(raw_reward_batch).any():
                print("Warning: Sampled batch contains NaN values. Returning None.")
                return None

        except Exception as e:
            print(f"Error converting sampled batch to numpy arrays: {e}")
            return None

        return (state_norm_traj_batch, action_batch, raw_reward_batch, next_state_norm_traj_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Policy network (Actor) for SAC, optionally with RNN. Expects pre-normalized state inputs."""

    def __init__(self, config: SACConfig, world_config: WorldConfig):
        super(Actor, self).__init__()
        self.config = config
        self.world_config = world_config # world_config.normalize_state indicates if input is normalized
        self.use_rnn = config.use_rnn
        self.state_dim = config.state_dim # Dimension of basic state (already normalized if world_config.normalize_state)
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length # Not directly used by MLP Actor if only last state is input

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            # RNN processes sequences of (already normalized) basic_states
            rnn_input_dim = self.state_dim 
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
            # MLP uses only the last (already normalized) basic state
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

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass through the network.
           network_input is pre-normalized:
             - RNN: (batch, seq_len, state_dim) - sequence of normalized basic states
             - MLP: (batch, state_dim) - last normalized basic state
        """
        next_hidden_state = None
        if self.use_rnn:
            # network_input is (batch, seq_len, state_dim) - already normalized basic states
            rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
            mlp_input = rnn_output[:, -1, :] 
        else:
            # network_input is (batch, state_dim) - already normalized last basic state
            mlp_input = network_input 

        x = mlp_input
        for layer in self.layers:
            x = layer(x)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, next_hidden_state

    def sample(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        mean, log_std, next_hidden_state = self.forward(network_input, hidden_state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action_normalized = torch.tanh(x_t)
        log_prob_unbounded = normal.log_prob(x_t)
        clamped_tanh = action_normalized.clamp(-0.999999, 0.999999) 
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7) 
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)
        return action_normalized, log_prob, torch.tanh(mean), next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.config.rnn_type == 'lstm':
            c_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
            return (h_zeros, c_zeros)
        elif self.config.rnn_type == 'gru':
            return h_zeros
        return None

class Critic(nn.Module):
    """Q-function network (Critic) for SAC, optionally with RNN. Expects pre-normalized state inputs."""
    def __init__(self, config: SACConfig, world_config: WorldConfig):
        super(Critic, self).__init__()
        self.config = config
        self.world_config = world_config # world_config.normalize_state indicates if input is normalized
        self.use_rnn = config.use_rnn
        self.state_dim = config.state_dim # Dimension of basic state (already normalized)
        self.action_dim = config.action_dim

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # RNN processes sequences of (already normalized) basic_states
            rnn_cell = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU
            self.rnn1 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            self.rnn2 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            mlp_input_dim = config.rnn_hidden_size + self.action_dim
        else:
            # MLP uses only the last (already normalized) basic state + action
            mlp_input_dim = self.state_dim + self.action_dim
            self.rnn1, self.rnn2 = None, None

        self.q1_layers = nn.ModuleList()
        q1_mlp_input = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.q1_layers.append(nn.Linear(q1_mlp_input, hidden_dim))
            self.q1_layers.append(nn.ReLU())
            q1_mlp_input = hidden_dim
        self.q1_out = nn.Linear(q1_mlp_input, 1)

        self.q2_layers = nn.ModuleList()
        q2_mlp_input = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.q2_layers.append(nn.Linear(q2_mlp_input, hidden_dim))
            self.q2_layers.append(nn.ReLU())
            q2_mlp_input = hidden_dim
        self.q2_out = nn.Linear(q2_mlp_input, 1)

    def forward(self, network_input: torch.Tensor, action: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass returning both Q-values.
           network_input is pre-normalized:
             - RNN: (batch, seq_len, state_dim) - sequence of normalized basic states
             - MLP: (batch, state_dim) - last normalized basic state
           action shape: (batch, action_dim)
        """
        next_hidden_state = None
        if self.use_rnn:
            h1_in, h2_in = None, None
            if isinstance(hidden_state, tuple) and len(hidden_state) == 2:
                 h1_in, h2_in = hidden_state
            rnn_out1, next_h1 = self.rnn1(network_input, h1_in)
            rnn_out2, next_h2 = self.rnn2(network_input, h2_in)
            next_hidden_state = (next_h1, next_h2)
            mlp_input1 = torch.cat([rnn_out1[:, -1, :], action], dim=1)
            mlp_input2 = torch.cat([rnn_out2[:, -1, :], action], dim=1)
        else:
            mlp_input1 = torch.cat([network_input, action], dim=1)
            mlp_input2 = torch.cat([network_input, action], dim=1)

        x1 = mlp_input1
        for layer in self.q1_layers: x1 = layer(x1)
        q1 = self.q1_out(x1)
        x2 = mlp_input2
        for layer in self.q2_layers: x2 = layer(x2)
        q2 = self.q2_out(x2)
        return q1, q2, next_hidden_state

    def q1_forward(self, network_input: torch.Tensor, action: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        next_hidden_state = None
        if self.use_rnn:
            h1_in = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
            rnn_out1, next_h1 = self.rnn1(network_input, h1_in)
            next_hidden_state = (next_h1, None) 
            mlp_input1 = torch.cat([rnn_out1[:, -1, :], action], dim=1)
        else:
            mlp_input1 = torch.cat([network_input, action], dim=1)
        x1 = mlp_input1
        for layer in self.q1_layers: x1 = layer(x1)
        q1 = self.q1_out(x1)
        return q1, next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h_zeros = lambda: torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.config.rnn_type == 'lstm':
            return ((h_zeros(), h_zeros()), (h_zeros(), h_zeros())) # (h1_lstm, h2_lstm) where each is (h,c)
        elif self.config.rnn_type == 'gru':
            return (h_zeros(), h_zeros()) # (h1_gru, h2_gru)
        return None

class SAC:
    """Soft Actor-Critic. Assumes states from World are already normalized if world_config.normalize_state is True."""
    def __init__(self, config: SACConfig, world_config: WorldConfig, training_config: TrainingConfig, device: torch.device = None):
        self.config = config
        self.world_config = world_config 
        self.training_config = training_config
        self.gamma = config.gamma; self.tau = config.tau; self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha; self.use_rnn = config.use_rnn
        self.trajectory_length = world_config.trajectory_length
        self.state_dim = config.state_dim; self.action_dim = config.action_dim

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent using device: {self.device}")
        print(f"SAC Agent expects states to be normalized by World: {world_config.normalize_state}")
        if self.use_rnn: print(f"SAC Agent using RNN: Type={config.rnn_type}, Hidden={config.rnn_hidden_size}, Layers={config.rnn_num_layers}")
        else: print(f"SAC Agent using MLP")

        if self.training_config.normalize_rewards:
            self.reward_normalizer = RunningMeanStd(shape=(1,)).to(self.device)
            print("SAC Agent reward normalization enabled.")
        else:
            self.reward_normalizer = None

        self.actor = Actor(config, world_config).to(self.device)
        self.critic = Critic(config, world_config).to(self.device)
        self.critic_target = Critic(config, world_config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters(): p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device) 

    def select_action(self, state: dict, actor_hidden_state: Optional[Tuple] = None, evaluate: bool = False) -> Tuple[float, Optional[Tuple]]:
        """Select action. `state` dict contains pre-normalized 'basic_state' or 'full_trajectory'."""
        # 'basic_state' is the last normalized basic state from trajectory
        # 'full_trajectory' is (N, normalized_feature_dim)
        # Actor input depends on RNN or MLP
        
        with torch.no_grad():
            if self.use_rnn:
                # Input: sequence of normalized basic states (N, state_dim)
                # state['full_trajectory'] is (N, feature_dim=norm_s_dim+action_dim+reward_dim)
                # We need just the normalized state part for RNN actor
                norm_basic_state_seq = torch.FloatTensor(state['full_trajectory'][:, :self.state_dim]).to(self.device).unsqueeze(0) # (1, N, state_dim)
                actor_input = norm_basic_state_seq
            else:
                # Input: last normalized basic state (state_dim)
                actor_input = torch.FloatTensor(state['basic_state']).to(self.device).unsqueeze(0) # (1, state_dim)

            self.actor.eval()
            if evaluate:
                _, _, action_mean_squashed, next_actor_hidden_state = self.actor.sample(actor_input, actor_hidden_state)
                action_normalized = action_mean_squashed
            else:
                action_normalized, _, _, next_actor_hidden_state = self.actor.sample(actor_input, actor_hidden_state)
            self.actor.train()

        return action_normalized.detach().cpu().numpy()[0, 0], next_actor_hidden_state

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        sampled_batch = memory.sample(batch_size)
        if sampled_batch is None: return None

        # state_norm_traj_batch: (batch, N, feature_dim = norm_s_dim + a_dim + r_raw_dim)
        # action_batch: (batch, action_dim) - action taken *after* state_norm_traj_batch
        # raw_reward_batch: (batch,) - raw reward for (state_norm_traj_batch, action_batch)
        # next_state_norm_traj_batch: (batch, N, feature_dim) - S'
        # done_batch: (batch,)
        state_norm_traj_batch, action_batch_tensor, raw_reward_batch_tensor, next_state_norm_traj_batch, done_batch_tensor = (
            torch.FloatTensor(d).to(self.device) if not isinstance(d, torch.Tensor) else d.to(self.device) for d in sampled_batch
        )
        # Ensure correct shapes for rewards and dones
        raw_reward_batch_tensor = raw_reward_batch_tensor.unsqueeze(1)
        done_batch_tensor = done_batch_tensor.unsqueeze(1)

        # Reward Normalization (if enabled) for r_t
        if self.reward_normalizer:
            self.reward_normalizer.update(raw_reward_batch_tensor) # Update with current batch's raw rewards
            reward_batch_tensor = self.reward_normalizer.normalize(raw_reward_batch_tensor)
        else:
            reward_batch_tensor = raw_reward_batch_tensor

        # Prepare network inputs from the normalized trajectories
        if self.use_rnn:
            # RNNs take sequence of normalized basic states: (batch, N, state_dim)
            current_norm_basic_state_seq = state_norm_traj_batch[:, :, :self.state_dim]
            next_norm_basic_state_seq = next_state_norm_traj_batch[:, :, :self.state_dim]
            current_network_input = current_norm_basic_state_seq
            next_network_input = next_norm_basic_state_seq
        else:
            # MLPs take last normalized basic state: (batch, state_dim)
            current_last_norm_basic_state = state_norm_traj_batch[:, -1, :self.state_dim]
            next_last_norm_basic_state = next_state_norm_traj_batch[:, -1, :self.state_dim]
            current_network_input = current_last_norm_basic_state
            next_network_input = next_last_norm_basic_state
        
        initial_actor_hidden = self.actor.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_hidden = self.critic.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None

        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.sample(next_network_input, initial_actor_hidden) # Use same hidden state logic if RNN for actor
            target_q1, target_q2, _ = self.critic_target(next_network_input, next_action, initial_critic_hidden) # Use same hidden for critic target
            target_q_min = torch.min(target_q1, target_q2)
            current_alpha = self.log_alpha.exp().item()
            target_q_entropy = target_q_min - current_alpha * next_log_prob
            y = reward_batch_tensor + (1.0 - done_batch_tensor) * self.gamma * target_q_entropy

        current_q1, current_q2, _ = self.critic(current_network_input, action_batch_tensor, initial_critic_hidden)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

        for p in self.critic.parameters(): p.requires_grad = False
        action_pi, log_prob_pi, _, _ = self.actor.sample(current_network_input, initial_actor_hidden)
        q1_pi, q2_pi, _ = self.critic(current_network_input, action_pi, initial_critic_hidden)
        q_pi_min = torch.min(q1_pi, q2_pi)
        current_alpha = self.log_alpha.exp().item()
        actor_loss = (current_alpha * log_prob_pi - q_pi_min).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
        for p in self.critic.parameters(): p.requires_grad = True

        alpha_loss_item = None
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_item = alpha_loss.item()

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(),
                'alpha': self.alpha, 'alpha_loss': alpha_loss_item if alpha_loss_item is not None else 0.0}

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
            save_dict['log_alpha'] = self.log_alpha 
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        if self.reward_normalizer:
            save_dict['reward_normalizer_state_dict'] = self.reward_normalizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path): print(f"Warn: SAC model file not found: {path}. Skip loading."); return
        print(f"Loading SAC model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if self.reward_normalizer and 'reward_normalizer_state_dict' in checkpoint:
            self.reward_normalizer.load_state_dict(checkpoint['reward_normalizer_state_dict'])
            print("Loaded SAC reward normalizer statistics.")
        elif self.reward_normalizer:
            print("Warning: SAC reward normalizer statistics not found in checkpoint (reward_normalizer is active).")

        if self.auto_tune_alpha and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha'].to(self.device)
            if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr) # Re-init optimizer
            try: self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except Exception as e: print(f"Warn: Could not load SAC alpha opt state: {e}. Reinit.")
            self.alpha = self.log_alpha.exp().item()
        elif not self.auto_tune_alpha and 'log_alpha' in checkpoint: # Fixed alpha from checkpoint
            try: self.log_alpha = checkpoint['log_alpha'].to(self.device); self.alpha = self.log_alpha.exp().item()
            except: print("Warn: Could not load fixed log_alpha from checkpoint. Using config alpha.")
        
        self.critic_target.load_state_dict(self.critic.state_dict()) 
        for p in self.critic_target.parameters(): p.requires_grad = False
        self.actor.train(); self.critic.train(); self.critic_target.train() 
        print(f"SAC model loaded successfully from {path}")


# --- Training Loop (train_sac) ---
def train_sac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True,
              models_save_path: str = None, tensorboard_log_path: str = None):
    sac_config = config.sac
    train_config = config.training
    buffer_config = config.replay_buffer
    world_config = config.world # This now contains normalization settings
    cuda_device = config.cuda_device

    tb_log_path = tensorboard_log_path or os.path.join("runs", f"sac_fixednorm_rnn_{config.sac.use_rnn}_{int(time.time())}")
    model_path_base = models_save_path or train_config.models_dir 

    os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logs: {tb_log_path}")

    if torch.cuda.is_available():
        device = torch.device(cuda_device)
        if 'cuda' in cuda_device:
            try: torch.cuda.set_device(device); print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e: print(f"Warn: CUDA device {cuda_device} failed. E: {e}"); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else: device = torch.device("cpu"); print("GPU not available, using CPU.")

    agent = SAC(config=sac_config, world_config=world_config, training_config=train_config, device=device)
    memory = ReplayBuffer(config=buffer_config, world_config=world_config)
    os.makedirs(model_path_base, exist_ok=True)

    latest_model_file = None # Logic to find latest model
    if os.path.exists(model_path_base) and os.path.isdir(model_path_base):
        model_files = [f for f in os.listdir(model_path_base) if f.startswith("sac_") and f.endswith(".pt")]
        if model_files:
            try: latest_model_file = max([os.path.join(model_path_base, f) for f in model_files], key=os.path.getmtime)
            except Exception as e: print(f"Could not find latest model in {model_path_base}: {e}")

    total_steps, start_episode = 0, 1
    if latest_model_file and os.path.exists(latest_model_file):
        print(f"\nResuming SAC training from: {latest_model_file}")
        agent.load_model(latest_model_file)
        try: # Parse episode and steps from filename
            parts = os.path.basename(latest_model_file).split('_')
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            step_part = next((p for p in parts if p.startswith('step')), None)
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            if step_part: total_steps = int(step_part.replace('step', '').split('.')[0])
            print(f"Resuming at step {total_steps}, episode {start_episode}")
        except: print(f"Warn: Could not parse steps/ep from {latest_model_file}. Starting new counts."); total_steps=0; start_episode=1
    else: print(f"\nStarting SAC training from scratch. Models saved in {model_path_base}.")

    episode_rewards, all_losses = [], {'critic_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
    timing_metrics = { 'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100) }
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1), desc="Training SAC", unit="episode", initial=start_episode-1, total=train_config.num_episodes)
    world = World(world_config=world_config)

    for episode in pbar:
        world.reset()
        state = world.encode_state() # state dict contains normalized features if world_config.normalize_state
        ep_reward_raw, ep_steps = 0, 0
        actor_hidden = agent.actor.get_initial_hidden_state(1, device) if agent.use_rnn else None
        ep_losses_temp = {k: [] for k in all_losses}
        updates_made_this_ep = 0

        for step_in_episode in range(train_config.max_steps):
            action_norm, next_actor_hidden = agent.select_action(state, actor_hidden_state=actor_hidden, evaluate=False)
            
            step_time_start = time.time()
            world.step(action_norm, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            timing_metrics['env_step_time'].append(time.time() - step_time_start)

            raw_reward_this_step = world.reward # World provides raw reward
            next_state = world.encode_state() # Contains normalized features
            done = world.done
            
            # Replay buffer stores (current_normalized_trajectory, action_taken, raw_reward, next_normalized_trajectory, done)
            memory.push(state, action_norm, raw_reward_this_step, next_state, done)

            state = next_state
            if agent.use_rnn: actor_hidden = next_actor_hidden
            ep_reward_raw += raw_reward_this_step
            ep_steps += 1; total_steps += 1

            if total_steps >= train_config.learning_starts and total_steps % train_config.train_freq == 0:
                for _ in range(train_config.gradient_steps):
                    if len(memory) >= train_config.batch_size:
                        update_time_start = time.time()
                        losses = agent.update_parameters(memory, train_config.batch_size) 
                        timing_metrics['parameter_update_time'].append(time.time() - update_time_start)
                        if losses and not any(np.isnan(v) for v in losses.values() if isinstance(v, (float, np.float64))):
                            for k, v in losses.items(): 
                                if isinstance(v, (float, np.float64)): ep_losses_temp[k].append(v)
                            updates_made_this_ep +=1
                        else: break # Skip if no losses or NaN
                    else: break # Not enough samples in memory
            if done: break
        
        episode_rewards.append(ep_reward_raw)
        avg_losses = {k: np.mean(v) if v else 0 for k,v in ep_losses_temp.items()}
        if updates_made_this_ep > 0:
            for k,v_list in all_losses.items(): 
                if not np.isnan(avg_losses[k]): v_list.append(avg_losses[k])
        
        if episode % train_config.log_frequency == 0:
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Env_Step_ms_Avg100', np.mean(timing_metrics['env_step_time'])*1000, total_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time'])*1000, total_steps)
            
            writer.add_scalar('Reward/Episode_Raw', ep_reward_raw, total_steps)
            writer.add_scalar('Steps/Episode', ep_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Buffer/Size', len(memory), total_steps)
            writer.add_scalar('Error/Distance_EndEp', world.error_dist, total_steps)

            if updates_made_this_ep > 0:
                if not np.isnan(avg_losses['critic_loss']): writer.add_scalar('Loss/Critic_AvgEp', avg_losses['critic_loss'], total_steps)
                if not np.isnan(avg_losses['actor_loss']): writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                if not np.isnan(avg_losses['alpha']): writer.add_scalar('Alpha/Value_AvgEp', avg_losses['alpha'], total_steps)
                if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): writer.add_scalar('Loss/Alpha_AvgEp', avg_losses['alpha_loss'], total_steps)
            else: writer.add_scalar('Alpha/Value_AvgEp', agent.alpha, total_steps)
            
            if agent.reward_normalizer and agent.reward_normalizer.count > agent.reward_normalizer.epsilon:
                writer.add_scalar('Stats/SAC_RewardNorm_Mean', agent.reward_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/SAC_RewardNorm_Std', torch.sqrt(agent.reward_normalizer.var[0].clamp(min=0.0)).item(), total_steps)

            avg_rew100 = np.mean(episode_rewards[-min(100, len(episode_rewards)):]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100_Raw', avg_rew100, total_steps)
            pbar.set_postfix({'avg_rew10': f'{np.mean(episode_rewards[-min(10,len(episode_rewards)):]) if episode_rewards else 0:.2f}', 'steps': total_steps, 'alpha': f"{agent.alpha:.3f}"})

        if episode % train_config.save_interval == 0:
            agent.save_model(os.path.join(model_path_base, f"sac_ep{episode}_step{total_steps}.pt"))

    pbar.close(); writer.close()
    print(f"SAC Training finished. Total steps: {total_steps}")
    agent.save_model(os.path.join(model_path_base, f"sac_final_ep{train_config.num_episodes}_step{total_steps}.pt"))
    if run_evaluation: print("\nStarting evaluation..."); evaluate_sac(agent, config)
    return agent, episode_rewards


# --- Evaluation Loop (evaluate_sac) ---
def evaluate_sac(agent: SAC, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world # Contains normalization settings
    vis_config = config.visualization

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
    agent.actor.eval() # No critic eval needed for action selection
    if agent.reward_normalizer: agent.reward_normalizer.eval() # Eval mode for reward norm if used during eval (usually not)

    print(f"\nRunning SAC Evaluation for {eval_config.num_episodes} episodes...")
    world = World(world_config=world_config)

    for episode in range(eval_config.num_episodes):
        world.reset()
        state = world.encode_state() # Normalized state from world
        ep_reward_raw, episode_frames = 0, []
        actor_hidden = agent.actor.get_initial_hidden_state(1, agent.device) if agent.use_rnn else None

        if eval_config.render and vis_available:
            current_vis_config = vis_config.model_copy(); current_vis_config.save_dir = vis_save_dir_actual
            reset_trajectories()
            try:
                fname = f"sac_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame = visualize_world(world, current_vis_config, fname, True)
                if initial_frame and os.path.exists(initial_frame): episode_frames.append(initial_frame)
            except Exception as e: print(f"Warn: Vis failed init ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            action_norm, next_actor_hidden = agent.select_action(state, actor_hidden, evaluate=True)
            world.step(action_norm, training=False, terminal_step=(step == eval_config.max_steps - 1))
            raw_reward = world.reward # Raw reward from world
            next_state = world.encode_state() # Normalized state
            done = world.done

            if eval_config.render and vis_available:
                current_vis_config = vis_config.model_copy(); current_vis_config.save_dir = vis_save_dir_actual
                try:
                    fname = f"sac_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, current_vis_config, fname, True)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")
            
            state = next_state
            if agent.use_rnn: actor_hidden = next_actor_hidden
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
            gif_fname = f"sac_eval_episode_{episode+1}.gif"
            try:
                gif_p = save_gif(gif_fname, current_vis_config, episode_frames, current_vis_config.delete_frames_after_gif)
                if gif_p: all_gif_paths.append(gif_p)
            except Exception as e: print(f"Warn: Failed GIF save ep {episode+1}. E: {e}")

    agent.actor.train()
    if agent.reward_normalizer: agent.reward_normalizer.train()

    avg_eval_rew = np.mean(eval_rewards_raw) if eval_rewards_raw else 0
    std_eval_rew = np.std(eval_rewards_raw) if eval_rewards_raw else 0
    success_rt = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0
    print("\n--- SAC Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}, Avg Raw Reward: {avg_eval_rew:.2f} +/- {std_eval_rew:.2f}")
    print(f"Success Rate (Err <= {world_config.success_threshold}): {success_rt:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_save_dir_actual)}'")
    print("--- End SAC Evaluation ---\n")
    return eval_rewards_raw, success_rt