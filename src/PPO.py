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

# Local imports
from world import World
from configs import DefaultConfig, PPOConfig, TrainingConfig, WorldConfig, CORE_STATE_DIM
from torch.utils.tensorboard import SummaryWriter
import math
from utils import RunningMeanStd 

class PPOMemory:
    """Memory buffer for PPO. Stores normalized basic states and raw rewards."""
    def __init__(self, batch_size=64):
        self.states_norm = [] # List of (already normalized by World) basic_state tuples
        self.actions = [] 
        self.probs = []   
        self.vals = []    
        self.rewards_raw = [] # List of raw rewards
        self.dones = []   
        self.batch_size = batch_size

    def store(self, normalized_basic_state, action, probs, vals, reward_raw, done):
        """Store a transition. `normalized_basic_state` is from World. `reward_raw` is raw."""
        if not isinstance(normalized_basic_state, (tuple, list)):
            print(f"Warning: PPO Memory storing non-tuple state: {type(normalized_basic_state)}")
        # Basic check for NaNs in normalized state (should be rare if world normalization is robust)
        if any(np.isnan(x) for x in normalized_basic_state if isinstance(x, (float, np.float32, np.float64))):
            return 

        self.states_norm.append(normalized_basic_state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards_raw.append(reward_raw) # Store raw reward
        self.dones.append(done)

    def clear(self):
        self.states_norm, self.actions, self.probs, self.vals, self.rewards_raw, self.dones = [], [], [], [], [], []

    def generate_batches(self):
        n_transitions = len(self.states_norm)
        batch_start = np.arange(0, n_transitions, self.batch_size)
        indices = np.arange(n_transitions, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        try:
            # states_norm are already normalized tuples/lists from World
            states_norm_arr = np.array(self.states_norm, dtype=np.float32) 
            if np.isnan(states_norm_arr).any():
                 print("Error: NaN found in PPO memory states_norm_arr!")
                 return None, None, None, None, None, None, None 

            actions_arr = np.array(self.actions, dtype=np.float32).reshape(-1, 1)
            probs_arr = np.array(self.probs, dtype=np.float32).reshape(-1, 1)
            vals_arr = np.array(self.vals, dtype=np.float32).reshape(-1, 1)
            rewards_raw_arr = np.array(self.rewards_raw, dtype=np.float32) # Raw rewards
            dones_arr = np.array(self.dones, dtype=np.float32)

        except Exception as e:
             print(f"Error converting PPO memory to arrays: {e}")
             return None, None, None, None, None, None, None

        return states_norm_arr, actions_arr, probs_arr, vals_arr, rewards_raw_arr, dones_arr, batches

    def __len__(self):
        return len(self.states_norm)


class PolicyNetwork(nn.Module):
    """Actor network for PPO. Takes pre-normalized basic_state as input."""
    def __init__(self, config: PPOConfig):
        super(PolicyNetwork, self).__init__()
        self.state_dim = config.state_dim 
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, normalized_basic_state): 
        x = F.relu(self.fc1(normalized_basic_state))
        x = F.relu(self.fc2(x))
        action_mean = self.mean(x)
        action_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

    def sample(self, normalized_basic_state):
        mean, std = self.forward(normalized_basic_state)
        distribution = Normal(mean, std)
        x_t = distribution.sample()
        action_normalized = torch.tanh(x_t) 
        log_prob_unbounded = distribution.log_prob(x_t)
        clamped_tanh = action_normalized.clamp(-0.999999, 0.999999)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)
        return action_normalized, log_prob

    def evaluate(self, normalized_basic_state, action_normalized):
        mean, std = self.forward(normalized_basic_state)
        distribution = Normal(mean, std)
        action_tanh = torch.clamp(action_normalized, -0.99999, 0.99999) 
        action_original_space = torch.atanh(action_tanh) 
        log_prob_unbounded = distribution.log_prob(action_original_space)
        log_det_jacobian = torch.log(1.0 - action_tanh.pow(2) + 1e-7) 
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)
        entropy = distribution.entropy().sum(1, keepdim=True)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network for PPO. Takes pre-normalized basic_state as input."""
    def __init__(self, config: PPOConfig):
        super(ValueNetwork, self).__init__()
        self.state_dim = config.state_dim 
        self.hidden_dim = config.hidden_dim
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, normalized_basic_state): 
        x = F.relu(self.fc1(normalized_basic_state))
        x = F.relu(self.fc2(x))
        return self.value(x)

class PPO:
    """PPO. Assumes states from World are already normalized if world_config.normalize_state=True."""
    def __init__(self, config: PPOConfig, training_config: TrainingConfig, world_config: WorldConfig, device: torch.device = None): # Added world_config
        self.config = config
        self.training_config = training_config 
        self.world_config = world_config # To check world_config.normalize_state
        self.gamma = config.gamma; self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip; self.n_epochs = config.n_epochs
        self.entropy_coef = config.entropy_coef; self.value_coef = config.value_coef
        self.state_dim = config.state_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")
        print(f"PPO Agent expects states to be normalized by World: {self.world_config.normalize_state}")

        if self.training_config.normalize_rewards:
            self.reward_normalizer = RunningMeanStd(shape=(1,)).to(self.device)
            print("PPO Agent reward normalization enabled.")
        else:
            self.reward_normalizer = None

        self.actor = PolicyNetwork(config).to(self.device)
        self.critic = ValueNetwork(config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.memory = PPOMemory(batch_size=config.batch_size)

    def select_action(self, state: dict, evaluate=False):
        """Select action. `state` dict contains pre-normalized 'basic_state'."""
        # state['basic_state'] is already normalized by World
        normalized_basic_state_tuple = state['basic_state'] 
        state_tensor_normalized = torch.FloatTensor(normalized_basic_state_tuple).to(self.device).unsqueeze(0)

        with torch.no_grad():
            self.actor.eval(); self.critic.eval()
            if evaluate:
                action_mean, _ = self.actor.forward(state_tensor_normalized)
                action_normalized = torch.tanh(action_mean)
            else:
                action_normalized, log_prob = self.actor.sample(state_tensor_normalized)
                value = self.critic(state_tensor_normalized)
                # Store the (already normalized) basic state and raw reward in memory
                self.memory.store(
                    normalized_basic_state_tuple, # Store World-normalized state
                    action_normalized.cpu().numpy()[0, 0],
                    log_prob.cpu().numpy()[0, 0],
                    value.cpu().numpy()[0, 0],
                    0, False # Placeholder raw reward, done (will be updated by store_transition)
                )
            self.actor.train(); self.critic.train()
        return action_normalized.detach().cpu().numpy()[0, 0]

    def store_transition(self, raw_reward, done):
        """Store raw reward and done flag for the last transition in memory."""
        if self.memory.rewards_raw and self.memory.rewards_raw[-1] == 0 and self.memory.dones[-1] is False:
            self.memory.rewards_raw[-1] = raw_reward # Store raw reward
            self.memory.dones[-1] = done
        elif len(self.memory.rewards_raw) < len(self.memory.states_norm): # Should be states_norm
             self.memory.rewards_raw.append(raw_reward)
             self.memory.dones.append(done)

    def update_parameters(self):
        if len(self.memory) < self.config.batch_size: return None

        # Memory provides already normalized states (states_norm_arr) and raw rewards (raw_rewards_arr)
        states_norm_arr, actions_arr, old_log_probs_arr, values_arr, raw_rewards_arr, dones_arr, batches = self.memory.generate_batches()
        if states_norm_arr is None: self.memory.clear(); return None

        # Reward Normalization (if enabled)
        if self.reward_normalizer:
            rewards_tensor_raw = torch.from_numpy(raw_rewards_arr).to(self.device).unsqueeze(1)
            self.reward_normalizer.update(rewards_tensor_raw)
            normalized_rewards_tensor = self.reward_normalizer.normalize(rewards_tensor_raw)
            rewards_for_gae = normalized_rewards_tensor.squeeze(1).cpu().numpy()
        else:
            rewards_for_gae = raw_rewards_arr 
        
        returns = self._compute_returns_and_advantages(rewards_for_gae, dones_arr, values_arr.squeeze(), states_norm_arr) # Pass states_norm_arr
        if returns is None: self.memory.clear(); return None

        actor_losses, critic_losses, entropies = [], [], []
        advantages = returns - values_arr.squeeze()
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(self.device)
        if len(advantages_tensor) > 1: advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # States are already normalized
        states_norm_tensor = torch.from_numpy(states_norm_arr).to(self.device)

        for _ in range(self.n_epochs):
            for batch_indices in batches:
                batch_states_normalized = states_norm_tensor[batch_indices] # Already normalized
                batch_actions = torch.tensor(actions_arr[batch_indices], dtype=torch.float32).to(self.device)
                batch_old_log_probs = torch.tensor(old_log_probs_arr[batch_indices], dtype=torch.float32).to(self.device)
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                new_log_probs, entropy = self.actor.evaluate(batch_states_normalized, batch_actions)
                new_values = self.critic(batch_states_normalized) 
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.actor_optimizer.zero_grad(); self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step(); self.critic_optimizer.step()
                actor_losses.append(actor_loss.item()); critic_losses.append(critic_loss.item()); entropies.append(entropy.mean().item())
        self.memory.clear() 
        return {'actor_loss': np.mean(actor_losses), 'critic_loss': np.mean(critic_losses), 'entropy': np.mean(entropies)}

    def _compute_returns_and_advantages(self, rewards_arr, dones_arr, values_arr, states_norm_arr): # Added states_norm_arr
        if len(rewards_arr) == 0: return None 
        returns = np.zeros_like(rewards_arr)
        gae = 0.0
        last_value = 0
        if not dones_arr[-1]:
            # Get the last normalized state from memory (it was stored as tuple)
            last_normalized_basic_state_tuple = self.memory.states_norm[-1]
            last_state_tensor_normalized = torch.FloatTensor(last_normalized_basic_state_tuple).to(self.device).unsqueeze(0)
            with torch.no_grad():
                last_value = self.critic(last_state_tensor_normalized).cpu().numpy()[0, 0]

        for step in reversed(range(len(rewards_arr))):
            next_value = last_value if step == len(rewards_arr) - 1 else values_arr[step + 1]
            delta = rewards_arr[step] + self.gamma * next_value * (1 - int(dones_arr[step])) - values_arr[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones_arr[step])) * gae
            returns[step] = gae + values_arr[step]
        return returns 

    def save_model(self, path: str):
        print(f"Saving PPO model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'device_type': self.device.type
        }
        if self.reward_normalizer: # Only save reward_normalizer if it exists
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
    world_config = config.world # Pass world_config to PPO agent
    cuda_device = config.cuda_device

    tb_log_path = tensorboard_log_path or os.path.join("runs", f"ppo_fixednorm_{int(time.time())}")
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
            except: pass # Keep latest_model_file as None

    total_steps, start_episode = 0, 1
    if latest_model_file and os.path.exists(latest_model_file):
        print(f"\nResuming PPO training from: {latest_model_file}")
        agent.load_model(latest_model_file)
        try:
            parts = os.path.basename(latest_model_file).split('_')
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            step_part = next((p for p in parts if p.startswith('step')), None)
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            if step_part: total_steps = int(step_part.replace('step', '').split('.')[0])
            print(f"Resuming at step {total_steps}, episode {start_episode}")
        except: print(f"Warn: Could not parse from {latest_model_file}. New counts."); total_steps=0;start_episode=1
    else: print(f"\nStarting PPO training from scratch. Models saved in {model_path_base}")

    episode_rewards, timing_metrics = [], {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1), desc="Training PPO", unit="episode", initial=start_episode-1, total=train_config.num_episodes)
    world = World(world_config=world_config)
    learn_steps = 0 

    for episode in pbar:
        world.reset()
        state = world.encode_state() # World provides normalized state if configured
        ep_reward_raw, ep_steps = 0, 0

        for step_in_episode in range(train_config.max_steps):
            action_normalized = agent.select_action(state, evaluate=False) 
            step_start_time = time.time()
            world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            timing_metrics['env_step_time'].append(time.time() - step_start_time)

            raw_reward_this_step = world.reward # World provides raw reward
            next_state = world.encode_state() # Normalized state from world
            done = world.done
            agent.store_transition(raw_reward_this_step, done) # Store raw reward

            state = next_state
            ep_reward_raw += raw_reward_this_step
            ep_steps += 1; total_steps += 1; learn_steps += 1

            if learn_steps >= ppo_config.steps_per_update:
                update_start_time = time.time()
                losses = agent.update_parameters() 
                if losses: timing_metrics['parameter_update_time'].append(time.time() - update_start_time)
                if losses:
                    writer.add_scalar('Loss/Actor', losses['actor_loss'], total_steps)
                    writer.add_scalar('Loss/Critic', losses['critic_loss'], total_steps)
                    writer.add_scalar('Policy/Entropy', losses['entropy'], total_steps)
                learn_steps = 0 
            if done: break
        
        episode_rewards.append(ep_reward_raw)
        if episode % train_config.log_frequency == 0:
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Env_Step_ms_Avg100', np.mean(timing_metrics['env_step_time'])*1000, total_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time'])*1000, total_steps)
            
            writer.add_scalar('Reward/Episode_Raw', ep_reward_raw, total_steps)
            writer.add_scalar('Steps/Episode', ep_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Error/Distance_EndEp', world.error_dist, total_steps)
            writer.add_scalar('Buffer/PPO_Memory_Size', len(agent.memory), total_steps)

            if agent.reward_normalizer and agent.reward_normalizer.count > agent.reward_normalizer.epsilon:
                writer.add_scalar('Stats/PPO_RewardNorm_Mean', agent.reward_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/PPO_RewardNorm_Std', torch.sqrt(agent.reward_normalizer.var[0].clamp(min=0.0)).item(), total_steps)
            
            avg_rew100 = np.mean(episode_rewards[-min(100,len(episode_rewards)):]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100_Raw', avg_rew100, total_steps)
            pbar.set_postfix({'avg_rew10_raw': f'{np.mean(episode_rewards[-min(10,len(episode_rewards)):]) if episode_rewards else 0:.2f}', 'steps': total_steps})

        if episode % train_config.save_interval == 0:
            agent.save_model(os.path.join(model_path_base, f"ppo_ep{episode}_step{total_steps}.pt"))

    pbar.close(); writer.close()
    print(f"PPO Training finished. Total steps: {total_steps}")
    agent.save_model(os.path.join(model_path_base, f"ppo_final_ep{train_config.num_episodes}_step{total_steps}.pt"))
    if run_evaluation: print("\nStarting evaluation..."); evaluate_ppo(agent, config)
    return agent, episode_rewards


# --- Evaluation Loop (evaluate_ppo) ---
def evaluate_ppo(agent: PPO, config: DefaultConfig):
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
    agent.actor.eval(); agent.critic.eval() # Set actor/critic to eval
    if agent.reward_normalizer: agent.reward_normalizer.eval() # Eval mode for reward norm

    print(f"\nRunning PPO Evaluation for {eval_config.num_episodes} episodes...")
    world = World(world_config=world_config)

    for episode in range(eval_config.num_episodes):
        world.reset()
        state = world.encode_state() # Normalized state from World
        ep_reward_raw, episode_frames = 0, []

        if eval_config.render and vis_available:
            current_vis_config = vis_config.model_copy(); current_vis_config.save_dir = vis_save_dir_actual
            reset_trajectories()
            try:
                fname = f"ppo_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame = visualize_world(world, current_vis_config, fname, True)
                if initial_frame and os.path.exists(initial_frame): episode_frames.append(initial_frame)
            except Exception as e: print(f"Warn: Vis failed init ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            action_normalized = agent.select_action(state, evaluate=True) 
            world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            raw_reward = world.reward # Raw reward
            next_state = world.encode_state() # Normalized state
            done = world.done

            if eval_config.render and vis_available:
                current_vis_config = vis_config.model_copy(); current_vis_config.save_dir = vis_save_dir_actual
                try:
                    fname = f"ppo_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, current_vis_config, fname, True)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")
            
            state = next_state
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

    agent.actor.train(); agent.critic.train() # Revert to train mode
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