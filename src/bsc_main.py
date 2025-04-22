import argparse
import os
import torch
import multiprocessing as mp # Add import
import time # Add import for timing if needed

# Import training functions for available algorithms
from SAC import train_sac, evaluate_sac
from PPO import train_ppo, evaluate_ppo
from TSAC import train_tsac, evaluate_tsac

from configs import CONFIGS, DefaultConfig

# Import the vectorized environment and utils
try:
    from vec_env import SubprocVecEnv, make_env
    VEC_ENV_AVAILABLE = True
except ImportError:
    print("Warning: vec_env.py not found or cloudpickle not installed. Parallel environment execution disabled.")
    VEC_ENV_AVAILABLE = False
    SubprocVecEnv = None # Define as None if import fails

def main(config_name: str,
         cuda_device: str = None,
         algorithm: str = None,
         run_evaluation: bool = True,
         num_envs: int = 1): # Add num_envs argument
    """Main function to train and optionally evaluate the RL agent."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}'")

    if cuda_device:
        config.cuda_device = cuda_device
        print(f"Overriding CUDA device: {cuda_device}")

    effective_algorithm = algorithm if algorithm else config.algorithm
    if algorithm and algorithm != config.algorithm:
         print(f"Overriding config algorithm '{config.algorithm}' with command line argument: '{algorithm}'")
         config.algorithm = algorithm
    elif not algorithm and config_name == "default":
         print(f"Using algorithm specified in default config: '{config.algorithm}'")
    else:
         print(f"Using algorithm: '{effective_algorithm}'")

    if num_envs > 1 and not VEC_ENV_AVAILABLE:
        print(f"Warning: Requested {num_envs} parallel environments but vec_env/cloudpickle is not available. Running with 1 environment.")
        num_envs = 1
    elif num_envs <= 0:
         print("Warning: --num-envs must be positive. Setting to 1.")
         num_envs = 1

    # Note: Multi-GPU handling might need more sophisticated setup for parallel envs
    # For now, assume single GPU for the agent, multiple CPU cores for envs
    use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1


    model_dir = config.training.models_dir
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved in: {os.path.abspath(model_dir)}")

    agent = None
    episode_rewards = [] # Or completed_episode_rewards for parallel
    final_model_path = None

    # --- Training Phase ---
    if effective_algorithm.lower() == "ppo":
        print("Training PPO agent...")
        if num_envs > 1:
             print("Warning: Parallel environments currently only implemented for SAC. Running PPO with 1 environment.")
        # PPO train function likely doesn't accept num_envs yet
        agent, episode_rewards = train_ppo(config=config, use_multi_gpu=use_multi_gpu, run_evaluation=False)
        # Naming based on total episodes in config
        final_model_path = os.path.join(model_dir, f"ppo_final_ep{config.training.num_episodes}.pt")
        if agent: agent.save_model(final_model_path)
        print(f"Final PPO model saved to {final_model_path}")

    elif effective_algorithm.lower() == "tsac":
        print("Training T-SAC agent...")
        if num_envs > 1:
             print("Warning: Parallel environments currently only implemented for SAC. Running T-SAC with 1 environment.")
        # T-SAC train function likely doesn't accept num_envs yet
        agent, episode_rewards = train_tsac(config=config, use_multi_gpu=use_multi_gpu, run_evaluation=False)
        # Naming based on total episodes in config
        final_model_path = os.path.join(model_dir, f"tsac_final_ep{config.training.num_episodes}.pt")
        if agent: agent.save_model(final_model_path)
        print(f"Final T-SAC model saved to {final_model_path}")

    elif effective_algorithm.lower() == "sac":
        print(f"Training SAC agent with {num_envs} environment(s)...")
        # Pass num_envs to train_sac
        # train_sac now handles creating VecEnv internally if num_envs > 1
        agent, completed_episode_rewards = train_sac( # Capture completed rewards deque/list
            config=config,
            use_multi_gpu=use_multi_gpu,
            run_evaluation=False,
            num_envs=num_envs
        )
        # Adjust final model saving path based on actual completed episodes (or keep config target?)
        # Using config target for consistency, but actual steps might differ
        final_model_path = os.path.join(model_dir, f"sac_final_comp_ep{config.training.num_episodes}.pt") # Naming convention changed in train_sac
        if agent: agent.save_model(final_model_path)
        print(f"Final SAC model saved to {final_model_path}")
        # Assign episode_rewards for potential later use, maybe average?
        episode_rewards = list(completed_episode_rewards) if completed_episode_rewards else []


    else:
        raise ValueError(f"Unknown algorithm specified: {effective_algorithm}. Choose 'sac', 'ppo', or 'tsac'.")

    # --- Evaluation Phase (Optional) ---
    # Evaluation typically runs on a single environment
    if run_evaluation and agent is not None:
        print(f"\nEvaluating {effective_algorithm.upper()} agent...")
        if effective_algorithm.lower() == "ppo":
            evaluate_ppo(agent=agent, config=config)
        elif effective_algorithm.lower() == "tsac":
            evaluate_tsac(agent=agent, config=config)
        elif effective_algorithm.lower() == "sac":
            evaluate_sac(agent=agent, config=config) # evaluate_sac uses single env
    elif not run_evaluation:
         print("\nSkipping evaluation phase.")

    print(f"\nTraining {'and evaluation ' if run_evaluation else ''}complete.")
    if run_evaluation and config.evaluation.render:
         print(f"Find potential visualizations in the {config.visualization.save_dir} directory.")


if __name__ == "__main__":
    # Set start method for multiprocessing if needed (esp. on Windows/macOS)
    # Needs to be done *before* processes are created
    if os.name == 'nt' or os.name == 'posix': # Check if system supports setting start method
        try:
            # 'spawn' is safer than 'fork' for CUDA and some libraries
            mp.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError as e:
            current_method = mp.get_start_method(allow_none=True)
            if current_method != 'spawn':
                 print(f"Warning: Could not force multiprocessing start method to 'spawn' (currently {current_method}). Error: {e}")
            else:
                 print("Multiprocessing start method already set to 'spawn'.") # Or handle differently
        except AttributeError:
            print("Warning: mp.set_start_method not available. Using default multiprocessing start method.") # Python < 3.4

    parser = argparse.ArgumentParser(description="Train and evaluate RL agent for landmark tracking.")
    parser.add_argument(
        "--config", "-c", type=str, default="default",
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')"
    )
    parser.add_argument(
        "--algorithm", "-a", type=str, default=None,
        choices=["sac", "ppo", "tsac"],
        help="RL algorithm to use ('sac', 'ppo', 'tsac'). Overrides config."
    )
    # Add argument for number of environments
    parser.add_argument(
        "--num-envs", "-n", type=int, default=1,
        help="Number of parallel environments for training (only SAC currently supported)." # Update help text
    )
    # Add option to skip evaluation
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--evaluate", action="store_true", default=True,
        help="Run evaluation after training (default)."
    )
    eval_group.add_argument(
        "--no-evaluate", dest="evaluate", action="store_false",
        help="Skip evaluation after training."
    )

    args = parser.parse_args()
    main(config_name=args.config,
         cuda_device=args.device,
         algorithm=args.algorithm,
         run_evaluation=args.evaluate,
         num_envs=args.num_envs) # Pass num_envs to main
