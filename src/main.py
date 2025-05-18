import argparse
import os
import torch
import time
import json

# Import training functions for available algorithms
from SAC import train_sac, evaluate_sac
from PPO import train_ppo, evaluate_ppo

from configs import CONFIGS, DefaultConfig

# Note: Visualization imports are now conditional within evaluate_* functions

def main(config_name: str, cuda_device: str = None, algorithm: str = None, run_evaluation: bool = True):
    """Main function to train and optionally evaluate the RL agent."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name].model_copy(deep=True) # Use a copy to ensure original CONFIGS is not modified
    print(f"Using configuration: '{config_name}'")

    if cuda_device:
        config.cuda_device = cuda_device
        print(f"Overriding CUDA device: {cuda_device}")

    effective_algorithm = algorithm if algorithm else config.algorithm
    if algorithm and algorithm != config.algorithm:
         print(f"Overriding config algorithm '{config.algorithm}' with command line argument: '{algorithm}'")
         config.algorithm = algorithm # Update the config instance
    elif not algorithm and config_name == "default":
         print(f"Using algorithm specified in default config: '{config.algorithm}'")
    else:
         print(f"Using algorithm: '{effective_algorithm}'")
    
    # This ensures the specific algo config's models_dir is set if it was defined that way
    config._update_models_dir_per_algo()


    # --- Experiment Directory Setup ---
    timestamp_ms = int(time.time() * 1000)
    exp_name = f"{effective_algorithm}_{config_name}_exp_{timestamp_ms}"
    # Ensure experiment_base_dir is accessible from config.training
    experiment_path = os.path.join(config.training.experiment_base_dir, exp_name)
    
    models_save_path = os.path.join(experiment_path, "models")
    tensorboard_log_path = os.path.join(experiment_path, "tensorboard")
    config_dump_path = os.path.join(experiment_path, "config.json")

    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(models_save_path, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)
    print(f"Experiment data will be saved in: {os.path.abspath(experiment_path)}")

    # Save the effective configuration
    try:
        with open(config_dump_path, 'w') as f:
            json.dump(config.model_dump(), f, indent=4)
        print(f"Effective configuration saved to: {config_dump_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")
    # --- End Experiment Directory Setup ---

    use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_multi_gpu:
         print(f"Detected {torch.cuda.device_count()} GPUs.")
         # Note: Actual multi-GPU utilization depends on algorithm implementation

    # model_dir = config.training.models_dir # This is now the *default* models_dir, not experiment specific
    # print(f"Default models directory (for reference/older scripts): {os.path.abspath(model_dir)}")


    agent = None
    episode_rewards = []
    # final_model_path = None # Model saving is handled within train_* functions

    # --- Training Phase ---
    if effective_algorithm.lower() == "ppo":
        print("Training PPO agent...")
        agent, episode_rewards = train_ppo(
            config=config, 
            use_multi_gpu=use_multi_gpu, 
            run_evaluation=False,
            models_save_path=models_save_path,
            tensorboard_log_path=tensorboard_log_path
            )
        # Model saving handled by train_ppo

    elif effective_algorithm.lower() == "sac":
        print("Training SAC agent...")
        agent, episode_rewards = train_sac(
            config=config, 
            use_multi_gpu=use_multi_gpu, 
            run_evaluation=False,
            models_save_path=models_save_path,
            tensorboard_log_path=tensorboard_log_path
            )
    else:
        raise ValueError(f"Unknown algorithm specified: {effective_algorithm}. Choose 'sac' or 'ppo'.")

    # --- Evaluation Phase (Optional) ---
    if run_evaluation and agent is not None:
        print(f"\nEvaluating {effective_algorithm.upper()} agent...")
        if effective_algorithm.lower() == "ppo":
            evaluate_ppo(agent=agent, config=config)
        elif effective_algorithm.lower() == "sac":
            evaluate_sac(agent=agent, config=config)
    elif not run_evaluation:
         print("\nSkipping evaluation phase.")

    print(f"\nTraining {'and evaluation ' if run_evaluation else ''}complete.")
    if run_evaluation and config.evaluation.render:
         # Visualization save_dir might be relative to experiment path or absolute
         # For now, assume vis_config.save_dir is a simple name like "world_snapshots"
         # and will be created inside the experiment_path by visualization.py if not absolute.
         # This part might need refinement if vis_config.save_dir is intended to be global.
         # Let's assume visualize_world prepends experiment_path if vis_config.save_dir is relative.
         vis_output_dir = os.path.join(experiment_path, config.visualization.save_dir)
         if not os.path.isabs(config.visualization.save_dir):
            print(f"Find potential visualizations in the '{os.path.abspath(vis_output_dir)}' directory.")
         else:
            print(f"Find potential visualizations in the '{os.path.abspath(config.visualization.save_dir)}' directory.")


if __name__ == "__main__":
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
        choices=["sac", "ppo"],
        help="RL algorithm to use ('sac' or 'ppo'). Overrides config."
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
    main(config_name=args.config, cuda_device=args.device, algorithm=args.algorithm, run_evaluation=args.evaluate)