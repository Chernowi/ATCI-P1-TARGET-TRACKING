import argparse
import os
import torch
import sys
import time
import json

# Prevent specific visualization imports (though they are already conditional)
sys.modules['matplotlib'] = None
sys.modules['matplotlib.pyplot'] = None
sys.modules['imageio'] = None
sys.modules['imageio.v2'] = None
sys.modules['PIL'] = None

# Import necessary modules AFTER potentially blocking visualization libs
from SAC import train_sac, evaluate_sac
from PPO import train_ppo, evaluate_ppo
from configs import CONFIGS, DefaultConfig


def bsc_main(config_name: str, cuda_device: str = None, algorithm: str = None, run_evaluation: bool = True):
    """
    Main function for basic execution (no visualization imports guaranteed).
    Trains and optionally evaluates the RL agent.
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name].model_copy(deep=True) # Use a copy to avoid modifying global CONFIGS
    print(f"Using configuration: '{config_name}' (Basic Mode - No Viz Imports)")

    # --- Force disable rendering in config ---
    original_render_setting = config.evaluation.render
    config.evaluation.render = False
    if original_render_setting:
        print("Rendering automatically disabled in basic mode.")

    if cuda_device:
        config.cuda_device = cuda_device
        print(f"Overriding CUDA device: {cuda_device}")

    effective_algorithm = algorithm if algorithm else config.algorithm
    if algorithm and algorithm != config.algorithm:
         print(f"Overriding config algorithm '{config.algorithm}' with command line argument: '{algorithm}'")
         config.algorithm = algorithm # Update the config instance
    else:
         print(f"Using algorithm: '{effective_algorithm}'")

    # --- Experiment Directory Setup ---
    timestamp_ms = int(time.time() * 1000)
    exp_name = f"{effective_algorithm}_{config_name}_exp_{timestamp_ms}"
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

    # model_dir = config.training.models_dir # No longer used directly here for saving
    # os.makedirs(model_dir, exist_ok=True)
    # print(f"Models will be saved in: {os.path.abspath(model_dir)}") # Updated by experiment_path

    agent = None
    episode_rewards = []
    final_model_path = None # Will be constructed inside train_*

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
        # train_ppo saves the final model itself.
        # final_model_path = os.path.join(models_save_path, f"ppo_final_ep{config.training.num_episodes}.pt")
        # agent.save_model(final_model_path) # Agent saves its own model now
        # print(f"Final PPO model saved to {final_model_path}")

    elif effective_algorithm.lower() == "sac":
        print("Training SAC agent...")
        agent, episode_rewards = train_sac(
            config=config, 
            use_multi_gpu=use_multi_gpu, 
            run_evaluation=False,
            models_save_path=models_save_path,
            tensorboard_log_path=tensorboard_log_path
        )
        # final_model_path = os.path.join(models_save_path, f"sac_final_ep{config.training.num_episodes}.pt")
        # agent.save_model(final_model_path)
        # print(f"Final SAC model saved to {final_model_path}")

    else:
        raise ValueError(f"Unknown algorithm specified: {effective_algorithm}. Choose 'sac' or 'ppo'.")

    # --- Evaluation Phase (Optional) ---
    # The evaluate_* functions already handle the conditional import/disabling of viz
    if run_evaluation and agent is not None:
        print(f"\nEvaluating {effective_algorithm.upper()} agent (basic mode)...")
        if effective_algorithm.lower() == "ppo":
            evaluate_ppo(agent=agent, config=config)
        elif effective_algorithm.lower() == "sac":
            evaluate_sac(agent=agent, config=config)
    elif not run_evaluation:
         print("\nSkipping evaluation phase.")

    print(f"\nBasic Mode Training {'and evaluation ' if run_evaluation else ''}complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate RL agent (Basic Mode - No Visualization Imports).")
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
    bsc_main(config_name=args.config, cuda_device=args.device, algorithm=args.algorithm, run_evaluation=args.evaluate)