# Reinforcement Learning for Range-Only Underwater Target Tracking

This project implements Reinforcement Learning (RL) agents (SAC and PPO) for the task of range-only underwater target localization and tracking using an Autonomous Underwater Vehicle (AUV). The agent learns to navigate the AUV to effectively estimate the position of a potentially moving target using only range (distance) measurements.

This work is inspired by and builds upon the research and environment presented in:

*   **Original GitHub Repository:** [RLforUTracking by imasmitja](https://github.com/imasmitja/RLforUTracking)
*   **Associated Research Paper:**
    Masmitja, I., Martin, M., Katija,K., Gomariz, S., & Navarro, J. (2023). A reinforcement learning path planning approach for range-only underwater target localization with autonomous vehicles. *arXiv preprint arXiv:2301.06863*.

The core idea is to train an RL agent to control the AUV's yaw angle, guiding its movement to gather informative range measurements. These measurements are then fed into an estimator (Particle Filter or Least Squares) to predict the target's location. The agent's reward function encourages efficient estimation.

## Project Structure

```
.
├── configs.py            # All configuration classes (Pydantic models)
├── PPO.py                # PPO agent implementation and training/evaluation
├── SAC.py                # SAC agent implementation and training/evaluation
├── world.py              # Simulation environment
├── world_objects.py      # Defines Location, Velocity, and Object classes
├── particle_filter.py    # Particle Filter estimator
├── least_squares.py      # Least Squares estimator
├── visualization.py      # Plotting and GIF generation utilities
├── utils.py              # Utility functions (e.g., RunningMeanStd)
├── main.py               # Main script to run training and evaluation
├── run_ppo_experiment.py # Script to evaluate a trained PPO model
├── run_sac_experiment.py # Script to evaluate a trained SAC model
├── run_manual_policy.py  # Script to run a hardcoded manual policy
├── requirements.txt      # Python package dependencies
├── README.md             # This file
└── hyperparameter_guide.md # Guide for tuning parameters in configs.py
└── models/               # Default directory for saved model checkpoints (organized by algorithm)
└── experiments/          # Default base directory for experiment logs, models, and configs
```

## Key Features

*   **Two RL Algorithms:**
    *   Soft Actor-Critic (SAC)
    *   Proximal Policy Optimization (PPO)
*   **Flexible State Representation:**
    *   Supports using the last N steps of (state, action, reward) as input (trajectory state).
    *   Optionally uses Recurrent Neural Networks (LSTM/GRU) within SAC and PPO to process state trajectories.
*   **Target Estimators:**
    *   Particle Filter (PF)
    *   Linearized Least Squares (LS)
*   **Comprehensive Configuration:**
    *   Uses Pydantic for typed and validated configuration management (`configs.py`).
    *   Pre-defined configurations for different scenarios (e.g., signal quality, RNN/MLP, PER).
    *   Easy creation of hyperparameter variations.
*   **Prioritized Experience Replay (PER):** Optional for SAC.
*   **Reward Normalization:** Optional, using a running mean and standard deviation.
*   **Experiment Management:**
    *   Saves models, TensorBoard logs, and effective configurations for each run in a unique `experiments/` subdirectory.
    *   Supports resuming training from saved checkpoints.
*   **Visualization:**
    *   Generates 2D top-down views of the simulation.
    *   Saves sequences of frames as GIFs for qualitative analysis.
*   **Evaluation Scripts:** Separate scripts to load and evaluate trained models.
*   **Manual Policy:** A baseline script to run a hardcoded approach-and-circle policy.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install PyTorch, NumPy, Matplotlib, Pydantic, TensorBoard, tqdm, and imageio.
    *Note: Ensure you install the correct PyTorch version for your CUDA setup if you plan to use a GPU. Visit [pytorch.org](https://pytorch.org/) for specific installation commands.*

## Usage

### Training a New Agent

The `main.py` script is used for training.

```bash
python main.py --config <config_name> --algorithm <sac|ppo> --device <cuda_device|cpu>
```

**Arguments:**

*   `--config <config_name>` or `-c <config_name>`:
    *   Specifies the configuration profile from `configs.py` to use.
    *   Examples: `default`, `sac_rnn`, `ppo_mlp`, `default_poor_signal`, `sac_mlp_actor_lr_low`.
    *   Default: `default` (which is SAC MLP).
    *   Run `python main.py --help` to see a list of available built-in config names.
*   `--algorithm <sac|ppo>` or `-a <sac|ppo>`:
    *   Overrides the algorithm specified in the chosen config.
    *   If not provided, the algorithm from the config is used.
*   `--device <cuda_device|cpu>` or `-d <cuda_device|cpu>`:
    *   Specifies the device to use (e.g., `cuda:0`, `cpu`).
    *   Overrides the `cuda_device` in the chosen config.
*   `--evaluate` / `--no-evaluate`:
    *   Whether to run evaluation after training (default is to evaluate).

**Example Training Commands:**

*   Train a default SAC MLP agent:
    ```bash
    python main.py -c default
    ```
*   Train a PPO agent with RNN using the `ppo_rnn` config on `cuda:0`:
    ```bash
    python main.py -c ppo_rnn -d cuda:0
    ```
*   Train a SAC agent with PER, overriding the algorithm from `default` (which is already SAC but shows how override works):
    ```bash
    python main.py -c sac_per -a sac
    ```

Training progress will be logged to TensorBoard, and models/configs will be saved in a timestamped subdirectory under `experiments/`.

### Evaluating a Trained Agent

Use `run_sac_experiment.py` or `run_ppo_experiment.py`.

**Arguments for `run_sac_experiment.py` / `run_ppo_experiment.py`:**

*   `--model <path_to_model_or_experiment_dir>` or `-m <path>`: (Required)
    *   Path to the specific `.pt` model file.
    *   OR path to the experiment directory (e.g., `experiments/sac_default_exp_167.../`). The script will try to load the `config.json` from this directory and the latest/final model from its `models/` subdirectory.
*   `--config <config_name>` or `-c <config_name>`:
    *   Fallback configuration name if `config.json` is not found in the experiment directory.
    *   Default: `default`.
*   `--episodes <num>` or `-e <num>`: Number of evaluation episodes (overrides config).
*   `--steps <num>` or `-s <num>`: Max steps per evaluation episode (overrides config).
*   `--render` / `--no-render`: Enable/disable rendering (overrides config).

**Example Evaluation Commands:**

*   Evaluate a SAC model located in an experiment directory:
    ```bash
    python run_sac_experiment.py -m experiments/sac_default_exp_xxxxxxxxxxxxx
    ```
*   Evaluate a specific PPO model file, rendering the output:
    ```bash
    python run_ppo_experiment.py -m experiments/ppo_ppo_rnn_exp_yyyyyyyyyyyyy/models/ppo_final_ep30000_stepenv9000000.pt --render
    ```

Visualizations (if rendering is enabled) will be saved in the `visualization.save_dir` (default: `world_snapshots`) within the loaded model's experiment directory.

## Hyperparameter Tuning

A detailed guide on the available parameters in `configs.py` and their expected impact on training can be found in `hyperparameter_guide.md`. This is crucial for achieving good performance.

## Configuration System (`configs.py`)

*   **Pydantic Models:** Configurations are defined using Pydantic, providing data validation and type hints.
*   **`DefaultConfig`:** A top-level model that aggregates configurations for SAC, PPO, training, world, estimators, etc.
*   **Pre-defined Variants:** The `CONFIGS` dictionary at the end of `configs.py` provides several pre-defined configurations for quick use and experimentation.
*   **Hyperparameter Variations:** The script also programmatically generates variations of SAC MLP and PPO MLP configurations by modifying specific learning rates, hidden dimensions, etc., and adds them to `CONFIGS`.
*   **Experiment-Specific Configs:** When training starts, the *effective* configuration (including any command-line overrides) is saved as `config.json` in the experiment directory. This ensures reproducibility.

