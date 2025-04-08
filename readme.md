This project implements a reinforcement learning system for target tracking using Soft Actor-Critic (SAC) algorithm integrated with particle filters. The system uses TensorBoard for monitoring training progress and provides visualization capabilities.

## Overview

This project implements an agent that learns to track a moving landmark (target) in a simulated environment. The agent uses:

- **Soft Actor-Critic (SAC)** algorithm for reinforcement learning
- **Particle filters** for state estimation of the target
- **TensorBoard** for monitoring training progress

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Training

To train a new SAC agent:

```bash
python src/main.py --config=default
```

Options:
- `--config`: Choose configuration profile (`default` or `vast`)
- `--n-step`: Use N-Step SAC implementation

### Evaluation

To evaluate a trained model:

```bash
python src/run_experiment.py --model=sac_models/sac_final.pt
```

Options:
- `--model` or `-m`: Path to the trained model
- `--episodes` or `-e`: Number of episodes to run
- `--steps` or `-s`: Maximum steps per episode
- `--render`: Enable visualization
- `--no-render`: Disable visualization
- `--n-step`: Use with N-Step SAC models

### TensorBoard

To launch TensorBoard for monitoring training progress:

```bash
python tensorboard_main.py
```

Options:
- `--port` or `-p`: Port for TensorBoard server (default: 6006)
- `--config` or `-c`: Configuration name (default: `default`)

This will start a TensorBoard server and launch training. Access TensorBoard through your web browser at `http://localhost:6006`.

## Project Structure

- src: Source code
  - SAC.py: Standard SAC implementation
  - nStepSAC.py: N-step returns SAC implementation
  - world.py: Simulation environment
  - particle_filter.py: Particle filter implementation
  - visualization.py: Visualization utilities
  - configs.py: Configuration parameters
  - run_experiment.py: Script for running evaluations
- sac_models: Saved model checkpoints
- runs: TensorBoard logs
- world_snapshots: Visualization outputs

## Configurations

The system offers different configuration profiles in configs.py:
- `default`: Standard settings for local development
- `vast`: High-performance settings for training on more powerful hardware

See hyperparameter_guide.md for detailed explanations of hyperparameters.

## Acknowledgments

The particle filter implementation is based on the work by Ivan Masmitja Rusinol.
