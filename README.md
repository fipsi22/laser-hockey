# Laser Hockey: Rainbow-DQN

A Rainbow-DQN implementation for the [hockey-env](https://github.com/martius-lab/hockey-env) Gymnasium environment. This project demonstrates a modular reinforcement learning setup using Rainbow-DQN, including prioritized replay, multi-step learning, dueling networks, and noisy nets.
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)  

## Features

- Rainbow-DQN agent with:
  - Dueling Q-Network
  - Double Q-Learning (DDQN)
  - Noisy Nets for exploration
  - Prioritized Experience Replay
  - Multi-step returns
  - Distributional Q-values (QR-DQN)
  - Geometric Data Augmentation
  - Symmetry-Aware Regularization
- Training scripts with configurable hyperparameters
- Evaluation scripts and plotting utilities

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/fipsi22/laser-hockey.git
cd laser-hockey
pip install -r requirements.txt
```

## Quick Start

### Training
Execute the following command to initiate the training loop. All model configurations, environment settings, and hyperparameters are consolidated within the training script for ease of modification:
```bash
python3 /laser-hockey/rainbow/rainbow_train.py
```
### Evaluation
To assess the performance of a saved agent against the scripted baselines, use the evaluation utility:
```bash
python3 /laser-hockey/rainbow/util/run_evaluation.py
```
For a head-to-head evaluation between multiple trained agents or different model checkpoints, execute the tournament script:
```bash
python3 /laser-hockey/rainbow/util/head_to_head_tournament.py
```


