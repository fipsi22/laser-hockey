# Laser Hockey: Rainbow-DQN

A Rainbow-DQN implementation for the [hockey-env](https://github.com/martius-lab/hockey-env) Gymnasium environment. This project demonstrates a modular reinforcement learning setup using Rainbow-DQN, including prioritized replay, multi-step learning, dueling networks, and noisy nets.
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)  

## Features

- Rainbow-DQN agent with:
  - Dueling Q-Network
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
