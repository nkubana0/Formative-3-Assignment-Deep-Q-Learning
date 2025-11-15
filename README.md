

## Environment: Assault

**Game Description**: Control a spaceship that moves sideways while a mothership circles overhead deploying enemy drones. Destroy enemies and dodge their attacks to score points.

- **Environment ID**: `AssaultNoFrameskip-v4` / `ALE/Assault-v5`
- **Action Space**: Discrete(7) - [NOOP, FIRE, UP, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
- **Observation Space**: Box(0, 255, (210, 160, 3), uint8) - RGB frames
- **Preprocessing**: 84x84 grayscale, 4 frames stacked
- **Difficulty**: Mode 0, Difficulty 0 (default)

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 2GB free disk space
- Can Ideally have GPU if you want faster training

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/shemankubana/Deep-Q-Learning.git
cd Deep-Q-Learning
```

#### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for macOS users**: If you encounter issues with square brackets, use quotes:
```bash
pip install "gymnasium[atari,accept-rom-license]"
```

#### 4. Verify Installation

```bash
python check_setup.py
```

You should see:
```
✓ SETUP COMPLETE - Ready to train!
```

## Project Structure

```
Deep-Q-Learning/
├── train.py                 # Training script (Task 1)
├── play.py                  # Evaluation script (Task 2)
├── run_experiments.py       # Hyperparameter experiment runner
├── check_setup.py          # Environment verification
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── dqn_model.zip           # Trained model (generated after training)
├── models/                 # Saved models directory
│   ├── best/               # Best performing models
│   ├── checkpoints/        # Training checkpoints
│   └── experiment_XX/      # Individual experiment models
├── logs/                   # Training logs and TensorBoard data
│   ├── dqn_atari/         # Default training logs
│   └── experiment_XX/      # Individual experiment logs
└── videos/                 # Recorded gameplay videos
```

## Usage

### Task 1: Training the Agent

#### Basic Training

Train with default hyperparameters:
```bash
python train.py --timesteps 1000000
```

#### Policy Comparison

Proceed to comparing CNNPolicy and MLPPolicy:

**CNN Policy** (Recommended for Atari - extracts spatial features):
```bash
python train.py --policy CnnPolicy --timesteps 500000
```

**MLP Policy** (For comparison - less effective for image input):
```bash
python train.py --policy MlpPolicy --timesteps 500000
```

#### Custom Hyperparameters

```bash
python train.py \
  --timesteps 500000 \
  --lr 0.0001 \
  --gamma 0.99 \
  --batch-size 32 \
 8

# Experiment 9: Quick Exploitation
python train.py --timesteps 500000 --exp-fraction 0.05 --experiment 9

# Experiment 10: Aggressive Learning
python train.py --timesteps 500000 --lr 1e-3 --batch-size 64 --eps-end 0.02 --exp-fraction 0.15 --experiment 10
```

## Hyperparameter Configurations

| Exp | Name | Learning Rate | Gamma (γ) | Batch Size | ε Start | ε End | Exp Fraction | Expected Behavior |
|-----|------|---------------|-----------|------------|---------|-------|--------------|-------------------|
| 1 | Baseline | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Standard DQN configuration |
| 2 | High LR | 5e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Faster learning, potentially less stable |
| 3 | Low LR | 5e-5 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Slower but more stable learning |
| 4 | High Gamma | 1e-4 | 0.995 | 32 | 1.0 | 0.01 | 0.1 | Values long-term rewards more |
| 5 | Low Gamma | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 0.1 | Focuses on immediate rewards |
| 6 | Large Batch | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | More stable gradient estimates |
| 7 | Small Batch | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 0.1 | Noisier but more frequent updates |
| 8 | Ext Explore | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.2 | More exploration time |
| 9 | Quick Exploit | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.05 | Fast transition to exploitation |
| 10 | Aggressive | 1e-3 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | Very fast learning with stability |

### Observed Results

*(To be filled after running experiments)*

| Exp | Avg Reward | Max Reward | Convergence Speed | Stability | Key Observations |
|-----|------------|------------|-------------------|-----------|------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |



## Monitoring Training

### TensorBoard

Monitor training progress in real-time:

```bash
tensorboard --logdir ./logs/
```

Then open your browser to: `http://localhost:6006`

**Metrics Available**:
- Episode reward mean (reward trends)
- Episode length mean
- Exploration rate (epsilon decay)
- Training loss
- Q-value estimates

### Training Logs

CSV logs are saved in `logs/dqn_atari/progress.csv` with columns:
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/ep_len_mean` - Average episode length
- `time/total_timesteps` - Total training steps
- `rollout/exploration_rate` - Current epsilon value

---

## Video Demonstration

A video demonstration of the trained agent is included showing:
- The agent loading the trained model (`dqn_model.zip`)
- Multiple episodes of gameplay
- The agent using GreedyQPolicy for optimal performance
- Real-time interaction with the Assault environment

**To generate your own video**:
```bash
python play.py --record --episodes 5
```

Videos are saved in the `videos/` directory.

## DQN Architecture

### Network Structure

- **Policy**: CNNPolicy (Convolutional Neural Network)
- **Input**: 84x84x4 grayscale images (4 stacked frames)
- **Convolutional Layers**:
  - Conv1: 32 filters, 8x8 kernel, stride 4, ReLU
  - Conv2: 64 filters, 4x4 kernel, stride 2, ReLU
  - Conv3: 64 filters, 3x3 kernel, stride 1, ReLU
- **Fully Connected**: 512 units, ReLU
- **Output Layer**: Q-values for each of 7 actions

### Key Features

- **Experience Replay**: Buffer size of 100,000 transitions
- **Target Network**: Updated every 1,000 steps
- **Double DQN**: Reduces overestimation bias
- **Frame Stacking**: 4 consecutive frames for temporal information
- **Frame Skipping**: Action repeated every 4 frames
- **Epsilon-Greedy Exploration**: Decays from 1.0 to 0.01
- **Reward Clipping**: Not applied (preserves score information)

## References

- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Atari Environments](https://ale.farama.org/environments/)
- [Assault Environment Documentation](https://ale.farama.org/environments/assault/)
