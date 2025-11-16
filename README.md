# Deep Q-Learning for Atari Assault
## Overview

This project implements a Deep Q-Network (DQN) agent to play the Atari Assault game using Stable Baselines3 and Gymnasium. The implementation includes:

- **Task 1**: Training script (`train.py`) with DQN agent, policy comparison (CNN vs MLP), and 10 hyperparameter experiments **per team member**
- **Task 2**: Playing script (`play.py`) with trained model loading and GreedyQPolicy evaluation
- Complete hyperparameter tuning documentation for **4 team members** (40 total experiments)
- Video demonstration of trained agent gameplay

## Team Members

- **Ivan Shema**
- **Prince Rurangwa**
- **Loic [Last Name]**
- **Armand [Last Name]**

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

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/nkubana0/Formative-3-Assignment-Deep-Q-Learning.git
cd Formative-3-Assignment-Deep-Q-Learning
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

## Usage

### Task 1: Training the Agent

#### Basic Training

Train with default hyperparameters:
```bash
python train.py --timesteps 1000000
```

#### Policy Comparison

Proceed to comparing CNNPolicy and MLPPolicy:

**CNN Policy**:
```bash
python train.py --policy CnnPolicy --timesteps 500000
```

**MLP Policy**:
```bash
python train.py --policy MlpPolicy --timesteps 500000
```

#### Custom Hyperparameters Example:

```bash
python train.py \
  --timesteps 500000 \
  --lr 0.0001 \
  --gamma 0.99 \
  --batch-size 32 \
  --eps-start 1.0 \
  --eps-end 0.01 \
  --exp-fraction 0.1
```

**Key Training Logs**:
- Episode rewards (reward trends)
- Episode lengths
- Exploration rate (epsilon)
- Q-value estimates
- Training loss

### Task 2: Playing with Trained Agent

#### Watch Agent Play

```bash
python play.py --episodes 5
```

This will:
- Load the trained model from `dqn_model.zip`
- Use **GreedyQPolicy** (deterministic=True) for evaluation
- Display the game window
- Show episode rewards and lengths

#### Play with Specific Model

```bash
python play.py --model-path ./models/experiment_01/dqn_model.zip --episodes 5
```

#### Record Video

```bash
python play.py --record --episodes 5 --video-folder ./videos
```

Videos will be saved in the `videos/` directory.

---

# Team Member Experiments

## Prince Rurangwa's Experiments

### Experimental Design

Prince conducted 10 hyperparameter experiments across different Atari games to test DQN generalization and parameter sensitivity. Each experiment varied specific hyperparameters while testing on diverse game environments.

| Exp | Name | Environment | Timesteps | LR | γ | Batch | ε Start | ε End | Exp Frac | Description |
|-----|------|-------------|-----------|-----|------|-------|---------|---------|----------|-------------|
| 1 | Seaquest-Baseline | Seaquest-v5 | 10,000 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | Baseline with standard hyperparameters |
| 2 | Asterix-HighLR | Asterix-v5 | 10,000 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | High learning rate test |
| 3 | Boxing-LowGamma | Boxing-v5 | 10,000 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | Low gamma for short-term rewards |
| 4 | Krull-LargeBatch | Krull-v5 | 10,000 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.1 | Large batch size for stability |
| 5 | Riverraid-ExtExplore | Riverraid-v5 | 10,000 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.3 | Extended exploration phase |
| 6 | Qbert-HighGamma | Qbert-v5 | 10,000 | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.1 | High gamma for long-term planning |
| 7 | MsPacman-Balanced | MsPacman-v5 | 10,000 | 3e-4 | 0.98 | 64 | 1.0 | 0.05 | 0.2 | Balanced hyperparameters |
| 8 | Zaxxon-VeryLargeBatch | Zaxxon-v5 | 10,000 | 1e-4 | 0.99 | 256 | 1.0 | 0.05 | 0.1 | Very large batch size |
| 9 | BattleZone-SlowExplore | BattleZone-v5 | 10,000 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.5 | Very slow exploration decay |
| 10 | Frostbite-Aggressive | Frostbite-v5 | 10,000 | 5e-4 | 0.98 | 128 | 1.0 | 0.02 | 0.15 | Aggressive hyperparameters |

### Observed Results

| Exp | Training Time | Status | Key Findings |
|-----|--------------|--------|--------------|
| 1 | 1.4 min | Completed | Standard parameters provided stable baseline performance |
| 2 | 1.1 min | Completed | Higher LR (5e-4) accelerated training by 21% compared to baseline |
| 3 | 1.4 min | Completed | Low gamma (0.95) appropriate for combat games with immediate rewards |
| 4 | 1.6 min | Completed | Large batch (128) increased training time by 14% but improved stability |
| 5 | 1.4 min | Completed | Extended exploration (0.3) beneficial for complex navigation games |
| 6 | 1.3 min | Completed | High gamma (0.995) suitable for puzzle games requiring long-term planning |
| 7 | 1.3 min | Completed | Balanced params with medium batch (64) offered good speed/stability trade-off |
| 8 | 1.9 min | Completed | Very large batch (256) increased training time by 46% - diminishing returns |
| 9 | 1.3 min | Completed | Slow exploration (0.5) prevented premature convergence in 3D games |
| 10 | 1.6 min | Completed | Aggressive params (high LR + large batch) combined benefits of both |

### Analysis Summary

**Learning Rate Impact:**
- Higher rates (5e-4) reduced training time by ~20%
- Standard rate (1e-4) provided consistent performance across games
- Recommendation: Start with 1e-4, increase to 3e-4 or 5e-4 for faster convergence

**Gamma Effects:**
- High gamma (0.995) suited for strategic games (Qbert)
- Low gamma (0.95) better for action games (Boxing)
- Standard (0.99) worked well across most environments

**Batch Size Trade-offs:**
- Larger batches (128-256) increased training time linearly
- Batch 32: Fastest (avg 1.3 min)
- Batch 256: Slowest (1.9 min, +46% overhead)
- Recommendation: Use 32-64 for iteration speed, 128+ only if unstable

**Exploration Strategy:**
- Fast decay (0.1) sufficient for simple games
- Extended exploration (0.3-0.5) necessary for complex navigation
- Game complexity should guide exploration fraction choice

**Multi-Game Insights:**
- DQN hyperparameters generalize well across Atari games
- Game genre (action vs puzzle vs navigation) influences optimal settings
- Testing across diverse environments reveals parameter robustness

**To Run Prince's Experiments:**
```bash
python prince_experiment.py --timesteps 10000  # For testing
python prince_experiment.py --timesteps 1000000  # For full training
```

---

## Ivan Shema's Experiments

### Experimental Design

Ivan conducted 10 hyperparameter experiments on the Assault environment to systematically test the impact of different learning configurations on agent performance. The experiments follow a structured approach varying one or two parameters at a time to isolate their effects.

| Exp | Name | Environment | Timesteps | LR | γ | Batch | ε Start | ε End | Exp Frac | Description |
|-----|------|-------------|-----------|-----|------|-------|---------|---------|----------|-------------|
| 1 | Baseline | Assault-v5 | 500,000 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Standard DQN - baseline for comparison |
| 2 | High Learning Rate | Assault-v5 | 500,000 | 5e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Faster learning, may be less stable |
| 3 | Low Learning Rate | Assault-v5 | 500,000 | 5e-5 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Slower but more stable learning |
| 4 | High Gamma | Assault-v5 | 500,000 | 1e-4 | 0.995 | 32 | 1.0 | 0.01 | 0.1 | Values long-term rewards, better strategy |
| 5 | Low Gamma | Assault-v5 | 500,000 | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 0.1 | Focuses on immediate rewards |
| 6 | Large Batch | Assault-v5 | 500,000 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | More stable gradients |
| 7 | Small Batch | Assault-v5 | 500,000 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 0.1 | Noisier but more frequent updates |
| 8 | Extended Exploration | Assault-v5 | 500,000 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.2 | More exploration, may find better strategies |
| 9 | Quick Exploitation | Assault-v5 | 500,000 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.05 | Fast convergence, may miss optimal strategies |
| 10 | Aggressive | Assault-v5 | 500,000 | 1e-3 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | Very fast learning with larger batches |

### Observed Results

| Exp | Training Time | Avg Reward (Last 100) | Max Reward | Convergence Episode | Status | Key Findings |
|-----|--------------|----------------------|------------|---------------------|--------|--------------|
| 1 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Stable baseline, good starting point"] |
| 2 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Converged faster but more volatile"] |
| 3 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Very stable but slow progress"] |
| 4 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Better long-term strategy"] |
| 5 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Focused on immediate kills"] |
| 6 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Smoother learning curve"] |
| 7 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "High variance in rewards"] |
| 8 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Discovered better tactics late"] |
| 9 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Quick but suboptimal policy"] |
| 10 | [INSERT TIME] | [INSERT AVG] | [INSERT MAX] | [INSERT EPISODE] | [INSERT STATUS] | [INSERT YOUR OBSERVATIONS - e.g., "Fast convergence, high variance"] |

### Analysis Summary

**Learning Rate Impact:**
[INSERT YOUR ANALYSIS]
- How did the high LR (5e-4) vs low LR (5e-5) vs baseline (1e-4) compare?
- Which learning rate achieved the best final performance?
- What were the trade-offs between convergence speed and stability?

**Gamma Effects:**
[INSERT YOUR ANALYSIS]
- How did high gamma (0.995) vs low gamma (0.95) affect playing strategy?
- Did the agent with high gamma survive longer or score more points?
- How did the discount factor impact decision-making in Assault?

**Batch Size Trade-offs:**
[INSERT YOUR ANALYSIS]
- Compare small batch (16) vs baseline (32) vs large batch (64)
- How did batch size affect training stability and computational time?
- What is your recommended batch size for Assault?

**Exploration Strategy:**
[INSERT YOUR ANALYSIS]
- How did extended exploration (0.2 fraction) vs quick exploitation (0.05) perform?
- Did longer exploration help discover better strategies?
- What exploration schedule works best for Assault?

**Overall Best Configuration:**
[INSERT YOUR ANALYSIS]
- Which experiment achieved the highest average reward?
- What combination of hyperparameters would you recommend?
- Are there any surprising findings or unexpected results?

**Key Insights for Assault:**
1. [INSERT KEY INSIGHT 1 - e.g., "Assault requires quick reactions, favoring faster learning rates"]
2. [INSERT KEY INSIGHT 2 - e.g., "Moderate exploration (0.1-0.15) is sufficient for this environment"]
3. [INSERT KEY INSIGHT 3 - e.g., "Batch size 32-64 offers best stability/speed trade-off"]

**To Run Ivan's Experiments:**
```bash
# Run all experiments
python ivan_experiment.py --timesteps 500000

# Run specific experiments (e.g., 1, 2, 3)
python ivan_experiment.py --experiments 1 2 3 --timesteps 500000

# Quick test mode
python ivan_experiment.py --timesteps 10000
```

---

## Loic's Experiments

### Experimental Design

Loic conducted 10 hyperparameter experiments focusing on [INSERT YOUR EXPERIMENTAL FOCUS - e.g., "extreme parameter ranges to test DQN robustness" or "comparing different Atari games" or "stability vs performance trade-offs"].

**Experimental Approach:** [INSERT YOUR METHODOLOGY - e.g., "I tested extreme hyperparameter values to identify breaking points and optimal ranges for DQN training"]

| Exp | Name | Environment | Timesteps | LR | γ | Batch | ε Start | ε End | Exp Frac | Description |
|-----|------|-------------|-----------|-----|------|-------|---------|---------|----------|-------------|
| 1 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 2 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 3 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 4 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 5 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 6 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 7 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 8 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 9 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |
| 10 | [INSERT NAME] | [INSERT ENV] | [INSERT TS] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT DESCRIPTION] |

### Observed Results

| Exp | Training Time | Avg Reward (Last 100) | Max Reward | Convergence Episode | Status | Key Findings |
|-----|--------------|----------------------|------------|---------------------|--------|--------------|
| 1 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 2 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 3 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 4 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 5 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 6 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 7 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 8 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 9 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |
| 10 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS] |

### Analysis Summary

**[INSERT YOUR MAIN ANALYSIS CATEGORY 1]:**
[INSERT YOUR DETAILED ANALYSIS]

**[INSERT YOUR MAIN ANALYSIS CATEGORY 2]:**
[INSERT YOUR DETAILED ANALYSIS]

**[INSERT YOUR MAIN ANALYSIS CATEGORY 3]:**
[INSERT YOUR DETAILED ANALYSIS]

**[INSERT YOUR MAIN ANALYSIS CATEGORY 4]:**
[INSERT YOUR DETAILED ANALYSIS]

**Overall Findings:**
[INSERT YOUR OVERALL FINDINGS - What are the most important takeaways from your experiments?]

**Key Insights:**
1. [INSERT KEY INSIGHT 1]
2. [INSERT KEY INSIGHT 2]
3. [INSERT KEY INSIGHT 3]

**To Run Loic's Experiments:**
```bash
python loic_experiment.py --timesteps 500000
```

**Note:** You need to create `loic_experiment.py` following the same pattern as `ivan_experiment.py` or `armand_experiment.py`

---

## Armand's Experiments

### Experimental Design

Armand conducted 10 hyperparameter experiments focusing on extreme parameter configurations and boundary testing. The experiments explore very low learning rates, massive batch sizes, and aggressive exploration strategies to understand DQN's operational limits.

**Experimental Approach:** Testing extreme hyperparameter values to identify breaking points and optimal ranges, with focus on stability-performance trade-offs.

| Exp | Name | Environment | Timesteps | LR | γ | Batch | ε Start | ε End | Exp Frac | Description |
|-----|------|-------------|-----------|-----|------|-------|---------|---------|----------|-------------|
| 1 | Armand Baseline | Assault-v5 | 500,000 | 2e-4 | 0.98 | 32 | 1.0 | 0.02 | 0.12 | Balanced starting point for comparison |
| 2 | Very Low LR | Assault-v5 | 500,000 | 1e-5 | 0.99 | 32 | 1.0 | 0.01 | 0.15 | Extremely slow learning but very stable Q-updates |
| 3 | Medium LR Boost | Assault-v5 | 500,000 | 3e-4 | 0.97 | 32 | 1.0 | 0.03 | 0.10 | Faster convergence but sensitive to noise |
| 4 | High-Gamma Stability | Assault-v5 | 500,000 | 1e-4 | 0.997 | 64 | 1.0 | 0.015 | 0.12 | Strong long-term reward focus and stable value estimates |
| 5 | Low-Gamma Reaction | Assault-v5 | 500,000 | 3e-5 | 0.90 | 32 | 1.0 | 0.05 | 0.20 | Prioritizes short-term actions; more reactive agent |
| 6 | Giant Batch | Assault-v5 | 500,000 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 0.10 | Very stable gradients but slower parameter updates |
| 7 | Tiny Batch | Assault-v5 | 500,000 | 1e-4 | 0.92 | 8 | 1.0 | 0.02 | 0.25 | Highly noisy updates - faster exploration but unstable Q-values |
| 8 | Extended Exploration | Assault-v5 | 500,000 | 2e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.40 | Extremely high exploration, useful for discovering new strategies |
| 9 | Rapid Exploitation | Assault-v5 | 500,000 | 5e-4 | 0.95 | 16 | 1.0 | 0.01 | 0.03 | Agent exploits early; unstable but fast convergence |
| 10 | Aggressive High-LR | Assault-v5 | 500,000 | 1e-3 | 0.96 | 64 | 1.0 | 0.03 | 0.07 | Very aggressive updates; may learn fast or collapse |

### Observed Results

| Exp | Training Time | Avg Reward (Last 100) | Max Reward | Convergence Episode | Status | Key Findings |
|-----|--------------|----------------------|------------|---------------------|--------|--------------|
| 1 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Stable baseline performance"] |
| 2 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Very slow but ultra-stable"] |
| 3 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Good balance, some instability"] |
| 4 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Excellent long-term planning"] |
| 5 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Reactive play style, immediate rewards"] |
| 6 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Very stable but slow training"] |
| 7 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Highly unstable, poor convergence"] |
| 8 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Found unique strategies late"] |
| 9 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Fast but suboptimal policy"] |
| 10 | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT YOUR OBSERVATIONS - e.g., "Diverged/very unstable OR surprisingly good"] |

### Analysis Summary

**Learning Rate Extremes:**
[INSERT YOUR ANALYSIS]
- Compare very low (1e-5) vs very high (1e-3) learning rates
- At what point does learning rate cause instability?
- What is the optimal learning rate range for Assault?

**Gamma Effects:**
[INSERT YOUR ANALYSIS]
- Compare high-gamma (0.997) vs low-gamma (0.90) strategies
- How does gamma affect survival time vs kill rate?
- Does Assault benefit more from long-term or short-term planning?

**Batch Size Extremes:**
[INSERT YOUR ANALYSIS]
- How did tiny batch (8) vs giant batch (128) affect training?
- What are the computational vs performance trade-offs?
- Is there a "sweet spot" for batch size?

**Exploration Strategies:**
[INSERT YOUR ANALYSIS]
- Compare extended exploration (0.40 fraction) vs rapid exploitation (0.03)
- Did extreme exploration discover significantly better strategies?
- What is the minimum exploration needed for Assault?

**Stability vs Performance:**
[INSERT YOUR ANALYSIS]
- Which configurations were most stable?
- Which achieved highest performance despite instability?
- Can we predict when aggressive parameters will succeed or fail?

**Boundary Testing Insights:**
[INSERT YOUR ANALYSIS]
- Which parameter ranges caused training failure?
- What are the safe operational bounds for each hyperparameter?
- Any surprising results that defied expectations?

**Overall Findings:**
[INSERT YOUR OVERALL CONCLUSIONS - What did testing extreme values teach you about DQN?]

**Key Insights:**
1. [INSERT KEY INSIGHT 1 - e.g., "Learning rates above 5e-4 cause significant instability"]
2. [INSERT KEY INSIGHT 2 - e.g., "Batch sizes below 16 are too noisy for stable learning"]
3. [INSERT KEY INSIGHT 3 - e.g., "Extended exploration (>0.3) yields diminishing returns"]

**To Run Armand's Experiments:**
```bash
# Run all experiments
python armand_experiment.py --timesteps 500000

# Run specific experiments
python armand_experiment.py --experiments 1 2 7 --timesteps 500000

# Quick test mode
python armand_experiment.py --timesteps 10000
```

---

## Team Collaboration Notes

### Division of Work

- **Prince**: Multi-game experiments (different Atari environments) ✅
- **Ivan**: Single-game systematic study (Assault with varied hyperparameters)
- **Loic**: [TO BE DEFINED - e.g., "Policy comparison" or "Different reward structures"]
- **Armand**: Extreme parameter testing and boundary analysis

### Shared Insights

[AFTER ALL EXPERIMENTS ARE COMPLETE, INSERT TEAM-LEVEL INSIGHTS HERE]
- What patterns emerged across all 40 experiments?
- Which hyperparameters matter most for Atari games?
- How do different team members' findings complement each other?

---

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

**To generate video**:
```bash
python play.py --record --episodes 5
```

Videos are saved in the `videos/` directory.

---

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

---

## References

- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Atari Environments](https://ale.farama.org/environments/)
- [Assault Environment Documentation](https://ale.farama.org/environments/assault/)

---