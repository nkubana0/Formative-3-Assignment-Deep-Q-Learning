"""
My 10 Hyperparameter Experiments for Assault
- Cyusa Loic
"""

import subprocess
import json
import os
from datetime import datetime


MY_EXPERIMENTS = [
    {
        "id": 1,
        "name": "My Baseline",
        "params": {"lr": 1e-4, "gamma": 0.99, "batch_size": 32, "exp_fraction": 0.1, "eps_end": 0.01},
        "expected_behavior": "Standard DQN - baseline for comparison"
    },
    {
        "id": 2,
        "name": "Slower LR",
        "params": {"lr": 7.5e-5, "gamma": 0.99, "batch_size": 32, "exp_fraction": 0.1, "eps_end": 0.01},
        "expected_behavior": "Slower, but potentially more stable learning."
    },
    {
        "id": 3,
        "name": "Faster LR",
        "params": {"lr": 2.5e-4, "gamma": 0.99, "batch_size": 32, "exp_fraction": 0.1, "eps_end": 0.01},
        "expected_behavior": "Faster learning, but may become unstable."
    },
    {
        "id": 4,
        "name": "Small Batch",
        "params": {"lr": 1e-4, "gamma": 0.99, "batch_size": 16, "exp_fraction": 0.1, "eps_end": 0.01},
        "expected_behavior": "Noisier updates, but can sometimes learn faster."
    },
    {
        "id": 5,
        "name": "Small Batch & Fast LR",
        "params": {"lr": 2.5e-4, "gamma": 0.99, "batch_size": 16, "exp_fraction": 0.1, "eps_end": 0.01},
        "expected_behavior": "A very volatile combo. May learn fast or fail."
    },
    {
        "id": 6,
        "name": "Large Batch & Slow LR",
        "params": {"lr": 7.5e-5, "gamma": 0.99, "batch_size": 64, "exp_fraction": 0.1, "eps_end": 0.01},
        "expected_behavior": "Very stable, but might converge slowly."
    },
    {
        "id": 7,
        "name": "Short-Sighted",
        "params": {"lr": 1e-4, "gamma": 0.97, "batch_size": 32, "exp_fraction": 0.1, "eps_end": 0.01},
        "expected_behavior": "Focuses more on immediate rewards."
    },
    {
        "id": 8,
        "name": "Longer Exploration",
        "params": {"lr": 1e-4, "gamma": 0.99, "batch_size": 32, "exp_fraction": 0.2, "eps_end": 0.01},
        "expected_behavior": "Spends 20% of time exploring."
    },
    {
        "id": 9,
        "name": "Original DQN Epsilon",
        "params": {"lr": 1e-4, "gamma": 0.99, "batch_size": 32, "exp_fraction": 0.1, "eps_end": 0.1},
        "expected_behavior": "Explores more; never fully greedy (ends at 10% random)."
    },
    {
        "id": 10,
        "name": "My Combo",
        "params": {"lr": 2.5e-4, "gamma": 0.99, "batch_size": 64, "exp_fraction": 0.15, "eps_end": 0.05},
        "expected_behavior": "A balanced 'fast and stable' attempt."
    }
]

# ==============================================================================
# 2. SCRIPT TO RUN THE EXPERIMENTS
# ==============================================================================

def run_experiment(exp_config, env_name='AssaultNoFrameskip-v4', timesteps=500_000):
    """
    Run single experiment with specified configuration
    """
    exp_id = exp_config['id']
    params = exp_config['params']
    
    print(f"\n{'#'*80}")
    print(f"# MY EXPERIMENT {exp_id}: {exp_config['name']}")
    print(f"{'#'*80}")
    print(f"Expected: {exp_config['expected_behavior']}")
    print(f"\nHyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'#'*80}\n")
    
    # Build command
    cmd = [
        'python', 'train.py',
        '--env', env_name,
        '--timesteps', str(timesteps),
        '--experiment', str(exp_id)
    ]
    
    # Add all params from the dictionary to the command
    for key, value in params.items():
        cmd.append(f"--{key.replace('_', '-')}") # e.g., batch_size -> --batch-size
        cmd.append(str(value))
    
    # Run training in a subprocess
    try:
        # Using subprocess.run to execute the command
        # This will wait for each experiment to finish before starting the next
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ My Experiment {exp_id} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ My Experiment {exp_id} failed: {e}\n")
        return False

# ==============================================================================
# 3. MAIN SCRIPT RUNNER
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run my 10 hyperparameter experiments on Assault')
    parser.add_argument('--env', type=str, default='AssaultNoFrameskip-v4',
                        help='Atari environment')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Training timesteps per experiment')
    parser.add_argument('--experiments', nargs='+', type=int,
                        help='Specific experiments to run (e.g., --experiments 1 2 3)')
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if args.experiments:
        experiments_to_run = [exp for exp in MY_EXPERIMENTS if exp['id'] in args.experiments]
    else:
        experiments_to_run = MY_EXPERIMENTS
        
    print(f"\n{'='*80}")
    print(f"Running {len(experiments_to_run)} of my experiments on Assault")
    print(f"Environment: {args.env}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"{'='*80}\n")
    
    # Run experiments
    results = {}
    for exp_config in experiments_to_run:
        success = run_experiment(exp_config, args.env, args.timesteps)
        results[exp_config['id']] = {
            'name': exp_config['name'],
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
    
    # Summary
    print(f"\n{'='*80}")
    print("MY EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results.values() if r['success'])
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {len(results) - successful}/{len(results)}")
    print(f"{'='*80}\n")