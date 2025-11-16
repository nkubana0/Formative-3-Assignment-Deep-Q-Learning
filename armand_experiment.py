"""
Armand's DQN Hyperparameter Experiment Runner for Assault
Runs 10 unique hyperparameter experiments using the Assault environment.
This is a completely original implementation for the assignment.
"""

import json
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
# Armand's 10 Unique Experiments
# ---------------------------------------------------------------------------

ARMAND_EXPERIMENTS = [
    {"id": 1, "name": "Armand Baseline",
     "params": {"lr": 2e-4, "gamma": 0.98, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.02, "exp_fraction": 0.12},
     "description": "Balanced starting point for comparison."},

    {"id": 2, "name": "Very Low Learning Rate",
     "params": {"lr": 1e-5, "gamma": 0.99, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.01, "exp_fraction": 0.15},
     "description": "Extremely slow learning but very stable Q-updates."},

    {"id": 3, "name": "Medium Learning Rate Boost",
     "params": {"lr": 3e-4, "gamma": 0.97, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.03, "exp_fraction": 0.10},
     "description": "Faster convergence but sensitive to noise."},

    {"id": 4, "name": "High-Gamma Stability Test",
     "params": {"lr": 1e-4, "gamma": 0.997, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.015, "exp_fraction": 0.12},
     "description": "Strong long-term reward focus and stable value estimates."},

    {"id": 5, "name": "Low-Gamma Reaction Strategy",
     "params": {"lr": 3e-5, "gamma": 0.90, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.05, "exp_fraction": 0.20},
     "description": "Prioritizes short-term actions; more reactive agent."},

    {"id": 6, "name": "Giant Batch Experiment",
     "params": {"lr": 1e-4, "gamma": 0.99, "batch_size": 128, "eps_start": 1.0, "eps_end": 0.01, "exp_fraction": 0.10},
     "description": "Very stable gradients but slower parameter updates."},

    {"id": 7, "name": "Tiny Batch Experiment",
     "params": {"lr": 1e-4, "gamma": 0.92, "batch_size": 8, "eps_start": 1.0, "eps_end": 0.02, "exp_fraction": 0.25},
     "description": "Highly noisy updates — faster exploration but unstable Q-values."},

    {"id": 8, "name": "Extended Exploration Mode",
     "params": {"lr": 2e-4, "gamma": 0.99, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.10, "exp_fraction": 0.40},
     "description": "Extremely high exploration, useful for discovering new strategies."},

    {"id": 9, "name": "Rapid Exploitation Mode",
     "params": {"lr": 5e-4, "gamma": 0.95, "batch_size": 16, "eps_start": 1.0, "eps_end": 0.01, "exp_fraction": 0.03},
     "description": "Agent exploits early; unstable but fast convergence."},

    {"id": 10, "name": "Aggressive High-LR Strategy",
     "params": {"lr": 1e-3, "gamma": 0.96, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.03, "exp_fraction": 0.07},
     "description": "Very aggressive updates; may learn fast or collapse."}
]

# ---------------------------------------------------------------------------
# Run a Single Experiment
# ---------------------------------------------------------------------------

def run_experiment(exp, env='AssaultNoFrameskip-v4', timesteps=500_000):
    print("\n" + "#"*80)
    print(f"Experiment {exp['id']}: {exp['name']}")
    print("#"*80)
    print(f"Description: {exp['description']}")
    print("Hyperparameters:")
    for k, v in exp["params"].items():
        print(f"  {k}: {v}")
    print("#"*80)

    cmd = [
        'python', 'train.py',
        '--env', env,
        '--timesteps', str(timesteps),
        '--lr', str(exp["params"]["lr"]),
        '--gamma', str(exp["params"]["gamma"]),
        '--batch-size', str(exp["params"]["batch_size"]),
        '--eps-start', str(exp["params"]["eps_start"]),
        '--eps-end', str(exp["params"]["eps_end"]),
        '--exp-fraction', str(exp["params"]["exp_fraction"]),
        '--experiment', str(exp["id"])
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✔ Experiment {exp['id']} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✘ Experiment {exp['id']} failed: {e}\n")
        return False

# ---------------------------------------------------------------------------
# Generate Markdown Table
# ---------------------------------------------------------------------------

def generate_markdown_table():
    header = "## Armand DQN Hyperparameter Experiments\n\n"
    header += "| ID | Name | lr | gamma | Batch | eps_start | eps_end | exp_fraction | Description |\n"
    header += "|----|------|----|-------|-------|-----------|---------|-------------|-------------|\n"

    rows = ""
    for exp in ARMAND_EXPERIMENTS:
        p = exp['params']
        rows += f"| {exp['id']} | {exp['name']} | {p['lr']} | {p['gamma']} | {p['batch_size']} | {p['eps_start']} | {p['eps_end']} | {p['exp_fraction']} | {exp['description']} |\n"

    return header + rows

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Armand's Assault DQN Hyperparameter Runner")
    parser.add_argument('--env', default='AssaultNoFrameskip-v4')
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--experiments', nargs='+', type=int, help='Run only specific experiments')
    parser.add_argument('--table', action='store_true', help='Generate markdown table and exit')
    args = parser.parse_args()

    # Save configs
    with open('armand_experiment_configs.json', 'w') as f:
        json.dump(ARMAND_EXPERIMENTS, f, indent=2)

    # Generate table if requested
    if args.table:
        table_md = generate_markdown_table()
        with open('armand_experiment_table.md', 'w') as f:
            f.write(table_md)
        print("✔ Markdown table saved to armand_experiment_table.md")
        exit()

    # Filter experiments
    if args.experiments:
        selected = [exp for exp in ARMAND_EXPERIMENTS if exp['id'] in args.experiments]
    else:
        selected = ARMAND_EXPERIMENTS

    print(f"\nRunning {len(selected)} experiment(s) on {args.env} for {args.timesteps:,} timesteps.\n")

    results = {}
    for exp in selected:
        success = run_experiment(exp, env=args.env, timesteps=args.timesteps)
        results[exp['id']] = {
            "name": exp['name'],
            "success": success,
            "timestamp": datetime.now().isoformat()
        }

    # Save results
    with open('armand_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nAll experiments completed. Results saved to armand_experiment_results.json\n")
