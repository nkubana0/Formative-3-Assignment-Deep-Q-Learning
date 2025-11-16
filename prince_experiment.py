import subprocess
import json
from datetime import datetime

EXPERIMENTS = [
    {
        "id": 1,
        "name": "Prince-Seaquest-HighLR",
        "env": "SeaquestNoFrameskip-v4",
        "timesteps": 1_000_000,
        "params": {
            "lr": 3e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "Seaquest with high learning rate - tests fast convergence on underwater navigation"
    },
    {
        "id": 2,
        "name": "Prince-Asterix-ModLR",
        "env": "AsterixNoFrameskip-v4",
        "timesteps": 500_000,
        "params": {
            "lr": 1.5e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "eps_start": 1.0,
            "eps_end": 0.05,
            "exp_fraction": 0.15
        },
        "expected_behavior": "Asterix with moderate learning rate and larger batch - balanced exploration"
    },
    {
        "id": 3,
        "name": "Prince-Boxing-LongTerm",
        "env": "BoxingNoFrameskip-v4",
        "timesteps": 750_000,
        "params": {
            "lr": 1e-4,
            "gamma": 0.995,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.02,
            "exp_fraction": 0.2
        },
        "expected_behavior": "Boxing with high gamma - emphasizes long-term strategic play"
    },
    {
        "id": 4,
        "name": "Prince-Krull-LargeBatch",
        "env": "KrullNoFrameskip-v4",
        "timesteps": 1_000_000,
        "params": {
            "lr": 2.5e-4,
            "gamma": 0.99,
            "batch_size": 128,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "Krull with very large batch size - stable gradient estimates"
    },
    {
        "id": 5,
        "name": "Prince-Riverraid-ShortTerm",
        "env": "RiverraidNoFrameskip-v4",
        "timesteps": 500_000,
        "params": {
            "lr": 2e-4,
            "gamma": 0.95,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.1,
            "exp_fraction": 0.25
        },
        "expected_behavior": "Riverraid with low gamma - focuses on immediate rewards"
    },
    {
        "id": 6,
        "name": "Prince-Qbert-SlowExplore",
        "env": "QbertNoFrameskip-v4",
        "timesteps": 1_000_000,
        "params": {
            "lr": 1e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "eps_start": 1.0,
            "eps_end": 0.05,
            "exp_fraction": 0.5
        },
        "expected_behavior": "Qbert with very slow exploration decay - thorough strategy discovery"
    },
    {
        "id": 7,
        "name": "Prince-MsPacman-Aggressive",
        "env": "MsPacmanNoFrameskip-v4",
        "timesteps": 750_000,
        "params": {
            "lr": 5e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.1
        },
        "expected_behavior": "MsPacman with aggressive learning rate - fast policy updates"
    },
    {
        "id": 8,
        "name": "Prince-Zaxxon-Mixed",
        "env": "ZaxxonNoFrameskip-v4",
        "timesteps": 500_000,
        "params": {
            "lr": 1.5e-4,
            "gamma": 0.98,
            "batch_size": 64,
            "eps_start": 0.9,
            "eps_end": 0.02,
            "exp_fraction": 0.2
        },
        "expected_behavior": "Zaxxon with mixed parameters - reduced initial exploration"
    },
    {
        "id": 9,
        "name": "Prince-BattleZone-MassiveBatch",
        "env": "BattleZoneNoFrameskip-v4",
        "timesteps": 1_000_000,
        "params": {
            "lr": 1e-4,
            "gamma": 0.99,
            "batch_size": 256,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "exp_fraction": 0.15
        },
        "expected_behavior": "BattleZone with massive batch size - extremely stable updates"
    },
    {
        "id": 10,
        "name": "Prince-Frostbite-Balanced",
        "env": "FrostbiteNoFrameskip-v4",
        "timesteps": 750_000,
        "params": {
            "lr": 2e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "eps_start": 1.0,
            "eps_end": 0.02,
            "exp_fraction": 0.12
        },
        "expected_behavior": "Frostbite with balanced parameters - control experiment"
    }
]


def run_experiment(exp_config, default_env='AssaultNoFrameskip-v4', default_timesteps=500_000):
    """Run a single experiment using train.py with the selected parameters."""
    exp_id = exp_config['id']
    params = exp_config['params']
    
    # Use experiment-specific env and timesteps if provided, otherwise use defaults
    env_name = exp_config.get('env', default_env)
    exp_timesteps = exp_config.get('timesteps', default_timesteps)

    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT {exp_id}: {exp_config['name']}")
    print(f"{'#'*80}")
    print(f"Environment: {env_name}")
    print(f"Timesteps: {exp_timesteps:,}")
    print(f"Expected: {exp_config.get('expected_behavior', '')}")
    print("\nHyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'#'*80}\n")

    cmd = [
        'python', 'train.py',
        '--env', env_name,
        '--timesteps', str(exp_timesteps),
        '--lr', str(params['lr']),
        '--gamma', str(params['gamma']),
        '--batch-size', str(params['batch_size']),
        '--eps-start', str(params['eps_start']),
        '--eps-end', str(params['eps_end']),
        '--exp-fraction', str(params['exp_fraction']),
        '--experiment', str(exp_id)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Experiment {exp_id} finished successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment {exp_id} failed: {e}\n")
        return False


def generate_results_table():
    """Return a markdown table summarizing experiments."""
    table = "## Prince's Hyperparameter Experiments\n\n"
    table += "| Exp | Name | Environment | Timesteps | lr | γ | Batch | ε Start | ε End | Exp Frac | Expected Behavior |\n"
    table += "|-----|------|-------------|-----------|-------|-------|-------|---------|---------|----------|-------------------|\n"
    for exp in EXPERIMENTS:
        p = exp['params']
        env = exp.get('env', 'AssaultNoFrameskip-v4')
        ts = exp.get('timesteps', 500_000)
        table += f"| {exp['id']} | {exp['name']} | {env} | {ts:,} | {p['lr']} | {p['gamma']} | {p['batch_size']} | {p['eps_start']} | {p['eps_end']} | {p['exp_fraction']} | {exp['expected_behavior']} |\n"
    return table


def save_experiment_configs():
    with open('prince_experiment_configs.json', 'w') as f:
        json.dump(EXPERIMENTS, f, indent=2)
    print("✓ Saved configs to prince_experiment_configs.json")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run Prince's hyperparameter experiments")
    parser.add_argument('--env', type=str, default='AssaultNoFrameskip-v4')
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--experiments', nargs='+', type=int,
                        help='Which experiments to run (ids)')
    parser.add_argument('--generate-table', action='store_true')

    args = parser.parse_args()

    if args.generate_table:
        table = generate_results_table()
        print(table)
        with open('prince_experiment_results_template.md', 'w') as f:
            f.write(table)
        print('✓ Table saved to prince_experiment_results_template.md')
    else:
        save_experiment_configs()

        if args.experiments:
            experiments_to_run = [exp for exp in EXPERIMENTS if exp['id'] in args.experiments]
        else:
            experiments_to_run = EXPERIMENTS

        print(f"\n{'='*80}")
        print(f"Running {len(experiments_to_run)} experiments on {args.env}")
        print(f"Timesteps per experiment: {args.timesteps:,}")
        print(f"{'='*80}\n")

        results = {}
        for exp_config in experiments_to_run:
            success = run_experiment(exp_config, args.env, args.timesteps)
            results[exp_config['id']] = {
                'name': exp_config['name'],
                'env': exp_config.get('env', args.env),
                'timesteps': exp_config.get('timesteps', args.timesteps),
                'success': success,
                'timestamp': datetime.now().isoformat()
            }

        with open('prince_experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print('EXPERIMENT SUMMARY')
        print(f"{'='*80}")
        successful = sum(1 for r in results.values() if r['success'])
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {len(results) - successful}/{len(results)}")
        print('\nResults saved to prince_experiment_results.json')
        print(f"{'='*80}\n")
