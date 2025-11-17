import json

# Final data for all 10 experiments
results_data = {
    "experiment_1": {
        "name": "My Baseline",
        "peak_reward": 701.4,
        "peak_timestep": 440000,
        "model_path": "./models/experiment_01/best/best_model.zip"
    },
    "experiment_2": {
        "name": "Slower LR",
        "peak_reward": 682.4,
        "peak_timestep": 480000,
        "model_path": "./models/experiment_02/best/best_model.zip"
    },
    "experiment_3": {
        "name": "Faster LR",
        "peak_reward": 692.4,
        "peak_timestep": 320000,
        "model_path": "./models/experiment_03/best/best_model.zip"
    },
    "experiment_4": {
        "name": "Small Batch",
        "peak_reward": 407.4,
        "peak_timestep": 400000,
        "model_path": "./models/experiment_04/best/best_model.zip"
    },
    "experiment_5": {
        "name": "Small Batch & Fast LR",
        "peak_reward": 646.6,
        "peak_timestep": 440000,
        "model_path": "./models/experiment_05/best/best_model.zip"
    },
    "experiment_6": {
        "name": "Large Batch & Slow LR",
        "peak_reward": 709.4,
        "peak_timestep": 320000,
        "model_path": "./models/experiment_06/best/best_model.zip"
    },
    "experiment_7": {
        "name": "Short-Sighted",
        "peak_reward": 470.4,
        "peak_timestep": 400000,
        "model_path": "./models/experiment_07/best/best_model.zip"
    },
    "experiment_8": {
        "name": "Longer Exploration",
        "peak_reward": 709.8,
        "peak_timestep": 400000,
        "model_path": "./models/experiment_08/best/best_model.zip"
    },
    "experiment_9": {
        "name": "Original DQN Epsilon",
        "peak_reward": 552.2,
        "peak_timestep": 280000,
        "model_path": "./models/experiment_09/best/best_model.zip"
    },
    "experiment_10": {
        "name": "My Combo",
        "peak_reward": 669.8,
        "peak_timestep": 160000,
        "model_path": "./models/experiment_10/best/best_model.zip"
    }
}

# --- This part finds and prints the winner ---
best_exp_id = None
best_score = -float('inf')

for exp_id, data in results_data.items():
    if isinstance(data["peak_reward"], (int, float)) and data["peak_reward"] > best_score:
        best_score = data["peak_reward"]
        best_exp_id = exp_id

print(f"\n{'='*50}")
print(f"WINNER: {best_exp_id} ({results_data[best_exp_id]['name']})")
print(f"Best Score: {best_score}")
print(f"Model Path: {results_data[best_exp_id]['model_path']}")
print(f"{'='*50}\n")

# === ⭐️ UPDATED FILENAME HERE ===
output_filename = "loic_experiment_results.json"
# ==================================

with open(output_filename, 'w') as f:
    json.dump(results_data, f, indent=4, sort_keys=True)

print(f"Successfully created {output_filename}!")
print("This file is now ready for your project submission.")