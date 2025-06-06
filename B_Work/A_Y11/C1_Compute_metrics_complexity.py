import os
import pandas as pd
import numpy as np
from tabulate import tabulate

model_map = {
    'nano':   'n',
    'small':  's',
    'medium': 'm',
    'large':  'l',
    'xlarge': 'x'
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'A_Y11_Models')

training_stats = {}

for model_name, short in model_map.items():
    idx = list(model_map.keys()).index(model_name) + 1
    run_folder = os.path.join(BASE_DIR, f"{idx}_runs_{model_name}")
    
    if not os.path.isdir(run_folder):
        print(f"Model folder not found: {run_folder}")
        training_stats[model_name] = (None, None)
        continue

    times = []

    for fold in range(1, 6):
        fold_folder = f"train{fold}"
        csv_path = os.path.join(run_folder, fold_folder, "results.csv")

        if not os.path.isfile(csv_path):
            print(f"File not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"Empty CSV: {csv_path}")
                continue

            time_value = df.iloc[-1, 1]
            times.append(float(time_value))
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        training_stats[model_name] = (avg_time, std_time)
    else:
        training_stats[model_name] = (None, None)

table = [[
    model,
    f"{avg:.2f}" if avg is not None else "N/A",
    f"{std:.2f}" if std is not None else "N/A"
] for model, (avg, std) in training_stats.items()]

print(tabulate(table, headers=["Model", "Mean (s)", "Std Dev (s)"]))
