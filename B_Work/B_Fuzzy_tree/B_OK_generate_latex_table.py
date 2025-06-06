import os
import json
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(script_dir, "Results", "B_Json_CrossVal")

file_model_info = {
    "ICDM_results_d3_t3_optim_False_pre_False_minex_25.json": (3, 3),
    "ICDM_results_d5_t3_optim_False_pre_False_minex_25.json": (5, 3),
    "ICDM_results_d7_t3_optim_False_pre_False_minex_25.json": (7, 3),
    "ICDM_results_d15_t3_optim_False_pre_False_minex_25.json": (15, 3),
    "ICDM_results_d3_t5_optim_False_pre_False_minex_25.json": (3, 5),
    "ICDM_results_d5_t5_optim_False_pre_False_minex_25.json": (5, 5),
    "ICDM_results_d7_t5_optim_False_pre_False_minex_25.json": (7, 5),
    "ICDM_results_d15_t5_optim_False_pre_False_minex_25.json": (15, 5),
}

def extract_train_test_metrics(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    stats = {s: {"accuracy": [], "1": [], "2": [], "3": []} for s in ["train", "test"]}

    for round_data in data["rounds"]:
        for split in ["train", "test"]:
            report = round_data[f"{split}_report"]
            stats[split]["accuracy"].append(report["accuracy"])
            for cls in ["1", "2", "3"]:
                stats[split][cls].append({
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1-score": report[cls]["f1-score"]
                })

    def mean_std(values):
        series = pd.Series(values)
        return f"{series.mean():.2f} $\\pm$ {series.std():.2f}"

    def metrics_label(data):
        return tuple(mean_std([d[k] for d in data]) for k in ["precision", "recall", "f1-score"])

    results = {}
    for s in ["train", "test"]:
        results[s] = {
            "accuracy": mean_std(stats[s]["accuracy"]),
            "meningioma": metrics_label(stats[s]["1"]),
            "glioma": metrics_label(stats[s]["2"]),
            "pituitary": metrics_label(stats[s]["3"]),
        }
    return results

combined_entries = []
for filename, (depth, T) in sorted(file_model_info.items(), key=lambda x: (x[1][1], x[1][0])):
    metrics = extract_train_test_metrics(os.path.join(json_dir, filename))
    for mode in ["train", "test"]:
        r = metrics[mode]
        combined_entries.append((
            T, mode.capitalize(), depth, r["accuracy"],
            *r["meningioma"], *r["glioma"], *r["pituitary"]
        ))

def generate_combined_latex_table(entries):
    lines = []
    for entry in entries:
        row = (
            f"{entry[0]} & {entry[1]} & {entry[2]} & {entry[3]} & {entry[4]} & {entry[5]} & {entry[6]} & "
            f"{entry[7]} & {entry[8]} & {entry[9]} & {entry[10]} & {entry[11]} & {entry[12]} \\\\"
        )
        lines.append(row)
    return r"""
\begin{table}[ht]
\caption{Average classification performance using Fuzzy Decision Trees for $T \in \{3, 5\}$.}
\label{tab:accuracy_fuzzy_combined}
\centering
\footnotesize
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{0.3em}
\begin{tabular}{c|c|c|c|ccc|ccc|ccc}
\toprule
\textbf{T} & \textbf{Type} & \textbf{Depth} & \textbf{Accuracy} &
\multicolumn{3}{c|}{\textbf{Meningioma}} &
\multicolumn{3}{c|}{\textbf{Glioma}} &
\multicolumn{3}{c}{\textbf{Pituitary Tumor}} \\
& & & & Precision & Recall & F1-score & Precision & Recall & F1-score & Precision & Recall & F1-score \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

latex_combined_table = generate_combined_latex_table(combined_entries)
print(latex_combined_table)

def extract_complexity_statistics(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    leaves, nodes, np_vals = [], [], []

    for i, round_data in enumerate(data['rounds']):
        if 'num_leaves' not in round_data:
            raise KeyError(f"'num_leaves' missing in fold {i+1} of {filepath}")
        if 'num_nodes' not in round_data:
            raise KeyError(f"'num_nodes' missing in fold {i+1} of {filepath}")
        if 'num_parameters' not in round_data:
            raise KeyError(f"'num_parameters' missing in fold {i+1} of {filepath}")

        leaves.append(round_data['num_leaves'])
        nodes.append(round_data['num_nodes'])
        np_vals.append(round_data['num_parameters'])

    return {
        "leaves_mean": np.mean(leaves),
        "leaves_std": np.std(leaves),
        "nodes_mean": np.mean(nodes),
        "nodes_std": np.std(nodes),
        "np_mean": np.mean(np_vals),
        "np_std": np.std(np_vals),
    }

rows = []
for filename, (depth, T) in sorted(file_model_info.items(), key=lambda x: (x[1][1], x[1][0])):
    stats = extract_complexity_statistics(os.path.join(json_dir, filename))
    rows.append({
        "T": T,
        "Depth": depth,
        "#Leaves": f"{stats['leaves_mean']:.2f} $\\pm$ {stats['leaves_std']:.2f}",
        "#Nodes": f"{stats['nodes_mean']:.2f} $\\pm$ {stats['nodes_std']:.2f}",
        "NP": f"{stats['np_mean']:.2f} $\\pm$ {stats['np_std']:.2f}",
    })

df = pd.DataFrame(rows)

def generate_complexity_latex_table(df):
    latex = r"""\begin{table}[ht]
\caption{Average complexity results achieved by Fuzzy Decision Trees for different values of $T$.}
\label{tab:complexity_fuzzy_combined}
\centering
\footnotesize
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{1.0em}
\begin{tabular}{ccccc}
\toprule
\textbf{T} & \textbf{Depth} & \textbf{\#Leaves} & \textbf{\#Nodes} & \textbf{NP} \\
\midrule
"""
    for t_val in sorted(df["T"].unique()):
        subset = df[df["T"] == t_val]
        latex += f"\\multirow{{{len(subset)}}}{{*}}{{{t_val}}}\n"
        for _, row in subset.iterrows():
            latex += f" & {row['Depth']} & {row['#Leaves']} & {row['#Nodes']} & {row['NP']} \\\\\n"
        latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex

latex_complexity_output = generate_complexity_latex_table(df)
print(latex_complexity_output)
