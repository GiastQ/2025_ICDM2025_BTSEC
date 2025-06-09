import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from math import floor, ceil
from tabulate import tabulate

CONF_THRESH_LIST = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
IOU_THRES = 0.50
IOU_OK_THRESH = 0.50

model_map = {
    'nano':   'n',
    'small':  's',
    'medium': 'm',
    'large':  'l',
    'xlarge': 'x'
}

summary = {}

# === Caricamento dati ===
for index, (model_name, short) in enumerate(model_map.items(), start=1):
    summary[model_name] = {}
    for CONF_THRES in CONF_THRESH_LIST:
        summary[model_name][CONF_THRES] = {}
        for fold in range(1, 6):
            c = f"{CONF_THRES:.2f}".replace("0.", "")
            thresh_folder = f"conf{CONF_THRES:.2f}_iou{IOU_THRES:.2f}_ok{IOU_OK_THRESH:.2f}".replace("0.", "")

            yolo_path = os.path.join('A_Y11', 'B_Y11_inference', thresh_folder,
                                     f'{index}_runs_{model_name}', f'train{fold}',
                                     f'metrics_Y11{short}{fold}_c{c}.csv')
            fuzzy0_path = os.path.join('B_Fuzzy_tree', 'Results', 'D_Test_and_Rules',
                                       f'ICDM_fold{fold}_depth5_fsets5_test_rules.csv')
            fuzzy1_path = os.path.join('C_Radiomics_Y11', thresh_folder,
                                       f'{index}_Y11_{model_name}',
                                       f'ICDM_{fold}_d5_t5_test_and_rules.csv')
            try:
                yolo_df = pd.read_csv(yolo_path, sep=';')
                fuzzy0_df = pd.read_csv(fuzzy0_path)
                fuzzy1_df = pd.read_csv(fuzzy1_path)

                merged = pd.merge(yolo_df, fuzzy0_df, on=["image_id", "patient_id", "true_class"], how="outer")
                merged = pd.merge(merged, fuzzy1_df, on=["image_id", "patient_id", "true_class"], how="outer")
                summary[model_name][CONF_THRES][fold] = merged

            except Exception as e:
                print(f"[ERROR] {model_name} | conf={CONF_THRES} | fold={fold}: {e}")

# === Calcolo metriche ===
all_reports = []
fidelity_per_fold = []

for model_name in summary:
    for conf_thres in summary[model_name]:
        for fold in summary[model_name][conf_thres]:
            df = summary[model_name][conf_thres][fold]

            # Fidelity
            if 'Y11_pred_class' in df.columns and 'fuzY11_pred_class' in df.columns:
                df_fid = df.dropna(subset=['Y11_pred_class', 'fuzY11_pred_class'])
                if not df_fid.empty:
                    fidelity = (df_fid['Y11_pred_class'] == df_fid['fuzY11_pred_class']).mean()
                    fidelity_per_fold.append({
                        'model': model_name,
                        'confidence_threshold': conf_thres,
                        'fold': fold,
                        'fidelity': fidelity
                    })

            # Report + Accuracy
            if 'true_class' in df.columns and 'fuzY11_pred_class' in df.columns:
                df_clean = df.dropna(subset=['true_class', 'fuzY11_pred_class'])
                try:
                    report = classification_report(
                        df_clean['true_class'],
                        df_clean['fuzY11_pred_class'],
                        output_dict=True,
                        zero_division=0
                    )
                    for class_label in report:
                        if class_label.isdigit():
                            cls = int(class_label)
                            all_reports.append({
                                'model': model_name,
                                'confidence_threshold': conf_thres,
                                'class': cls,
                                'fold': fold,
                                'avg_precision': report[class_label]['precision'],
                                'avg_recall': report[class_label]['recall']
                            })
                    if 'accuracy' in report:
                        all_reports.append({
                            'model': model_name,
                            'confidence_threshold': conf_thres,
                            'class': -1,
                            'fold': fold,
                            'avg_precision': np.nan,
                            'avg_recall': np.nan,
                            'avg_accuracy': report['accuracy']
                        })
                except Exception as e:
                    print(f"[REPORT ERROR] {model_name} | conf={conf_thres} | fold={fold}: {e}")

# === Aggregazioni ===
metrics_by_key = {}
accuracy_by_key = {}

for entry in all_reports:
    if entry['class'] == -1:
        key = (entry['model'], entry['confidence_threshold'])
        accuracy_by_key.setdefault(key, []).append(entry['avg_accuracy'])
    else:
        key = (entry['model'], entry['confidence_threshold'], entry['class'])
        metrics_by_key.setdefault(key, {'precision': [], 'recall': []})
        metrics_by_key[key]['precision'].append(entry['avg_precision'])
        metrics_by_key[key]['recall'].append(entry['avg_recall'])

aggregated_metrics = []

for (model, conf, cls), values in metrics_by_key.items():
    p, r = values['precision'], values['recall']
    f1 = [2 * pi * ri / (pi + ri) if (pi + ri) > 0 else 0.0 for pi, ri in zip(p, r)]
    aggregated_metrics.append({
        'model': model,
        'confidence_threshold': conf,
        'class': cls,
        'precision_mean': np.mean(p),
        'precision_std': np.std(p),
        'recall_mean': np.mean(r),
        'recall_std': np.std(r),
        'f1_mean': np.mean(f1),
        'f1_std': np.std(f1)
    })

df_metrics = pd.DataFrame(aggregated_metrics)
df_fidelity_full = pd.DataFrame(fidelity_per_fold)
df_fidelity_grouped = df_fidelity_full.groupby(['model', 'confidence_threshold'])['fidelity'].agg(['mean', 'std']).reset_index().rename(columns={'mean': 'fidelity_mean', 'std': 'fidelity_std'})

df_accuracy = pd.DataFrame([
    {
        'model': model,
        'confidence_threshold': conf,
        'accuracy_mean': np.mean(acc),
        'accuracy_std': np.std(acc)
    }
    for (model, conf), acc in accuracy_by_key.items()
])

df_final_metrics = pd.merge(df_metrics, df_fidelity_grouped, on=['model', 'confidence_threshold'], how='left')
df_final_metrics = pd.merge(df_final_metrics, df_accuracy, on=['model', 'confidence_threshold'], how='left')
df_final_metrics = df_final_metrics[df_final_metrics['class'] != 0]
df_final_metrics.to_csv("final_metrics_with_fidelity_and_accuracy.csv", index=False)

# === Grafici PR curve per classe ===
model_colors = {
    'nano': 'tab:blue', 'small': 'black',
    'medium': 'tab:green', 'large': 'tab:red', 'xlarge': 'tab:orange'
}
model_markers = {
    'nano': 'o', 'small': 's', 'medium': '^', 'large': 'D', 'xlarge': 'X'
}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 25, 'axes.labelsize': 27, 'xtick.labelsize': 23,
    'ytick.labelsize': 23, 'legend.fontsize': 20, 'lines.linewidth': 1.5,
    'lines.markersize': 8, 'grid.linestyle': '--', 'grid.linewidth': 0.5,
    'grid.color': 'gray',
})

w, h = 7, 6
df_final_metrics['class'] = df_final_metrics['class'].astype(str)
class_ids = sorted(df_final_metrics['class'].unique(), key=int)
models = df_final_metrics['model'].unique()

cls_pr = {cls: {} for cls in class_ids}
all_R_cls = []
all_P_cls = []

for cls in class_ids:
    class_data = df_final_metrics[df_final_metrics['class'] == cls]
    for model in models:
        model_data = class_data[class_data['model'] == model].sort_values(by='recall_mean')
        R_cls = model_data['recall_mean'].tolist()
        P_cls = model_data['precision_mean'].tolist()
        if R_cls and P_cls:
            cls_pr[cls][model] = {'R_cls': R_cls, 'P_cls': P_cls}
            all_R_cls.extend(R_cls)
            all_P_cls.extend(P_cls)

x_min_c = min(all_R_cls) - 0.01
x_max_c = max(all_R_cls) + 0.01
y_min_c = floor(min(all_P_cls) * 20) / 20
y_max_c = ceil(max(all_P_cls) * 20) / 20

for cls in class_ids:
    fig, ax = plt.subplots(figsize=(w, h))
    for m, vals in cls_pr[cls].items():
        ax.plot(vals['R_cls'], vals['P_cls'],
                label=m, color=model_colors.get(m, 'gray'),
                marker=model_markers.get(m, 'o'))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(x_min_c, x_max_c)
    ax.set_ylim(y_min_c, y_max_c)
    ax.legend(title="YOLO Model", loc='best', ncol=2, handletextpad=0.5, frameon=True)
    plt.tight_layout(pad=0.8)
    plt.savefig(f"pr_curve_class_{cls}_ieee_TOTAL.pdf", dpi=300, bbox_inches="tight")
    plt.close()

# === Fidelity vs Threshold ===
plt.figure(figsize=(7, 6))
for model in model_map:
    model_df = df_fidelity_grouped[df_fidelity_grouped['model'] == model].sort_values(by='confidence_threshold')
    plt.plot(
        model_df['confidence_threshold'],
        model_df['fidelity_mean'],
        label=model,
        color=model_colors.get(model, 'gray'),
        marker=model_markers.get(model, 'o')
    )
plt.xlabel("Confidence Threshold")
plt.ylabel("Fidelity")
plt.grid(True, which='both', alpha=0.3)
plt.xticks(CONF_THRESH_LIST)
plt.ylim(0.8, 1.0)
plt.yticks(np.linspace(0.8, 1.0, 5))
plt.legend(title="YOLO Model", fontsize=20, title_fontsize=22, loc='upper left', ncol=2, handletextpad=0.5, frameon=True)
plt.tight_layout(pad=0.8)
plt.savefig("fidelity_vs_threshold_ieee.pdf", dpi=300, bbox_inches="tight")
plt.close()

# === Tabella finale formattata ===
df_small = df_final_metrics[df_final_metrics["model"] == "small"]
conf_list = sorted(df_small["confidence_threshold"].unique())
classes = sorted(df_small["class"].unique())

rows = []
headers = ["Conf. Thresh."]
for cls in classes:
    headers += [f"P{cls}", f"R{cls}", f"F1{cls}"]
headers += ["Fidelity", "Accuracy"]

for conf in conf_list:
    row = [f"{conf:.2f}"]
    conf_data = df_small[df_small["confidence_threshold"] == conf]

    for cls in classes:
        cls_data = conf_data[conf_data["class"] == cls]
        if not cls_data.empty:
            p = f"{cls_data['precision_mean'].values[0]:.2f} ± {cls_data['precision_std'].values[0]:.2f}"
            r = f"{cls_data['recall_mean'].values[0]:.2f} ± {cls_data['recall_std'].values[0]:.2f}"
            f1 = f"{cls_data['f1_mean'].values[0]:.2f} ± {cls_data['f1_std'].values[0]:.2f}"
        else:
            p = r = f1 = "-"
        row += [p, r, f1]

    fid = conf_data[["fidelity_mean", "fidelity_std"]].drop_duplicates()
    if not fid.empty:
        f_str = f"{fid['fidelity_mean'].values[0]:.2f} ± {fid['fidelity_std'].values[0]:.2f}"
    else:
        f_str = "-"
    row += [f_str]

    acc_data = conf_data[["accuracy_mean", "accuracy_std"]].drop_duplicates()
    if not acc_data.empty:
        acc_str = f"{acc_data['accuracy_mean'].values[0]:.2f} ± {acc_data['accuracy_std'].values[0]:.2f}"
    else:
        acc_str = "-"
    row += [acc_str]

    rows.append(row)

print(tabulate(rows, headers=headers, tablefmt="grid"))
