import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_THRESH_LIST = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
IOU_THRES = 0.50
IOU_OK_THRESH = 0.50

model_map = {
    'nano': 'n',
    'small': 's',
    'medium': 'm',
    'large': 'l',
    'xlarge': 'x'
}

class_ids = [1, 2, 3]

table_data_per_threshold = {}
classes_per_threshold = {}

seg_pr = {m: {'th': [], 'P_seg': [], 'R_seg': []} for m in model_map}

cls_pr = {
    cls: {m: {'th': [], 'P_cls': [], 'R_cls': []} for m in model_map}
    for cls in class_ids
}

for CONF_THRES in CONF_THRESH_LIST:
    thresh_folder = (
        f"conf{CONF_THRES:.2f}_iou{IOU_THRES:.2f}_ok{IOU_OK_THRESH:.2f}"
        .replace("0.", "")
    )
    BASE_DIR = os.path.join(SCRIPT_DIR, 'B_Y11_inference', thresh_folder)

    class_records = []
    seg_records = []

    for model_name, short in model_map.items():
        idx = list(model_map.keys()).index(model_name) + 1
        run_folder = os.path.join(BASE_DIR, f"{idx}_runs_{model_name}")
        if not os.path.isdir(run_folder):
            continue

        for fold in range(1, 6):
            fold_folder = f"train{fold}"
            c_int = int(CONF_THRES * 100)
            csv_name = f"metrics_Y11{short}{fold}_c{c_int}.csv"
            csv_path = os.path.join(run_folder, fold_folder, csv_name)
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path, sep=';')

            y_true = df['true_class'].to_numpy()
            y_pred = df['Y11_pred_class'].to_numpy()
            classes = sorted({int(x) for x in np.concatenate([y_true, y_pred]) if x != 0})

            precisions = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
            recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
            f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

            for cls, p, r, f in zip(classes, precisions, recalls, f1s):
                class_records.append({
                    'model': model_name,
                    'fold': fold,
                    'class_id': cls,
                    'precision': p,
                    'recall': r,
                    'f1': f
                })

            iou_values = df['Y11_IoU'].to_numpy()
            good_seg = (iou_values >= IOU_OK_THRESH)
            TP = int(good_seg.sum())
            FP = int(((~good_seg) & (iou_values > 0)).sum())
            FN = int(((~good_seg) & (iou_values == 0)).sum())

            precision_seg = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall_seg = TP / (TP + FN) if (TP + FN) > 0 else np.nan

            seg_records.append({
                'model': model_name,
                'fold': fold,
                'IoU': iou_values.mean(),
                'Dice': df['Y11_Dice'].mean(),
                'P_seg': precision_seg,
                'R_seg': recall_seg
            })

    df_cls = pd.DataFrame(class_records)
    df_seg = pd.DataFrame(seg_records)

    if df_cls.empty:
        agg_cls = pd.DataFrame(columns=[
            'model', 'class_id', 'p_mean', 'p_std', 'r_mean', 'r_std', 'f_mean', 'f_std'
        ])
    else:
        agg_cls = (
            df_cls
            .groupby(['model', 'class_id'])
            .agg(
                p_mean=('precision', 'mean'),
                p_std=('precision', 'std'),
                r_mean=('recall', 'mean'),
                r_std=('recall', 'std'),
                f_mean=('f1', 'mean'),
                f_std=('f1', 'std')
            )
            .reset_index()
        )

    agg_seg = (
        df_seg
        .groupby('model')
        .agg(
            IoU_mean=('IoU', 'mean'),
            IoU_std=('IoU', 'std'),
            Dice_mean=('Dice', 'mean'),
            Dice_std=('Dice', 'std'),
            P_seg_mean=('P_seg', 'mean'),
            P_seg_std=('P_seg', 'std'),
            R_seg_mean=('R_seg', 'mean'),
            R_seg_std=('R_seg', 'std')
        )
        .reset_index()
    )

    for _, row in agg_seg.iterrows():
        m = row['model']
        seg_pr[m]['th'].append(CONF_THRES)
        seg_pr[m]['P_seg'].append(row['P_seg_mean'])
        seg_pr[m]['R_seg'].append(row['R_seg_mean'])

    for _, row in agg_cls.iterrows():
        m = row['model']
        cls = row['class_id']
        if cls in cls_pr:
            cls_pr[cls][m]['th'].append(CONF_THRES)
            cls_pr[cls][m]['P_cls'].append(row['p_mean'])
            cls_pr[cls][m]['R_cls'].append(row['r_mean'])

    classes_all = sorted(df_cls['class_id'].unique())
    headers = ['Model', 'IoU', 'Dice', 'P_seg', 'R_seg']
    for cls in classes_all:
        headers += [f'P_{cls}', f'R_{cls}', f'F1_{cls}']

    table_data = []
    for m in model_map:
        if m not in agg_seg['model'].values:
            continue
        seg = agg_seg.set_index('model').loc[m]
        row = [
            m,
            f"{seg.IoU_mean:.2f}±{seg.IoU_std:.2f}",
            f"{seg.Dice_mean:.2f}±{seg.Dice_std:.2f}",
            f"{seg.P_seg_mean:.2f}±{seg.P_seg_std:.2f}",
            f"{seg.R_seg_mean:.2f}±{seg.R_seg_std:.2f}"
        ]
        if not agg_cls.empty and m in agg_cls['model'].values:
            sub = agg_cls[agg_cls['model'] == m].set_index('class_id')
            for c in classes_all:
                if c in sub.index:
                    pm, ps = sub.loc[c, ['p_mean', 'p_std']]
                    rm, rs = sub.loc[c, ['r_mean', 'r_std']]
                    fm, fs = sub.loc[c, ['f_mean', 'f_std']]
                    row += [f"{pm:.2f}±{ps:.2f}", f"{rm:.2f}±{rs:.2f}", f"{fm:.2f}±{fs:.2f}"]
                else:
                    row += ['-', '-', '-']
        else:
            row += ['-', '-', '-'] * len(classes_all)
        table_data.append(row)

    table_data_per_threshold[CONF_THRES] = table_data
    classes_per_threshold[CONF_THRES] = classes_all
    print(f"\nSUMMARY FOR CONF_THRES = {CONF_THRES:.2f}")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
