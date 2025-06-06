import os
import sys
import pickle
import numpy as np
import pandas as pd
import json
from collections import Counter
from sklearn.metrics import classification_report

# === Step 1: Add path to FuzzyML library ===
sys.path.append(r"C:\Users\Giustino\Desktop\CARTELLA_LAVORO\ExP_PIETRO\Albero_Fuzzy_libreria_OK")
from FuzzyML.fmdt import fmdt

# === Step 2: Main function to generate predictions and rule activations ===
def generate_test_and_rules(
    model_path: str,
    test_csv_path: str,
    assoc_csv_path: str,
    output_dir: str,
    feature_columns: list,
    experiment_name: str,
    round_number: int,
    max_depth: int,
    num_fuzzy_sets: int,
    base_dir: str,
    conf_thres: float,
    iou_thres: float,
    iou_ok_thres: float
):
    # Load fuzzy decision tree model
    with open(model_path, "rb") as mf:
        tree: fmdt.FMDT = pickle.load(mf)

    # Load test data and association data
    test_df  = pd.read_csv(test_csv_path)
    assoc_df = pd.read_csv(assoc_csv_path)

    X_test = test_df[feature_columns].values
    y_true = test_df["true_class"].values

    ZERO_THRESHOLD = 5
    records = []

    for idx, x in enumerate(X_test):
        zero_count = np.sum(x == 0)

        if zero_count > ZERO_THRESHOLD:
            pred_class = 0
            activated_rule = "skipped"
            classwise_rules = {}
        else:
            pred_class, activated_rule, top_rules = tree.classify_with_rule(x, top_per_class=True)

            # Build per-class rule/match dictionary
            classwise_rules = {
                f"class_{cls}_rule": rule_str
                for cls, rule_str, _ in top_rules
            }
            classwise_rules.update({
                f"class_{cls}_match": match_score
                for cls, _, match_score in top_rules
            })

        true_class = y_true[idx]

        row = {
            "index": idx,
            "image_id": assoc_df.loc[idx, "IDfile"],
            "patient_id": assoc_df.loc[idx, "IDpersona"],
            "true_class": int(true_class),
            "fuzY11_pred_class": int(pred_class),
            "activated_ruleY11": activated_rule
        }

        row.update(classwise_rules)  # Add per-class rule info
        records.append(row)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{experiment_name}_{round_number}_d{max_depth}_t{num_fuzzy_sets}_test_and_rules.csv"
    out_path = os.path.join(output_dir, filename)
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"Saved test+rules CSV: {out_path}")

    # Summary statistics
    skipped_count = np.sum(df["fuzY11_pred_class"] == 0)
    accuracy = np.mean(df["fuzY11_pred_class"] == df["true_class"])
    rule_counts = Counter(df[df["activated_ruleY11"] != "skipped"]["activated_ruleY11"])
    rule_counts = dict(sorted(rule_counts.items()))

    clf_report = classification_report(
        df["true_class"],
        df["fuzY11_pred_class"],
        output_dict=True,
        zero_division=0
    )

    summary = {
        "experiment_name": experiment_name,
        "round_number": round_number,
        "max_depth": max_depth,
        "num_fuzzy_sets": num_fuzzy_sets,
        "num_test_samples": len(df),
        "num_skipped_samples": int(skipped_count),
        "accuracy": accuracy,
        "rule_activation_counts": rule_counts,
        "classification_report": clf_report
    }

    # Save summary JSON
    thresh_folder = f"conf{conf_thres:.2f}_iou{iou_thres:.2f}_ok{iou_ok_thres:.2f}".replace("0.", "")
    json_output_dir = os.path.join(
        base_dir, "B_Work", "C_Radiomics_Y11", thresh_folder, "Results", "B_Json"
    )
    os.makedirs(json_output_dir, exist_ok=True)

    json_filename = f"{experiment_name}_{round_number}_d{max_depth}_t{num_fuzzy_sets}_test_summary.json"
    json_path = os.path.join(json_output_dir, json_filename)

    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=4)

    print(f"Saved summary JSON: {json_path}")


# === Step 7: Main execution block ===
if __name__ == "__main__":
    # === Configuration ===
    CONF_THRES_LIST = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    IOU_THRES = 0.50
    IOU_OK_THRESH = 0.50

    EXPERIMENT_NAME = "ICDM"
    MAX_DEPTH = 5
    NUM_FUZZY_SETS = 5

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

    # === Features used for classification ===
    FEATURE_COLUMNS = [
        "original_shape2D_Elongation",
        "original_shape2D_MajorAxisLength",
        "original_shape2D_MinorAxisLength",
        "original_shape2D_Perimeter",
        "original_shape2D_MaximumDiameter",
        "original_shape2D_Sphericity",
        "original_firstorder_Mean",
        "original_firstorder_Median",
        "original_firstorder_Minimum",
        "original_firstorder_Maximum",
        "original_firstorder_Range",
        "original_firstorder_Uniformity",
        "Centroid_X",
        "Centroid_Y",
        "plane"
    ]

    # === Model versions to evaluate ===
    MODEL_MAP = {
        "nano": "n",
        "small": "s",
        "medium": "m",
        "large": "l",
        "xlarge": "x"
    }

    # === Step 8: Iterate over thresholds, models, and folds ===
    for conf_thres in CONF_THRES_LIST:
        thresh_folder = f"conf{conf_thres:.2f}_iou{IOU_THRES:.2f}_ok{IOU_OK_THRESH:.2f}".replace("0.", "")

        for model_idx, (model_name, model_suffix) in enumerate(MODEL_MAP.items(), 1):
            model_folder = f"{model_idx}_Y11_{model_name}"

            for fold in range(1, 6):  # 5 folds
                test_csv_name = f"{fold}_test_y11{model_suffix}.csv"

                test_csv_path = os.path.join(
                    BASE_DIR, "B_Work", "C_Radiomics_Y11",
                    thresh_folder, model_folder,
                    test_csv_name
                )

                assoc_csv_path = os.path.join(
                    BASE_DIR, "A_Datasets", "B_Train_Fuzzy", "A0_Fold_Index",
                    f"round_{fold}_test.csv"
                )

                model_path = os.path.join(
                    BASE_DIR, "B_Work", "B_Fuzzy_tree", "Results", "A_Models",
                    f"{EXPERIMENT_NAME}_model_r{fold}_"
                    f"d{MAX_DEPTH}_t{NUM_FUZZY_SETS}_optim_False_pre_False_minex_25.pkl"
                )

                output_dir = os.path.join(
                    BASE_DIR, "B_Work", "C_Radiomics_Y11",
                    thresh_folder, model_folder
                )

                # === Step 9: Run fuzzy classification ===
                generate_test_and_rules(
                    model_path=model_path,
                    test_csv_path=test_csv_path,
                    assoc_csv_path=assoc_csv_path,
                    output_dir=output_dir,
                    feature_columns=FEATURE_COLUMNS,
                    experiment_name=EXPERIMENT_NAME,
                    round_number=fold,
                    max_depth=MAX_DEPTH,
                    num_fuzzy_sets=NUM_FUZZY_SETS,
                    base_dir=BASE_DIR,
                    conf_thres=conf_thres,
                    iou_thres=IOU_THRES,
                    iou_ok_thres=IOU_OK_THRESH
                )
