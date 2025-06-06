import os
import sys
import time
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# === Custom modules import ===
sys.path.append(r"C:\Users\Giustino\Desktop\CARTELLA_LAVORO\ExP_PIETRO\Albero_Fuzzy_libreria_OK")
from FuzzyML.fmdt import fmdt
from FuzzyML.discretization.discretizer_base import fuzzyDiscretization
from FuzzyML.discretization import discretizer_fuzzy

# === Global parameters and configuration ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

experiment_name    = "ICDM"
file_tag_prefix    = "XY_v5_"

# === Define feature columns and discrete flags ===
features_and_is_categorical = (
    ("original_shape2D_Elongation",      False),  # A0
    ("original_shape2D_MajorAxisLength", False),  # A1
    ("original_shape2D_MinorAxisLength", False),  # A2
    ("original_shape2D_Perimeter",       False),  # A3
    ("original_shape2D_MaximumDiameter", False),  # A4
    ("original_shape2D_Sphericity",      False),  # A5
    ("original_firstorder_Mean",         False),  # A6
    ("original_firstorder_Median",       False),  # A7
    ("original_firstorder_Minimum",      False),  # A8
    ("original_firstorder_Maximum",      False),  # A9
    ("original_firstorder_Range",        False),  # A10
    ("original_firstorder_Uniformity",   False),  # A11
    ("Centroid_X",                       False),  # A12
    ("Centroid_Y",                       False),  # A13
    ("plane",                             True),  # A14
)
column_names      = [col for col, _ in features_and_is_categorical]
is_continuous     = [not is_cat for _, is_cat in features_and_is_categorical]

# Flags
auto_mode        = True     # If False, uses single settings
save_json        = True
save_model       = True
save_rules       = True

min_examples     = 25
use_preprocessing = False
use_optimization  = False

# Single values if auto_mode == False
max_depth_single      = 3
num_fuzzy_sets_single = 5

# Lists of values if auto_mode == True
max_depth_options      = [3, 5, 7, len(column_names)]
num_fuzzy_sets_options = [3, 5]

# === Base paths (all English folder names) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
base_data_dir    = os.path.join(base_dir ,"A_Datasets", "B_Train_Fuzzy", "A_XY_cooridnates_v5")
base_output_dir  = os.path.join(base_dir, "B_Work", "B_Fuzzy_tree", "Results")
json_dir         = os.path.join(base_output_dir, "B_Json_CrossVal")
model_dir        = os.path.join(base_output_dir, "A_Models")
rule_dir         = os.path.join(base_output_dir, "C_Rules")
test_rule_dir    = os.path.join(base_output_dir, "D_Test_and_Rules")
assoc_dir        = os.path.join(base_dir, "A_Datasets", "B_Train_Fuzzy", "A0_Fold_Index")

# Create output directories if needed
for directory in (json_dir, model_dir, rule_dir, test_rule_dir):
    os.makedirs(directory, exist_ok=True)

def run_experiment(max_depth, num_fuzzy_sets):
    print(f"\n=== EXPERIMENT: depth={max_depth}, fuzzy_sets={num_fuzzy_sets} ===")
    # Containers for results
    results = {
        "rounds": [],
        "avg_train": {},
        "avg_test": {},
        "avg_rule_length": 0,
        "std_rule_length": 0,
        "avg_num_nodes": 0,
        "std_num_nodes": 0,
        "avg_max_depth": 0,
        "std_max_depth": 0,
        "avg_num_leaves": 0,
        "std_num_leaves": 0,
        "avg_num_parameters": 0,
        "std_num_parameters": 0,
    }
    rule_lengths = []
    node_counts  = []
    leaf_counts  = []
    depths_reached = []
    parameter_counts = []
    models       = {}

    # Optional global preprocessing
    if use_preprocessing:
        all_test_sets = []
        for i in range(1, 6):
            fp = os.path.join(base_data_dir, f"{i}_{file_tag_prefix}test.csv")
            all_test_sets.append(pd.read_csv(fp))
        X_all = pd.concat(all_test_sets)[column_names]
        lower_bounds = X_all.quantile(0.05)
        upper_bounds = X_all.quantile(0.95)
        scaler = MinMaxScaler().fit(X_all)

    # Five-fold cross-validation
    for fold in range(1, 6):
        print(f"\n--- FOLD {fold} ---")
        # 1) Load training and test data
        train_fp = os.path.join(base_data_dir, f"{fold}_{file_tag_prefix}train.csv")
        test_fp  = os.path.join(base_data_dir, f"{fold}_{file_tag_prefix}test.csv")
        train_df = pd.read_csv(train_fp)
        test_df  = pd.read_csv(test_fp)

        # 2) Load associations (file ID, subject ID)
        assoc_fp = os.path.join(assoc_dir, f"round_{fold}_test.csv")
        assoc_df = pd.read_csv(assoc_fp)

        # Optional clipping and scaling
        if use_preprocessing:
            train_df[column_names] = train_df[column_names].clip(lower_bounds, upper_bounds, axis=1)
            test_df[column_names]  = test_df[column_names].clip(lower_bounds, upper_bounds, axis=1)
            train_df[column_names] = scaler.transform(train_df[column_names])
            test_df[column_names]  = scaler.transform(test_df[column_names])

        # Prepare arrays
        X_train, y_train = train_df[column_names].values, train_df["true_class"].values
        X_test,  y_test  = test_df[column_names].values,  test_df["true_class"].values
        num_classes = len(np.unique(y_train))

        # Discretization
        if use_optimization:
            fd = discretizer_fuzzy.FuzzyMDLFilter(
                num_classes, X_train, y_train, is_continuous, threshold=num_fuzzy_sets
            )
            splits = fd.run()
        else:
            fd = fuzzyDiscretization(num_fuzzy_sets, method="uniform")
            splits = fd.run(X_train, is_continuous)

        # Training
        start_time = time.time()
        tree = fmdt.FMDT(
            max_depth=max_depth,
            minNumExamples=min_examples,
            priorDiscretization=True,
            verbose=True,
            features='all'
        )
        tree.fit(
            X_train, y_train,
            continous=is_continuous,
            cPoints=splits,
            ftype="triangular"
        )
        train_duration = time.time() - start_time
        models[fold] = tree

        # Predictions and reports
        pred_train = tree.predict(X_train)
        pred_test  = tree.predict(X_test)
        acc_train  = np.mean(pred_train == y_train)
        acc_test   = np.mean(pred_test  == y_test)
        report_train = classification_report(y_train, pred_train, output_dict=True)
        report_test  = classification_report(y_test,  pred_test,  output_dict=True)
        report_train["accuracy"] = acc_train
        report_test["accuracy"]  = acc_test

        # Tree statistics
        rl     = tree.totalRuleLength()
        nn     = tree.numNodes()
        nl     = tree.numLeaves()
        md     = tree.get_max_depth()
        nparam = tree.num_parameters()

        # Record results for this fold
        results["rounds"].append({
            "fold": fold,
            "train_report": report_train,
            "test_report": report_test,
            "rule_length": rl,
            "num_nodes": nn,
            "num_leaves": nl,
            "num_features_used": tree.features_used(),
            "num_parameters": nparam,
            "max_depth_reached": md,
            "configured_max_depth": max_depth,
            "num_fuzzy_sets": num_fuzzy_sets,
            "training_time_sec": train_duration
        })

        rule_lengths.append(rl)
        node_counts.append(nn)
        leaf_counts.append(nl)
        depths_reached.append(md)
        parameter_counts.append(nparam)

        print(
            f"Fold {fold}: time={train_duration:.2f}s, "
            f"acc_train={acc_train:.4f}, acc_test={acc_test:.4f}, "
            f"params={nparam}, nodes={nn}, leaves={nl}, depth={md}"
        )

        # Save per-fold test predictions and activated rules
        records = []
        for idx, x in enumerate(X_test):
            pred, rule = tree.classify_with_rule(x)
            records.append({
                "index":            idx,
                "image_id":          assoc_df.loc[idx, "IDfile"],
                "patient_id":       assoc_df.loc[idx, "IDpersona"],
                "true_class":       y_test[idx],
                "fuzzy_prediction": pred,
                "activated_rule":   rule
            })
        out_df = pd.DataFrame(records)
        out_fn = f"{experiment_name}_fold{fold}_depth{max_depth}_fsets{num_fuzzy_sets}_test_rules.csv"
        out_df.to_csv(os.path.join(test_rule_dir, out_fn), index=False)
        print(f"Saved test+rules CSV: {out_fn}")

    # Aggregate statistics
    def mean_std(lst):
        return np.mean(lst), np.std(lst)

    results["avg_rule_length"], results["std_rule_length"]       = mean_std(rule_lengths)
    results["avg_num_nodes"],    results["std_num_nodes"]        = mean_std(node_counts)
    results["avg_max_depth"],    results["std_max_depth"]        = mean_std(depths_reached)
    results["avg_num_leaves"],   results["std_num_leaves"]       = mean_std(leaf_counts)
    results["avg_num_parameters"],results["std_num_parameters"]  = mean_std(parameter_counts)

    # Compute average train/test metrics across folds
    train_metrics = defaultdict(list)
    test_metrics  = defaultdict(list)
    for entry in results["rounds"]:
        for label, metrics in entry["train_report"].items():
            train_metrics[label].append(metrics)
        for label, metrics in entry["test_report"].items():
            test_metrics[label].append(metrics)

    def compute_avg_std_dict(metric_dict):
        output = {}
        for label, vals in metric_dict.items():
            if isinstance(vals[0], dict):
                output[label] = {}
                for m in vals[0]:
                    arr = [v[m] for v in vals]
                    output[label][f"{m}_mean"] = np.mean(arr)
                    output[label][f"{m}_std"]  = np.std(arr)
            else:
                output[f"{label}_mean"] = np.mean(vals)
                output[f"{label}_std"]  = np.std(vals)
        return output

    results["avg_train"] = compute_avg_std_dict(train_metrics)
    results["avg_test"]  = compute_avg_std_dict(test_metrics)

    # Save outputs: JSON summary, models, and rule text files
    if save_json:
        json_fn = (
            f"{experiment_name}_results_"
            f"d{max_depth}_t{num_fuzzy_sets}"
            f"_optim_{use_optimization}"
            f"_pre_{use_preprocessing}"
            f"_minex_{min_examples}.json"
        )
        with open(os.path.join(json_dir, json_fn), "w") as jf:
            json.dump(results, jf, indent=4)
        print(f"Saved JSON summary: {json_fn}")

    if save_model:
        for rnd, mdl in models.items():
            mdl_fn = (
                f"{experiment_name}_model_r{rnd}_"
                f"d{max_depth}_t{num_fuzzy_sets}"
                f"_optim_{use_optimization}"
                f"_pre_{use_preprocessing}"
                f"_minex_{min_examples}.pkl"
            )
            with open(os.path.join(model_dir, mdl_fn), "wb") as mf:
                pickle.dump(mdl, mf)
        print(f"Saved models with pattern: {experiment_name}_model_r*_d{max_depth}_t{num_fuzzy_sets}.pkl")

    if save_rules:
        for rnd, mdl in models.items():
            rules_fn = (
                f"{experiment_name}_rules_r{rnd}_"
                f"d{max_depth}_t{num_fuzzy_sets}"
                f"_optim_{use_optimization}"
                f"_pre_{use_preprocessing}"
                f"_minex_{min_examples}.txt"
            )
            with open(os.path.join(rule_dir, rules_fn), "w") as rf:
                for rule_text in mdl.show_rules(processed=True):
                    rf.write(rule_text + "\n")
        print(f"Saved rule files with pattern: {experiment_name}_rules_r*_d{max_depth}_t{num_fuzzy_sets}.txt")

# === Main entry point ===
if auto_mode:
    for depth in max_depth_options:
        for fsets in num_fuzzy_sets_options:
            run_experiment(depth, fsets)
else:
    run_experiment(max_depth_single, num_fuzzy_sets_single)
