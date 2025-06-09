# B_Work - Hybrid AI System Execution and Results

This folder contains all core scripts, training outputs, and evaluation results
for the hybrid AI pipeline described in the manuscript.

## Structure

The `B_Work/` directory contains the main operational logic and experimental outputs of the hybrid AI pipeline described in the manuscript. This folder is organized into four main blocks, each responsible for a distinct phase of the system lifecycle:

- `A_Y11/` –-> Focuses on experiments related to **training and evaluation of ** models on MRI images. It includes training scripts, weights trained on different dimensions and folds of the model, and post-processing metrics. All scripts in this block are executed in **Environment A** (Python 3.12).

- `B_Fuzzy_tree/` –-> Contains scripts, models, and results for **Fuzzy Decision Tree models**. This block is executed in **Environment B** (Python 3.8 – 32 bit).

- `C_Radiomics_Y11/` -–> This section deals with the **extraction of radiomic features from ROIs segmented by YOLOv11**, for each combination of model and threshold. It then uses those features to run **inference via FDT**. The final results, aggregated by model and configuration, are saved in JSON format. The scripts in this section must be run in **Environment B** for radiomic feature extraction and **Environment C** for FDT inference.

- `Z_dashboard/` –-> Optional scripts and tools to visualize the results, explore segmentation/classification performance, and enable qualitative inspection of predictions.

```text
B_Work/
├── A_Y11/                                # YOLOv11 training, inference, and metrics
│   ├── A_train_Y11_MRI.py                # Train YOLOv11 on MRI images
│   ├── B_Compute_inference.py            # Run YOLOv11 inference
│   ├── C1_Compute_metrics_complexity.py  # Analyze mask shape complexity
│   ├── C2_Compute_metrics_v2.py          # Compute classification metrics
│
│   ├── A_Y11_Models/
│   │   ├── 1_runs_nano/
│   │   ├── 2_runs_small/
│   │   ├── 3_runs_medium/
│   │   ├── 4_runs_large/
│   │   └── 5_runs_xlarge/
│   │       └── train{n}/weights/         # YOLOv11 weights for each fold
│
│   ├── B_Y11_inference/
│   │   ├── conf25_iou45_ok45/
│   │   ├── conf30_iou50_ok50/
│   │   ├── ...
│   │   └── conf60_iou50_ok50/
│   │       └── {size}/train{n}/          # Contains predicted masks and bounding boxes
│
│   ├── Paper_results/                    # Aggregated results used in the manuscript
│   └── Z_OLD_AND_DEBUG/                  # Legacy or debug scripts and results
│
├── B_Fuzzy_tree/                         # FDT training and explainability pipeline
│   ├── A_OK_train_fuzzy_tree_v4.py       # Train FDTs on radiomic features
│   ├── B_OK_generate_latex_table.py      # Generate LaTeX tables from results
│
│   ├── Results/
│   ├── A_Models/                         # Trained FDT models
│   ├── B_Json_CrossVal/                  # JSONs from cross-validation
│   ├── C_Rules/                          # Extracted fuzzy decision rules
│   ├── D_Test_and_Rules/                 # Rule-based predictions and evaluation per fold
│   └── OLD/                              # Legacy scripts and unused experiments
│
├── C_Radiomics_Y11/                      # Integration of YOLO & Fuzzy logic
│   ├── C_Radiomics_Y11.py                # Extract radiomic features from YOLO-identified ROIs
│   ├── D_Radiomics_Y11_fuzz.py           # Perform FDT inference on radiomics extracted from YOLO
│
│   ├── conf25_iou50_ok50/
│   ├── conf30_iou50_ok50/
│   ├── ...
│   └── conf60_iou50_ok50/
│       ├── 1_Y11_nano/
│       ├── ...
│       └── Results/B_Json/              # Comparative per-fold outputs in JSON
│
├── E_Auto_PR_v2.py                      # Aggregation of results and drawing of PR curves

```
-----------------------------------------------
## Before You Start: Download Required Large Files

The folders `A_Y11_Models/` (containing the trained YOLOv11 models) and `B_Y11_inference/` (containing all inference results across thresholds and model sizes)
are too large to be stored directly in this repository.

To proceed:

1. Download the ZIP archive containing both folders from the following link:
   ➤ [Insert download link here]

2. Extract the archive.

3. Place the extracted folders in their correct location inside the `B_Work/` directory,
   so that the structure matches the one described in the README.

**IMPORTANT: These files are essential for running evaluation scripts and generating results
without re-training the models from scratch.**

-----------------------------------------------

# Execution Order

The following is the recommended order for executing the complete hybrid AI pipeline. Each block must be executed in its specific Python environment, and the scripts should be run in the listed sequence.

Overview by Block:
- `A_Y11/` : Train YOLOv11 models, run inference, and compute segmentation/classification metrics.
- `B_Fuzzy_tree/` : Train interpretable Fuzzy Decision Tree models and export evaluation tables.
- `C_Radiomics_Y11/` : Extract radiomic features from YOLO-segmented regions and run FDT inference.

## A_Y11 – YOLOv11 Training and Evaluation

**Commands to execute in *Environment A*:**

    # Train YOLOv11 on MRI images (requires adjusting dataset path in the script and significant computational resources)
    # Experiments were conducted on a server with 2× Intel Xeon Platinum 8480+ CPUs (112 cores total), 2TB RAM, and 3× NVIDIA A100 80GB GPUs (CUDA 12.3, Ubuntu 22.04.3)
    python A_train_Y11_MRI.py

    # Run inference using trained YOLOv11 models
    python B_Compute_inference.py
    
    # Compute model complexity metrics
    python C1_Compute_metrics_complexity.py
        
    # Compute classification and segmentation performance metrics
    python C2_Compute_metrics_v2.py

Output directories:

- A_Y11_Models/      : Trained YOLOv11 model weights (organized by fold and model size)
- B_Y11_inference/   : Segmentation masks and annotated images with confidence scores (for visualization and debugging)

## B_Fuzzy_tree – Fuzzy Tree Training and Evaluation

**Commands to execute in Environment B (Python 3.8 – 32 bit)**:

    # Train Fuzzy Decision Trees using handcrafted radiomic features
    python A_OK_train_fuzzy_tree_v4.py

    # Generate LaTeX tables summarizing evaluation metrics
    python B_OK_generate_latex_table.py

Output directories:

- A_Models/           : Trained FDT models for each fold
- B_Json_CrossVal/    : JSON outputs from cross-validation (predictions and metrics)
- C_Rules/            : Extracted fuzzy decision rules used for interpretability
- D_Test_and_Rules/   : Test predictions with corresponding applied rules
- Results/            : Consolidated outputs used for evaluation and analysis
- OLD/                : Legacy scripts and deprecated experiments


## C_Radiomics_Y11 – Radiomics Extraction and FDT Inference on YOLO Regions

Commands to execute:

    # Extract radiomic features from YOLO-segmented ROIs
    # (Environment C – Python 3.8)
    python C_Radiomics_Y11.py

    # Perform Fuzzy Decision Tree inference on extracted radiomic features
    # (Environment B – Python 3.8, 32 bit)
    python D_Radiomics_Y11_fuzz.py

Output directories:

- confXX_iouYY_okZZ/1_Y11_nano/ to 5_Y11_xlarge/ : Results grouped by YOLO model size and threshold
- confXX_iouYY_okZZ/Results/B_Json/              : Final JSON outputs with predictions and rule-based classification results

## E_Auto_PR_v2.py  
**Commands to execute in Environment A**:

    python E_Auto_PR_v2.py
