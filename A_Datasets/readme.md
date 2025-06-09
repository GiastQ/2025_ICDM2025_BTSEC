A_Datasets - Dataset Overview
=============================

This folder contains all dataset components required to train and validate both modules of the hybrid AI system:

- YOLOv11 for segmentation and classification
- Fuzzy Decision Tree (FDT) for rule-based explainable classification

Structure
---------
```text
A_Datasets/
├── A_Train_Yolo/                       # Data for YOLOv11 training and validation
│   ├── fold_data/
│   │   ├── round_1/
│   │   │   ├── Train/
│   │   │   │   ├── images/
│   │   │   │   ├── labels/
│   │   │   │   └── masks/
│   │   │   ├── Validation/
│   │   │   │   ├── images/
│   │   │   │   ├── labels/
│   │   │   │   └── masks/
│   │   │   └── Test/
│   │   │       ├── images/
│   │   │       ├── labels/
│   │   │       └── masks/
│   │   ├── round_2/
│   │   ├── round_3/
│   │   ├── round_4/
│   │   └── round_5/
│   └── B_YOLO_Labelling.py            # Script for preparing or refining YOLO labels
│
├── B_Train_Fuzzy/
│   ├── A0_Fold_Index/                 # Fold-wise train/test indices
│   ├── A_XY_cooridnates_v5/           # Radiomic features and ROI metadata (Dataset for FDT trains)
│   ├── Human_tagging.xlsx             # Manual tagging metadata (Anatomical plane which the ROI is located)
│   └── Radiomics_Extraction.py        # Script for radiomics feature extraction (From original MRI images with Ground truth mask)


A_Train_Yolo - Segmentation and Classification Data
```
----------------------------------------------------

This folder contains 5 rounds (folds) of pre-split data used for YOLOv11-based detection and segmentation.

Each round includes:
- Train/: training samples
- Validation/: validation samples
- Test/: testing samples

Each subset includes:
- images/: cropped MRI images (.jpg)
- labels/: YOLO-format annotation files (.txt)
- masks/: binary segmentation masks (.jpg)


B_Train_Fuzzy - Data for Explainable FDT Classifier
---------------------------------------------------

This subfolder provides the data required for training and evaluating the FDT.

- A0_Fold_Index/: defines training/testing dataset partitions for cross-validation.
- A_XY_cooridnates_v5/: contains extracted radiomic features and spatial metadata.

These are used by scripts in B_Work/B_Fuzzy_tree/.

Usage in the Project
--------------------

- A_Train_Yolo/ is used by:
  - B_Work/A_Y11/ for YOLOv11 training
  - B_Work/A_Y11/B_Y11_inference/ for predictions

- B_Train_Fuzzy/ is used by:
  - B_Work/B_Fuzzy_tree/ for training the FDT


B_YOLO_Labelling.py - Mask Conversion Script
--------------------------------------------

The original dataset provides binary segmentation masks (foreground/background)
for each Region of Interest.

To use these masks in the YOLOv11 training pipeline, we need to convert them
into bounding box annotations in YOLO format.

This conversion is performed automatically by the script: B_YOLO_Labelling.py

This script scans the "masks/" folders, detects the binary mask content,
and creates YOLO-format .txt files inside the "labels/" folders.
The generated bounding boxes are axis-aligned rectangles enclosing the masks.

To run the script, use **Environment A** (Python 3.12):
```text
  python B_YOLO_Labelling.py
```

Radiomics_Extraction.py - Creating a tabular dataset from images
--------------------------------------------
To train FDT trees, it is necessary to extract radiomic features from MRI images.
The ROIs from which these features are computed are obtained by superimposing the MRI images onto the corresponding truth masks.

The creation of the tabular dataset is then achieved by executing: Radiomics_Extraction.py

To run the script, use **Environment C** (Python 3.8):
```text
  python Radiomics_Extraction.py
```

Original Dataset
----------------

The data used in this repository is based on the publicly available
"Brain Tumor Dataset" from Figshare:

  https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

Notes
-----

- Do NOT change folder or file names unless you update the scripts accordingly.
- Each image has a corresponding label and mask with the same prefix.
- Data is pre-split for 5-fold cross-validation.