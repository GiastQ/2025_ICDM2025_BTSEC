# A Hybrid AI-based System for Brain Tumor Segmentation and Explainable Classification from MRI Images (Under Review)

This repository accompanies the paper "A Hybrid AI-based System for Brain Tumor Segmentation and Explainable Classification from MRI Images", currently under peer review. It includes the full codebase, scripts, and instructions necessary to fully reproduce the experiments and results reported in the manuscript.

## Table of Contents

- [Abstract](#abstract)
- [Repository Folder Structure](#repository-folder-structure)
- [Prerequisites](#prerequisites)
  - [Environment A – Python 3.12](#environment-a--python-312)
  - [Environment B – Python 3.8 (32-bit)](#environment-b--python-38-32-bit)
  - [Environment C – Python 3.8](#environment-c--python-38)
- [Execution of the Codex](#execution-of-the-codex)


## Abstract

We present a hybrid AI system for brain tumor segmentation and explainable classification from MRI images. The framework combines YOLOv11 for automatic Region of Interest (ROI) detection and classification with a Fuzzy Decision Tree (FDT) that operates on handcrafted radiomic features. Each ROI is classified along two parallel paths: one based on YOLOv11 and the other on interpretable rule-based reasoning derived from the FDT. A Decision Support module compiles this information into an Image Analysis Report, exposing agreements and divergences between the two classification models. This enables human-in-the-loop decision-making and promotes transparency. Experiments on a public dataset show that the system provides competitive accuracy while maintaining explainability. Disagreement analyses confirm that most divergences reflect soft boundaries between classes, supporting trust in the system's outputs.

## Authors

Anonymous authors  
(Author details are withheld for double-blind review.)

## Paper Status

This work has been submitted and is currently under peer review.

## Repository Folder Structure

For more details, see the README inside each folder.

```text
A_Datasets/       Contains all datasets, organized in folds  
B_Work/           Contains all scripts, models, and results
```

Prerequisites
=============

First of all: bring patience – reproducing the full pipeline is a thorough process!

The system requires three distinct Python environments due to dependency constraints across different stages.


Environment A – Python 3.12
---------------------------

Used for training and validating the YOLOv11 model.

To create the environment:
```text
  python3.12 -m venv env_A
  source env_A/bin/activate          (On Windows: env_A\Scripts\activate)
  pip install -r requirements_environment_A.txt
```

Environment B – Python 3.8 (32-bit)
-----------------------------------

Required for training the Fuzzy Decision Tree (FDT).

Due to compatibility limitations, this environment MUST be 32-bit.

To create the environment:
```text
  python3.8 -m venv env_B
  source env_B/bin/activate          (On Windows: env_B\Scripts\activate)
  pip install -r requirements_environment_B.txt
```
Then install the custom FuzzyML library from Bitbucket:
```text
  git clone https://bitbucket.org/mbarsacchi/fuzzyml.git
  cd fuzzyml
```
And collate it correctly inside the script you use


Environment C – Python 3.8
-----------------------------------

Used for extracting handcrafted radiomic features from MRI images.

To create the environment:
```text
  python3.8 -m venv env_C
  source env_C/bin/activate          (On Windows: env_C\Scripts\activate)
  pip install -r requirements_environment_C.txt
```

Each environment is designed to run only a specific phase of the pipeline.
Refer to the relevant script headers or documentation comments to know
which environment is required.

## Execution of the codex
- Each script is prefixed by a letter (e.g., `A_`, `B_`, ...).  
  Scripts should be executed **in alphabetical order**, following their prefix.

- For more details, see the specific README.  