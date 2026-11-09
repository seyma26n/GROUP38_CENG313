# GROUP38_CENG313 — Image Matching Challenge 2025

![Course](https://img.shields.io/badge/Course-CENG313-blue)
![Group](https://img.shields.io/badge/Group-38-green)
![Python](https://img.shields.io/badge/Python-3-yellow)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)

> **Course:** CENG313  |  **Project:** Kaggle Image Matching Challenge 2025  |  **Group:** 38

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team Members](#2-team-members)
3. [Dataset](#3-dataset)
4. [Assignment 1 — Exploratory Data Analysis](#4-assignment-1--exploratory-data-analysis)
5. [Assignment 1 — Baseline Models](#5-assignment-1--baseline-models)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Experimental Results](#7-experimental-results)
8. [Repository Structure](#8-repository-structure)
9. [How to Run Assignment 1](#9-how-to-run-assignment-1)
10. [Progress Log](#10-progress-log)
11. [Kaggle Submission History](#11-kaggle-submission-history)
12. [Next Steps](#12-next-steps)

---

## 1. Project Overview

This project is based on the [Kaggle Image Matching Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025).

The challenge involves identifying images that belong to the same scene and estimating camera poses. Our work is organized into two assignments:

| Assignment | Focus |
|---|---|
| **Assignment 1** | EDA and baseline models: explore the training data, implement feature-based image clustering methods, and evaluate their performance. |
| **Assignment 2** | Proposed solution: develop an improved approach and compare it with the Assignment 1 baselines. |

> This README documents the implemented **Assignment 1** work.

## 2. Team Members

| Name | Student ID | GitHub | Main Contribution |
|---|---|---|---|
| Şeyma Nur Dağdelen | 24118080036 | [@seyma26n](https://github.com/seyma26n) | Exploratory Data Analysis (EDA) |
| Umida Salyyeva | 23118080411 | [@Umida0105](https://github.com/Umida0105) | Baseline Models and Evaluation |

## 3. Dataset

We use the training data from the Image Matching Challenge 2025. It contains images captured from different scenes, together with scene labels and camera pose information (rotation matrices and translation vectors).

The exploratory analysis covers the whole training dataset, while the baseline experiments focus on two selected subsets:

| Dataset | Number of Images | Ground-Truth Scenes |
|---|---:|---:|
| Stairs | 51 | 2 |
| imc2023_haiper | 54 | 3 |

> **Note:** Ground-truth scene labels are used **only** to evaluate the predicted clusters. They are never provided to the clustering algorithms.

## 4. Assignment 1 — Exploratory Data Analysis

The EDA investigates the dataset through seven analyses:

| # | Analysis | Description |
|---|---|---|
| 4.1 | Dataset Overview | Dataset structure, column names, data types, number of images, and dataset–scene combinations. |
| 4.2 | Dataset Quality Check | Missing values, duplicate records, and potential data quality issues. |
| 4.3 | Image Distribution Analysis | Distribution of images across datasets and scenes using descriptive statistics and visualizations. |
| 4.4 | Scene and Outlier Analysis | Number of scenes per dataset, and the analysis of outlier images and their distribution. |
| 4.5 | Image Resolution Analysis | Image dimensions, aspect ratios, and megapixel distributions for the selected datasets. |
| 4.6 | Visual Similarity Analysis | Visual comparison of images from the same scene and from different scenes to understand matching challenges. |
| 4.7 | Camera Pose Analysis | Rotation matrices and translation vectors, validation of pose data formats, and visualization of camera positions for a selected scene. |

## 5. Assignment 1 — Baseline Models

We implemented two feature-based image clustering baselines built on local feature extraction, descriptor matching, graph construction, and weighted Louvain community detection. Both are evaluated on the same Stairs and Haiper subsets.

### 5.1 Baseline 1 — SIFT + FLANN + Weighted Louvain

**Parameters:** `nfeatures = 1000`, ratio test = `0.75`, minimum good matches = `20`, Louvain `resolution = 1.0`, `seed = 42`. FLANN uses a KD-tree index. On Stairs, 1,275 image pairs were compared and 419 had at least 20 good matches.

1. Extract SIFT keypoints and descriptors from each image.
2. Match descriptors between image pairs using FLANN.
3. Apply Lowe's Ratio Test (threshold = 0.75).
4. Construct a weighted image-matching graph.
5. Connect image pairs with at least 20 good matches.
6. Apply weighted Louvain community detection to obtain image clusters.
7. Evaluate the predicted clusters against ground-truth scene labels.

### 5.2 Baseline 2 — ORB + BFMatcher + Weighted Louvain

**Parameters:** `nfeatures = 1000`, ratio test = `0.75`, minimum good matches = `20`, Louvain `resolution = 1.0`, `seed = 42`.

1. Extract ORB keypoints and binary descriptors.
2. Match descriptors using BFMatcher with Hamming distance.
3. Apply Lowe's Ratio Test (threshold = 0.75).
4. Construct a weighted image-matching graph.
5. Connect image pairs with at least 20 good matches.
6. Apply weighted Louvain community detection.
7. Evaluate the predicted clusters using the same metrics as Baseline 1.

Both baselines share the same clustering approach, so the comparison isolates the effect of **SIFT-based vs. ORB-based** feature matching.

## 6. Evaluation Metrics

| Metric | Description |
|---|---|
| **Adjusted Rand Index (ARI)** | Agreement between predicted clusters and ground-truth scene labels. |
| **Pairwise Precision** | Fraction of image pairs predicted to be in the same cluster that are correct. |
| **Pairwise Recall** | Fraction of actual same-scene image pairs that are successfully grouped together. |
| **Pairwise F1-score** | Harmonic mean of pairwise precision and recall. |

## 7. Experimental Results

### 7.1 Baseline Comparison

| Dataset | Method | ARI | Precision | Recall | Pairwise F1 |
|---|---|---:|---:|---:|---:|
| Stairs | SIFT | **0.4559** | 0.7963 | **0.6070** | **0.6888** |
| Stairs | ORB | 0.2329 | 0.7866 | 0.3154 | 0.4502 |
| Haiper | SIFT | **0.6650** | 0.6657 | **1.0000** | **0.7993** |
| Haiper | ORB | 0.1835 | **1.0000** | 0.1444 | 0.2523 |

### 7.2 Discussion

- **SIFT outperformed ORB on both datasets** in terms of ARI and Pairwise F1-score.
- On **Stairs**, SIFT reached an ARI of 0.4559 versus 0.2329 for ORB.
- On **Haiper**, SIFT reached an ARI of 0.6650 versus 0.1835 for ORB.
- Although ORB achieved a precision of 1.0000 on Haiper, its recall of only 0.1444 shows that many images of the same scene were not grouped together.

#### Limitations observed (SIFT baseline)

- **Stairs:** 6 clusters were produced although there are only 2 ground-truth scenes (over-segmentation).
- **Haiper:** all 23 Fountain images were grouped correctly, but the 15 Bike and 16 Chairs images were merged into one cluster. This explains the perfect recall (1.0000) with lower precision (0.6657).
- Visually similar images from different scenes can create misleading graph edges.
- Possible improvements: geometric verification (e.g., RANSAC) and more robust edge selection.

Overall, SIFT-based feature matching provides a stronger baseline for the selected clustering experiments, and these results serve as reference points for developing and evaluating improved methods in Assignment 2.

## 8. Repository Structure

```text
GROUP38_CENG313/
├── README.md
├── notebooks/
│   └── Assignment1_eda_baseline.ipynb
└── src/
    ├── __init__.py
    ├── data.py
    ├── models.py
    └── utils.py
```

The notebook contains the EDA, both baseline implementations, evaluation metrics, comparison tables, and visualizations.

## 9. How to Run Assignment 1

### Requirements

The notebook is designed to run in **Google Colab** with Python. Main libraries:

- NumPy
- Pandas
- Matplotlib
- Pillow
- OpenCV
- NetworkX
- Scikit-learn
- Kaggle API

### Execution Instructions

1. Open the [Assignment 1 notebook](notebooks/Assignment1_eda_baseline.ipynb) on GitHub, or open it in Google Colab.
2. Connect to a Google Colab runtime.
3. Make sure the training dataset is available in Google Drive at:

   ```text
   /content/drive/MyDrive/CENG313_Project/data
   ```

   The EDA section expects `train_labels.csv` and the selected training images in this location.

4. Mount Google Drive and run the EDA section.
5. For the baseline section, configure Kaggle API authentication using a Google Colab Secret named `KAGGLE_API_TOKEN`, and grant the notebook access to the secret.
6. Make sure your Kaggle account has access to the competition data.
7. Run the baseline setup and download cells, followed by the SIFT and ORB experiments.
8. Review the generated evaluation tables and comparison charts.

> **Note:** The notebook uses Google Drive data paths for the EDA and Kaggle downloads for the baseline experiments. Configure the required data and authentication before running the corresponding sections.

## 10. Progress Log

| Milestone | Work Completed |
|---|---|
| Project setup | Selected Image Matching Challenge 2025 and organized the team repository |
| Data exploration | Examined dataset structure, quality, image distribution, scenes, and outliers |
| Visual analysis | Analyzed image resolutions, visual similarities, and camera pose information |
| Baseline 1 | Implemented SIFT, FLANN matching, and weighted Louvain clustering |
| Baseline 2 | Implemented ORB, BFMatcher, and weighted Louvain clustering |
| Evaluation | Calculated ARI, Pairwise Precision, Recall, and F1-score |
| Comparison | Compared SIFT and ORB results using tables and graphs |

## 11. Kaggle Submission History

The Assignment 1 notebook documents experiments on selected training subsets rather than a Kaggle leaderboard submission. No leaderboard submission details are documented yet.

If submissions are made, their dates, methods, and scores will be recorded here.

## 12. Next Steps

Assignment 2 focuses on developing and evaluating a proposed solution for the Image Matching Challenge 2025. The Assignment 1 baselines provide the reference results for comparing future improvements.
