# OASIS: Ownership Authentication through Semantic and Structural Image Search

Official implementation of the paper:

**OASIS: Ownership Authentication through Semantic and Structural Image Search via Multi-Stage Retrieval and Confidence-Aware Verification**

## Overview

OASIS is a multi-stage content ownership verification framework designed to determine whether a query image originates from a source image despite undergoing ownership-preserving transformations such as:

- Cropping
- JPEG Compression
- Screenshot Acquisition
- Text Overlay Insertion
- Gaussian Noise Perturbation

The framework combines semantic retrieval, structural similarity analysis, deep visual representations, and confidence-aware verification to provide robust ownership authentication under realistic content redistribution scenarios.

The repository also includes the implementation of the proposed:

- OASIS-F (Fixed Multi-Stage Framework)
- OASIS-A (Adaptive Multi-Stage Framework)
- COCS (Confidence-Oriented Consistency Score)

---

## Repository Structure

```text
OASIS
│
├── README.md
├── DATASET.md
├── requirements.txt
├── LICENSE
│
├── scripts
│   ├── baselines
│   │   ├── dino_features.py
│   │   ├── resnet_features.py
│   │   └── ssim_baseline.py
│   │
│   ├── hybrid
│   │   ├── similarity.py
│   │   ├── hybrid_model.py
│   │   └── cocs.py
│   │
│   ├── 01_download_5k.py
│   ├── 02_preprocess.py
│   ├── 03_transformations.py
│   ├── 04_create_labels.py
│   ├── 05_generate_embeddings.py
│   ├── 06_build_faiss.py
│   ├── 07_query_search.py
│   ├── 08_evaluate.py
│   ├── 09_hybrid_query.py
│   ├── 10_evaluate_hybrid.py
│   ├── 11_train_fusion_model.py
│   ├── 12_evaluate_ml_model.py
│   ├── 13_Final_Evaluation.py
│   ├── 14_pipeline_fixed_fusion.py
│   └── 15_pipeline_adaptive_fusion.py
│
├── visual_analysis
│   ├── COD_OASIS(1).ipynb
│   ├── COD_OASIS(2).ipynb
│   ├── cod_oasis1.py
│   └── cod_oasis2.py
│
└── Dataset Available via Google Drive
```

---

### Folder Description

- **scripts/** contains the complete implementation of the OASIS framework, dataset generation pipeline, retrieval models, training procedures, and evaluation workflows.
- **baselines/** contains the standalone baseline retrieval implementations.
- **hybrid/** contains similarity fusion, hybrid retrieval, and COCS-related components.
- **visual_analysis/** contains scripts and notebooks used for qualitative analysis, visualization, figure generation, and result interpretation presented in the manuscript.
- **Dataset** is hosted externally via Google Drive due to GitHub storage limitations.

---

## Dataset

The experiments are conducted using a Content Ownership Verification Dataset derived from the Microsoft COCO dataset.

### Dataset Composition

- Original Images: 5,000
- Cropped Images: 5,000
- Compressed Images: 5,000
- Screenshot Images: 5,000
- Text Overlay Images: 5,000
- Noise-Perturbed Images: 5,000

Total Images:

```text
30,000 Images
```

Each transformed image preserves ownership identity while introducing realistic visual modifications commonly encountered in online content-sharing environments.

---

## Dataset Availability

The complete Content Ownership Verification Dataset (30,000 images) is available at:

👉 [Dataset Download](https://drive.google.com/drive/folders/14ssQlzOM0An6z37FP-U6-Hye4L5OzduR?usp=drive_link)

Additional dataset details are provided in [DATASET.md](DATASET.md).

---

## Experimental Pipeline

### Step 1: Dataset Collection

```bash
python 01_download_5k.py
```

Downloads and organizes source images.

### Step 2: Image Preprocessing

```bash
python 02_preprocess.py
```

Standardizes image dimensions and formats.

### Step 3: Transformation Generation

```bash
python 03_transformations.py
```

Generates:

- Crop
- Compression
- Screenshot
- Text Overlay
- Noise

variants.

### Step 4: Label Construction

```bash
python 04_create_labels.py
```

Creates ownership correspondence labels.

### Step 5: Feature Extraction

```bash
python 05_generate_embeddings.py
```

Extracts visual representations using:

- CLIP
- ResNet-50
- DINO

### Step 6: FAISS Index Construction

```bash
python 06_build_faiss.py
```

Builds retrieval indexes for efficient search.

### Step 7: Retrieval Evaluation

```bash
python 07_query_search.py
python 08_evaluate.py
```

Evaluates baseline retrieval models.

### Step 8: Hybrid Retrieval

```bash
python 09_hybrid_query.py
python 10_evaluate_hybrid.py
```

Evaluates semantic-structural fusion.

### Step 9: Machine Learning Retrieval

```bash
python 11_train_fusion_model.py
python 12_evaluate_ml_model.py
```

Trains and evaluates the Random Forest retrieval baseline.

### Step 10: Final Evaluation

```bash
python 13_Final_Evaluation.py
```

Generates overall performance statistics.

### Step 11: OASIS Evaluation

#### OASIS-F

```bash
python 14_pipeline_fixed_fusion.py
```

#### OASIS-A

```bash
python 15_pipeline_adaptive_fusion.py
```

---

## Baseline Models

The following retrieval approaches are evaluated:

- SSIM
- CLIP
- Hybrid CLIP–SSIM
- Random Forest Retrieval
- ResNet-50
- DINO

---

## Proposed Frameworks

### OASIS-F

A fixed multi-stage ownership verification framework that combines semantic retrieval, local verification, multi-model similarity aggregation, and confidence-aware decision making.

### OASIS-A

An adaptive ownership verification framework that dynamically adjusts similarity contributions according to retrieval confidence and transformation characteristics.

---

## Confidence-Oriented Consistency Score (COCS)

COCS is proposed as a confidence estimation metric that evaluates:

- Similarity consistency
- Candidate agreement
- Ranking stability

Higher COCS values indicate stronger ownership verification confidence.

---

## Citation

Citation information will be updated upon publication of the associated manuscript.

---

## License

This repository is released for academic and research purposes.

The original source images belong to the Microsoft COCO dataset and remain subject to their respective licensing terms.
