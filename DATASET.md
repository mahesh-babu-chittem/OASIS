# Content Ownership Verification Dataset

## Overview

This repository accompanies the paper:

**OASIS: Ownership Authentication through Semantic and Structural Image Search via Multi-Stage Retrieval and Confidence-Aware Verification**

The experiments were conducted using a Content Ownership Verification Dataset constructed from the Microsoft COCO dataset. The dataset was designed to evaluate ownership verification under realistic image redistribution scenarios.

---

## Dataset Statistics

| Category | Number of Images |
|-----------|----------------|
| Original Images | 5,000 |
| Cropped Images | 5,000 |
| Compressed Images | 5,000 |
| Screenshot Images | 5,000 |
| Text Overlay Images | 5,000 |
| Noise-Perturbed Images | 5,000 |
| **Total Images** | **30,000** |

Each transformed image preserves the ownership identity of its corresponding source image while introducing realistic modifications commonly encountered in online content-sharing environments.

---

## Transformations

The following ownership-preserving transformations were generated for every source image:

1. Cropping
2. JPEG Compression
3. Screenshot Acquisition
4. Text Overlay Insertion
5. Gaussian Noise Perturbation

These transformations simulate common image modifications observed across social media platforms, messaging applications, and content-sharing ecosystems.

---

## Dataset Download

The complete dataset is hosted externally due to GitHub storage limitations.

### Google Drive Link

**Dataset:**  
https://drive.google.com/drive/folders/14ssQlzOM0An6z37FP-U6-Hye4L5OzduR?usp=sharing

---

## Dataset Structure

```text
data/
└── processed/
    ├── original/
    ├── crop/
    ├── compression/
    ├── screenshot/
    ├── text_overlay/
    └── noise/
```

The `original` directory contains the 5,000 source images collected from the Microsoft COCO dataset. The remaining directories contain ownership-preserving transformed variants generated for experimental evaluation. Each source image is associated with five transformed versions corresponding to cropping, compression, screenshot acquisition, text overlay insertion, and Gaussian noise perturbation.
```

---

## Source Dataset

The original source images were derived from the Microsoft COCO dataset.

Researchers using this dataset should comply with the licensing and usage terms of the original COCO dataset.

COCO Dataset:
https://cocodataset.org/

---

## Reproducibility

This repository provides all scripts required to:

- Download source images
- Generate image transformations
- Create ownership labels
- Extract visual embeddings
- Build retrieval indices
- Train retrieval models
- Evaluate retrieval performance
- Evaluate OASIS-F
- Evaluate OASIS-A
- Compute COCS scores

The complete experimental pipeline can therefore be reproduced using the provided code and dataset resources.

---

## Citation

The citation for the associated manuscript will be made available upon publication.

**Citation details and DOI will be updated soon.**
