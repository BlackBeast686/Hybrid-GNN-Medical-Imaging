# Hybrid GNN for Medical Imaging

Hybrid deep learning models that combine a **ResNet50** convolutional image branch with a **Graph Isomorphism Network (GIN)** branch for multimodal medical image classification. The image branch learns visual features from scans, while the graph branch models relationships between tabular/clinical features, and the two are fused before the final classifier.

This repository contains two experiments on two different datasets:

| Notebook | Task | Dataset | Classes |
|----------|------|---------|---------|
| `Heart disease Detection.ipynb` | Multi-label thoracic finding classification | MIMIC-CXR (chest X-ray) | 11 (multi-label) |
| `Breast Cancer GNN Model.ipynb` | Breast histology classification | BACH / ICIAR2018 | 4 (Benign, InSitu, Invasive, Normal) |

> **Note on scope:** These are research/experimental notebooks and are **not** intended for clinical use or real diagnosis.

## Approach

Both models share the same core idea:

1. **Image branch** — a pretrained ResNet50 (ImageNet weights) extracts features from 224×224 images, followed by global average pooling.
2. **Graph branch** — a custom GIN layer (built on [Spektral](https://graphneural.network/)) processes tabular features, using an adjacency matrix built from feature similarity.
3. **Fusion** — image and graph features are concatenated and passed through dense layers to the final classifier.

The two notebooks differ in their dataset, label structure (multi-label sigmoid vs. single-label softmax), adjacency construction, and GIN implementation details.

## Results

Reported on the held-out test/validation split in each notebook:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| MIMIC-CXR (hybrid GIN) | 0.847 | 0.990 | 0.972 | 0.981 |
| BACH (hybrid GIN) | 0.813 | 0.823 | 0.813 | 0.812 |

> These numbers come from small sampled subsets and single runs — treat them as indicative rather than definitive.

## Getting started

### Requirements

The notebooks were developed in Google Colab with a GPU (T4) runtime. Main dependencies:

```
tensorflow>=2.18
spektral
torch_geometric
scikit-learn
pandas
numpy
opencv-python
matplotlib
seaborn
kagglehub
```

Install with:

```bash
pip install -r requirements.txt
```

### Data

The datasets are **not** included in this repository due to their size (MIMIC-CXR ~2 GB, BACH ~12.5 GB). Download them from Kaggle:

- **MIMIC-CXR:** `itsanmol124/mimic-cxr`
- **BACH:** `truthisneverlinear/bach-breast-cancer-histology-images`

The notebooks download these automatically via the Kaggle API / `kagglehub`. You'll need your own Kaggle credentials configured.

### Running

Open either notebook in Google Colab (or Jupyter with a GPU) and run the cells top to bottom. Update the dataset paths if you're not on Colab.

## Repository structure

```
.
├── Heart disease Detection.ipynb     # MIMIC-CXR hybrid model
├── Breast Cancer GNN Model.ipynb     # BACH hybrid model
├── requirements.txt
├── .gitignore
└── README.md
```

## License

Add a license of your choice (e.g. MIT) if you want others to reuse this work. The underlying datasets have their own licenses and usage terms — check Kaggle before redistributing any data.
