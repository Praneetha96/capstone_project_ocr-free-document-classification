# Capstone Project — OCR-Free Document Classification Using Vision-Language Models

CS5998 | Master of Data Science & Artificial Intelligence

## Project Overview

Document processing systems depend on OCR to extract text before classification. This project builds and compares two document classification pipelines:

- Baseline: OCR (Tesseract) → TF-IDF → LinearSVC
- VLM Pipeline: Direct image classification using Vision Transformer (ViT) — no OCR required

The core hypothesis is that Vision-Language Models can classify documents directly from visual features, making them more robust to handwriting, low-quality scans, and complex layouts where OCR fails.



## Dataset

RVL-CDIP — 16 document categories including letter, form, email, invoice, resume, and more.

| Split | Samples | Ratio |
|---|---|---|
| Train | 4000 | 80% |
| Validation | 500 | 10% |
| Test | 500 | 10% |

Dataset loaded via HuggingFace streaming — no full download required.

---

## Pipelines
### Baseline — OCR Pipeline
### VLM Pipeline — OCR Free

---

## Milestone 2 Results

| Method | Accuracy | Macro F1 | Inference/image | Requires OCR |
|---|---|---|---|---|
| OCR + TF-IDF + SVM (baseline) | 0.5660 | 0.5643 | 0.01ms | Yes |
| ViT — zero-shot (VLM) | 0.0700 | 0.0436 | 1170ms | No |

Note: The VLM pipeline uses `google/vit-base-patch16-224` in zero-shot mode — no fine-tuning on RVL-CDIP has been applied at this checkpoint. Fine-tuning is planned for Milestone 3 and is expected to significantly improve VLM performance.

---

## Repository Structure
capstone_project_ocr-free-document-classification/
├── notebooks/
│   ├── 00_setup.ipynb           # Environment setup
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Preprocessing pipelines
│   ├── 03_baseline_ocr.ipynb   # OCR baseline pipeline
│   └── 04_vlm_pipeline.ipynb   # VLM pipeline
├── outputs/
│   ├── figures/                 # EDA and results plots
│   └── results/                 # Metrics JSON and CSV files
├── data/
│   └── splits.json              # Fixed train/val/test splits
├── requirements.txt
└── README.md

---

## Environment

- Platform: Google Colab (T4 GPU)
- Python: 3.12
- Random seed: 42

## Setup

```bash
pip install -r requirements.txt
```

Additional system dependency:
```bash
apt-get install -y tesseract-ocr tesseract-ocr-eng
```

## How to Run

1. Open Google Colab
2. Run `notebooks/00_setup.ipynb` first — mounts Drive, installs packages, loads dataset
3. Run notebooks in order: `01_eda` → `02_preprocessing` → `03_baseline_ocr` → `04_vlm_pipeline`
4. All outputs are saved automatically to Google Drive

---

## Key Findings — Milestone 2

- Baseline OCR + SVM achieves 56.6% accuracy — strong performance for a text-only approach
- Best performing classes: `resume` (F1=0.97), `email` (F1=0.77), `scientific publication` (F1=0.68)
- Worst performing class: `handwritten` (F1=0.32) — OCR cannot reliably extract handwritten text
- Zero-shot ViT underperforms at this checkpoint — fine-tuning on RVL-CDIP is required
- Speed trade-off: SVM inference is near-instant, VLM takes ~1.17s per image

---

## Planned for Milestone 3

- Fine-tune ViT on RVL-CDIP training split
- Full evaluation on all 500 validation samples
- Key field extraction using FUNSD and SROIE datasets
- Complete performance comparison and error analysis
- Final report with reproducibility checklist

---

## Reproducibility

- Fixed random seed: `42`
- Fixed train/val/test splits saved to `data/splits.json`
- All package versions pinned in `requirements.txt`
- Hardware: Google Colab T4 GPU

---

## Author

Praneetha96 | CS5998 Capstone Project | 2026
