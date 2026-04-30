# Capstone Project — OCR-Free Document Classification Using Vision-Language Models

CS5998 | Master of Data Science & Artificial Intelligence

## Project Overview

Document processing systems require OCR for text extraction before document classification. Three different document classification pipelines are designed and evaluated in this project:

- **Baseline** : OCR (Tesseract) → TF-IDF → LinearSVC
- **VLM Pipeline** : Direct image classification using Vision Transformer (ViT) — no OCR required
- **Improved Pipeline**: Fine-tuned DistilBERT on OCR-extracted text

The core hypothesis is that transformer-based models will perform better than the traditional OCR pipelines for document classification, and that Vision-Language Models can classify documents directly from their visual properties without OCR.

## Dataset

RVL-CDIP — 16 document categories including letter, form, email, invoice, resume, and more.

| Dataset | Task | Train | Val/Test | Classes | Source |
|---|---|---|---|---|---|
| RVL-CDIP (subset) | Document classification | 4000 | 500/500 | 16 | HuggingFace |
| FUNSD | Form field extraction | 149 docs | 50 docs | 4 | guillaumejaume.github.io |
| SROIE | Invoice field extraction | ~500 | ~126 | 4+other | GitHub/zzzDavid |

Dataset loaded via HuggingFace streaming — no full download required.

---

## Pipelines
### Baseline — OCR Pipeline
Image → Grayscale + Denoise → Tesseract OCR → TF-IDF → LinearSVC → Predicted Class

### VLM Pipeline — OCR Free
Image → RGB + Resize 224×224 → ViT (google/vit-base-patch16-224) → Predicted Class

### Improved Pipeline — Fine-Tuned DistilBERT
OCR Text → DistilBERT Tokenization → Fine-tuned DistilBERT → Predicted Class

---

## Final Results — Milestone 3

### Document Classification

| Method | Accuracy | Macro F1 | Inference/image | Requires OCR |
|---|---|---|---|---|
| OCR + TF-IDF + SVM (baseline) | 0.5660 | 0.5643 | ~500ms | Yes |
| ViT zero-shot | 0.0700 | 0.0436 | 1170ms | No |
| DistilBERT fine-tuned | **0.6080** | **0.6026** | ~510ms | Yes |

### Key Field Extraction

| Task | Dataset | Method | Score |
|---|---|---|---|
| Form field extraction | FUNSD | DistilBERT NER | F1 = 0.2239 |
| Invoice field extraction | SROIE | DistilBERT classifier | Acc = 1.0000 |

---

## Repository Structure

capstone_project_ocr-free-document-classification/
├── notebooks/
│   ├── 00_setup.ipynb                  # Environment setup
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_preprocessing.ipynb         # Preprocessing pipelines
│   ├── 03_baseline_ocr.ipynb          # OCR baseline pipeline
│   ├── 04_vlm_pipeline.ipynb          # VLM zero-shot pipeline
│   ├── 05_vit_finetuning.ipynb        # DistilBERT fine-tuning
│   ├── 06_key_field_extraction.ipynb  # FUNSD and SROIE extraction
│   └── 07_full_evaluation.ipynb       # Full evaluation and comparison
├── outputs/
│   ├── figures/                        # EDA and results plots
│   └── results/                        # Metrics JSON and CSV files
├── data/
│   └── splits.json                     # Fixed train/val/test splits
├── requirements.txt
└── README.md

---

## Environment

- **Platform**: Google Colab (T4 GPU)
- **Python**: 3.12
- **Random seed**: 42

## Setup

```bash
pip install -r requirements.txt
```

**Additional system dependency:**
```bash
apt-get install -y tesseract-ocr tesseract-ocr-eng
```

## How to Run

1. Open Google Colab
2. Run `notebooks/00_setup.ipynb` first — mounts Drive, installs packages, loads dataset
3. Run notebooks in order: `01_eda` → `02_preprocessing` → `03_baseline_ocr` → `04_vlm_pipeline` → `05_vit_finetuning` → `06_key_field_extraction` → `07_full_evaluation`
4. All outputs are saved automatically to Google Drive

---

## Key Findings

1. Baseline OCR + SVM achieves **56.6% accuracy** - strong for a classical text-only approach
2. Zero-shot ViT achieves only **7% accuracy** - confirms domain adaptation is essential
3. Fine-tuned DistilBERT achieves **60.8% accuracy** - 4.2% improvement over baseline
4. Biggest improvement on handwritten class - F1 went from 0.32 to 0.53
5. Transformer models outperform rule-based approaches on both FUNSD and SROIE

---

## Reproducibility

- Fixed random seed: `42`
- Fixed train/val/test splits saved to `data/splits.json`
- All package versions pinned in `requirements.txt`
- Hardware: Google Colab T4 GPU
- Best model checkpoint: `best_distilbert_model.pt` (saved to Google Drive)

---

## Author
Praneetha96 | CS5998 Capstone Project | 2026
