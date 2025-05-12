# PituPhase_SurgeryAI
**Improving Surgical Phase Recognition using Self-Supervised Deep Learning**

This repository accompanies the study “Improving Surgical Phase Recognition using Self-Supervised Deep Learning”, and includes the full pipeline for data preparation, label processing, SSL pretraining (SimCLR and BYOL), and downstream evaluation.

---

**Repository Structure**

PituPhase_SurgeryAI/
│
├── data_preparation/                    # Convert raw videos to image frames
│   └── extract_frames.py                # Extract 1 FPS frames from surgical videos
│
├── label_processing/                    # Process phase annotations and create data splits
│   ├── convert_labels.py                # Convert raw .json annotations to .csv
│   └── create_partitions.py             # Split data into train/val/test (patient-wise)
│
├── ssl_pretraining/                     # Self-supervised learning models
│   ├── simclr_pretrain.py               # SimCLR training script
│   └── byol_pretrain.py                 # BYOL training script
│
├── downstream_evaluation/               # Evaluate representations on SPR classification
│   └── evaluate_classifier.py           # Train/test linear classifier on SSL embeddings
│
├── notebooks/                           # Interactive exploration and visualizations
│   └── exploratory.ipynb                # Frame samples, t-SNE plots, metrics, etc.
│
├── data/                                # Expected structure for local dataset
│   ├── videos/                          # Raw .mp4 or .avi videos (not included)
│   ├── frames/                          # Extracted images (organized per patient/video)
│   ├── annotations/                     # .json or .csv files with phase labels
│   └── splits/                          # CSVs listing filenames for train/val/test
│
├── models/                              # Saved model weights or checkpoints (optional)
│   └── simclr_model.pt
│   └── byol_model.pt
│
├── utils/                               # Helper functions
│   └── metrics.py                       # Metric functions (e.g., precision, recall, F1)
│   └── attention_pooling.py             # Shared pooling layer used in both models
│
├── requirements.txt                     # Python dependencies
├── README.md                            # Repository overview and usage
├── LICENSE                              # Open-source license (e.g., MIT)
└── .gitignore                           # Common patterns to ignore


---

## 🧪 Features

- Full preprocessing pipeline: video to frame conversion, label parsing, and stratified data splits
- SimCLR and BYOL pretraining with ResNet50 backbone
- Attention-weighted pooling layer integrated into both frameworks
- Downstream linear evaluation to assess learned representations
- Macro-averaged precision, recall, F1-score, and confusion matrix evaluation
- Configurable for reduced label subset experiments (50%, 25%, 10%)

---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/yourusername/PituPhase_SurgeryAI.git
cd PituPhase_SurgeryAI

# Create virtual environment
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
