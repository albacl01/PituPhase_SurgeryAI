# PituPhase_SurgeryAI
**Improving Surgical Phase Recognition using Self-Supervised Deep Learning**

This repository accompanies the study “Improving Surgical Phase Recognition using Self-Supervised Deep Learning”, and includes the full pipeline for data preparation, label processing, SSL pretraining (SimCLR and BYOL), and downstream evaluation.

---

**Repository Structure**

<pre> ``` PituPhase_SurgeryAI/ ├── data_preparation/ # Convert raw videos to image frames │ └── extract_frames.py # Extract 1 FPS frames from surgical videos │ ├── label_processing/ # Process phase annotations and create data splits │ ├── convert_labels.py # Convert .json annotations to .csv │ └── create_partitions.py # Train/val/test splits (patient-wise) │ ├── ssl_pretraining/ # Self-supervised training scripts │ ├── simclr_pretrain.py # SimCLR pretraining │ └── byol_pretrain.py # BYOL pretraining │ ├── downstream_evaluation/ # Evaluate representations with a linear classifier │ └── evaluate_classifier.py # Works for both SimCLR and BYOL │ ├── notebooks/ # Interactive visualizations and analyses │ └── exploratory.ipynb # Frame samples, t-SNE plots, metrics, etc. │ ├── data/ # Dataset structure (example) │ ├── videos/ # Raw .mp4 or .avi files (not included) │ ├── frames/ # Extracted images │ ├── annotations/ # .csv or .json with phase labels │ └── splits/ # train/val/test partitions │ ├── models/ # Saved model weights (optional) │ ├── simclr_model.pt │ └── byol_model.pt │ ├── utils/ # Helper modules │ ├── metrics.py # Precision, recall, F1, confusion matrix │ └── attention_pooling.py # Custom pooling layer │ ├── requirements.txt # Python dependencies ├── README.md # Project overview ├── LICENSE # License (MIT recommended) └── .gitignore # Ignore logs, models, etc. ``` </pre>
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
