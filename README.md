# PituPhase_SurgeryAI
**Improving Surgical Phase Recognition using Self-Supervised Deep Learning**

This repository accompanies the study “Improving Surgical Phase Recognition using Self-Supervised Deep Learning”, and includes the full pipeline for data preparation, label processing, SSL pretraining (SimCLR and BYOL), and downstream evaluation.

---

**Repository Structure**

PituPhase_SurgeryAI/
│
├── data_preparation/
│ └── extract_frames.py # Extract frames from videos (1 fps)
│
├── label_processing/
│ └── convert_labels.py # Convert JSON annotations to CSV
│ └── create_partitions.py # Train/val/test splits (patient-wise)
│
├── ssl_pretraining/
│ ├── simclr_pretrain.py # SimCLR self-supervised training
│ └── byol_pretrain.py # BYOL self-supervised training
│
├── downstream_evaluation/
│ └── evaluate_classifier.py # Linear classifier evaluation on frozen features
│
├── notebooks/
│ └── exploratory.ipynb # Jupyter notebook for visualization and experiments
│
└── README.md

yaml
Copiar
Editar

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
