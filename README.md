# PituPhase_SurgeryAI
**Improving Surgical Phase Recognition using Self-Supervised Deep Learning**

This repository accompanies the study “Improving Surgical Phase Recognition using Self-Supervised Deep Learning”, and includes the full pipeline for data preparation, label processing, SSL pretraining (SimCLR and BYOL), and downstream evaluation.

The paper explores the use of self-supervised learning (SSL) to improve surgical phase recognition in endoscopic pituitary surgery, a complex and highly variable procedure. To the best of our knowledge, this is the first work to apply self-supervision in this domain. 


We leverage two state-of-the-art SSL frameworks — SimCLR and BYOL — to pretrain a ResNet50 encoder using unlabeled surgical video frames. During pretraining, the model learns meaningful visual representations by comparing two augmented views of the same input image using a contrastive loss. An attention-weighted pooling operator is also added into the SSL pretraining stage refined spatial feature extraction. 

After pretraining, we freeze the encoder and train a linear classifier on labeled data to predict surgical phases, following the standard linear evaluation protocol.

Our results show that:

- SSL approaches outperform fully supervised models.

- SimCLR achieves the best performance, especially when paired with the attention layer.

- Models remain robust even when the available pool of training data is reduced, improving annotation efforts.

This two-step architecture enables label-efficient training for real-world surgical workflow analysis, while maintaining state-of-the-art accuracy.

![linear evaluation visual abstract](https://github.com/user-attachments/assets/7e769f9a-45dc-471e-992d-e684bbd6691d)



---
## Repository Structure

```
PituPhase_SurgeryAI/
├── data_preparation/              # Prepare the dataset
│   ├── extract_frames.py          # Extract 1 FPS frames from surgical videos
│   └── convert_labels.py          # Convert .json annotations to .csv
│   └── create_partitions.py       # Create data splits
│
├── SSL_Pretraining/               # Self-supervised training scripts
│   ├── simclr_pretrain.py         # SimCLR Pretraining
│   └── byol_pretrain.py           # BYOL Pretraining
│
├── downstream_evaluation/         # Evaluate representations with a linear classifier
│   └── evaluate_classifier.py     
│
├── notebooks/                     # Interactive visualizations and analyses
│   └── exploratory.ipynb          # Frame samples, t-SNE plots, metrics, etc.
│
│
├── models/                        # Saved model weights
│   ├── simclr_model.pt
│   └── byol_model.pt
│
├── utils/                         # Helper modules
│   ├── metrics.py                 # Precision, recall, F1, confusion matrix
│   └── attention_pooling.py       # Attention-weighted pooling operator
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── LICENSE                        # License (MIT recommended)
└── .gitignore                     # Ignore logs, models, etc.
```


---

## Features

- Full preprocessing pipeline: video to frame conversion, label parsing, and stratified data splits
- SimCLR and BYOL pretraining with ResNet50 backbone
- Attention-weighted pooling layer integrated into both frameworks
- Downstream linear evaluation to assess learned representations
- Macro-averaged precision, recall, F1-score, and confusion matrix evaluation
- Configurable for reduced label subset experiments (50%, 25%, 10%)

---

## Setup Instructions

```bash
git clone https://github.com/yourusername/PituPhase_SurgeryAI.git
cd PituPhase_SurgeryAI

# Create virtual environment
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
