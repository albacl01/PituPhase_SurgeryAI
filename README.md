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
│   ├── extract_frames.ipynb          # Extract 1 FPS frames from surgical videos
│   └── convert_labels.ipynb          # Convert .json annotations to .csv
│
├── SSL_Pretraining/               # Self-supervised training scripts
│   ├── simclr_pretrain.ipynb         # SimCLR Pretraining
│   └── byol_pretrain.ipynb           # BYOL Pretraining
│   └── attention_pooling.py       # Attention-weighted pooling operator
│
├── downstream_evaluation/         # Evaluate representations with a linear classifier
│   └── evaluate_classifier.ipynb     
│
├── README.md                      # Project overview
├── LICENSE                        # License (MIT recommended)
└── .gitignore                     # Ignore logs, models, etc.
```

---

## Important Information

The dataset associated with this paper is publicly available and can be found at: https://doi.org/10.21950/YDGPZM

The dataset includes 44 endoscopic pituitary surgery videos corresponding to complete patient surgeries. Each patient folder has been named with an anonymized patient identifier (e.g., '10XX') and contains all the video segments (.mp4 files) that together form the complete surgical procedure for that patient. Additionally, a single .json file is included per folder, providing frame-level annotations at 1fps for the entire procedure.

The files have been encrypted to ensure secure handling. Access requires the completion and approval of a Data Use Agreement (DUA), which is available in the dataset repository. The signed agreement should be submitted to the designated contact, Maria Paula de Toledo Heras at mtoledo@inf.uc3m.es. Once the request has been reviewed and approved by the dataset authors, the decryption password will be securely shared with the requester to enable file access.

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
