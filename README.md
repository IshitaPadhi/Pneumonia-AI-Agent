# 🫁 XPneumoNet — AI-Powered Pneumonia Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat&logo=mysql&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-86.5%25-brightgreen?style=flat)
![MacroF1](https://img.shields.io/badge/Macro%20F1-0.859-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

**A clinical-grade explainable AI system for 3-class pneumonia detection from chest X-rays,
built with DenseNet-121, Grad-CAM visualisation, and a full-stack MySQL patient management dashboard.**

[Features](#-features) · [Architecture](#-system-architecture) · [Dataset](#-dataset) · [Setup](#-setup--installation) · [Usage](#-usage) · [Results](#-results) · [Team](#-team)

</div>

---

## 📌 Overview

XPneumoNet is an academic AI project developed as part of the **CSS 2203 – Artificial Intelligence Lab** at **Manipal Institute of Technology, Manipal**. It classifies chest X-ray images into three categories — **Bacterial Pneumonia**, **Viral Pneumonia**, and **Normal** — using a fine-tuned DenseNet-121 model with explainability via Grad-CAM heatmaps.

The system is modelled as a **PEAS (Performance, Environment, Actuators, Sensors) AI Agent** and is deployed as an interactive Streamlit web application with MySQL-backed patient record management.

> ⚠️ **Disclaimer:** This project is for research and educational purposes only. It is not a substitute for professional clinical diagnosis.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **3-Class Detection** | Classifies X-rays as Bacterial Pneumonia, Viral Pneumonia, or Normal |
| 🧠 **Grad-CAM XAI** | Generates heatmaps highlighting regions that influenced the model's decision |
| 📊 **Probability Distribution** | Displays per-class confidence scores with visual bars |
| 🏥 **AI Clinical Recommendation** | Agent-generated severity assessment and medical advice |
| 📋 **Patient Records** | Full MySQL-backed patient history with live search |
| 📈 **Analytics Dashboard** | Case distribution, confidence trends, severity breakdown |
| 📄 **PDF Report Export** | Downloadable diagnostic report with patient details and AI findings |
| 🔁 **Monte Carlo Dropout** | Uncertainty estimation for more reliable confidence scores |
| 💾 **Master Recovery Cell** | Colab session resilience — auto-saves training checkpoints |

---

## 🏗 System Architecture

```
XPneumoNet/
│
├── model/
│   └── best_pneumonia_model.keras     # Trained DenseNet-121 weights
│
├── agent/
│   └── agent.py                       # PEAS AI agent — severity + advice logic
│
├── database/
│   ├── mysql_connect.py               # DB connection + all query functions
│   └── schema.sql                     # MySQL table definition
│
├── assets/                            # Evaluation graphs and figures
│
├── temp/                              # Runtime: uploaded images + heatmaps
│
├── app.py                             # Main Streamlit application
├── evaluate.py                        # Standalone evaluation script
├── .env                               # DB credentials (not committed)
├── requirements.txt
└── README.md
```

### PEAS Agent Model

| Component | Description |
|---|---|
| **Performance** | Correct classification, high confidence, accurate Grad-CAM focus |
| **Environment** | Chest X-ray images (DICOM/JPEG/PNG) |
| **Actuators** | Prediction label, confidence score, heatmap, clinical recommendation |
| **Sensors** | Image input, patient metadata |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Base Architecture** | DenseNet-121 (ImageNet pre-trained) |
| **Input Size** | 224 × 224 × 3 |
| **Output Classes** | 3 (BACTERIA, VIRUS, NORMAL) |
| **Loss Function** | Focal Loss (handles class imbalance) |
| **Optimizer** | Adam |
| **Explainability** | Grad-CAM (last convolutional layer) |
| **Uncertainty** | Monte Carlo Dropout |
| **Best Val Accuracy** | 88.0% (epoch 9) |
| **Test Accuracy** | **86.5%** |
| **Macro F1** | **0.859** |

### Model Comparison

| Model | Val Accuracy | Notes |
|---|---|---|
| DenseNet-121 | **86.5% test acc** | Primary model — selected for deployment |
| ResNet-50 | Lower | Comparison baseline |
| VGG-16 | Lower | Comparison baseline |
| EfficientNet-B3 | Lower | Comparison baseline |

---

## 📂 Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) — Kaggle (Paul Mooney)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Original Size:** ~1.2 GB
- **Reorganised into 3 classes:**
  - `BACTERIA/` — Bacterial pneumonia X-rays
  - `VIRUS/` — Viral pneumonia X-rays
  - `NORMAL/` — Healthy lungs
- **Preprocessing:** Resize to 224×224, normalise to [0, 1], augmentation applied during training

### Dataset Distribution

| Split | BACTERIA | NORMAL | VIRUS | Total |
|---|---|---|---|---|
| Train | 2530 | 1341 | 1345 | 5216 |
| Val | 8 | 8 | 8 | 24 |
| Test | 242 | 234 | 148 | 624 |

> BACTERIA represents 47.5% of the overall dataset, NORMAL 27.0%, VIRUS 25.5%.

<div align="center">
  <img src="assets/dataset_distribution.png" width="800" alt="Dataset class distribution bar chart and pie chart"/>
</div>

---

## 📊 Results

### Summary

| Metric | Value |
|---|---|
| **Overall Accuracy** | **86.5%** |
| **Macro Avg Precision** | 0.859 |
| **Macro Avg Recall** | 0.859 |
| **Macro Avg F1-Score** | 0.859 |
| **Test Set Size** | 624 images |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support | AUC-ROC | Avg Precision |
|---|---|---|---|---|---|---|
| **BACTERIA** | 0.874 | 0.888 | 0.881 | 242 | 0.957 | 0.934 |
| **NORMAL** | 0.891 | 0.872 | 0.881 | 234 | 0.970 | 0.959 |
| **VIRUS** | 0.812 | 0.818 | 0.815 | 148 | 0.944 | 0.825 |
| **Macro Avg** | **0.859** | **0.859** | **0.859** | 624 | — | — |

> The VIRUS class shows slightly lower scores due to inherent class imbalance (1345 vs 2530 BACTERIA training samples) and the radiologically ambiguous boundary between bacterial and viral consolidation patterns — consistent with published literature on this dataset.

---

### Confusion Matrix

<div align="center">
  <img src="assets/confusion_matrix.png" width="800" alt="Confusion matrix raw counts and normalised"/>
</div>

Key observations:
- BACTERIA recall: **89%** — only 24 misclassified as NORMAL, 3 as VIRUS
- NORMAL recall: **87%** — only 5 misclassified as BACTERIA
- VIRUS recall: **82%** — 26 misclassified as BACTERIA (clinically expected overlap)

---

### Per-Class Precision, Recall & F1

<div align="center">
  <img src="assets/per_class_metrics.png" width="750" alt="Bar chart of per-class precision recall and F1 scores"/>
</div>

---

### ROC Curves (One-vs-Rest)

<div align="center">
  <img src="assets/roc_curves.png" width="600" alt="ROC curves for each class showing AUC scores"/>
</div>

All three classes achieve AUC > 0.94, indicating strong discriminative ability even when classification confidence is borderline.

---

### Precision-Recall Curves

<div align="center">
  <img src="assets/precision_recall_curves.png" width="600" alt="Precision-recall curves for each class"/>
</div>

---

### Training History

<div align="center">
  <img src="assets/training_history.png" width="800" alt="DenseNet-121 training accuracy and loss over 10 epochs"/>
</div>

Model converges steadily over 10 epochs. Best validation accuracy of **88.0%** was achieved at epoch 9. Train and val curves track closely, indicating no significant overfitting.

---

### Model Confidence Distribution

<div align="center">
  <img src="assets/confidence_distribution.png" width="750" alt="Violin plot of model confidence per true class"/>
</div>

NORMAL predictions cluster tightly near 1.0 confidence. BACTERIA and VIRUS show wider spread — consistent with the higher visual similarity between those two classes.

---

### Grad-CAM Explainability

<div align="center">
  <img src="assets/gradcam.png" width="600" alt="Grad-CAM heatmaps showing model attention on chest X-rays for each class"/>
</div>

Grad-CAM confirms the model attends to clinically relevant lung regions. Bacterial cases show focal consolidation patterns; viral cases show more diffuse bilateral attention.

---

### Classification Results Summary

<div align="center">
  <img src="assets/results_summary.png" width="700" alt="XPneumoNet classification results summary table"/>
</div>

---

## 🗄 Database Schema

```sql
CREATE TABLE patients (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    patient_id  VARCHAR(50),
    name        VARCHAR(100),
    age         INT,
    date        DATE,
    prediction  VARCHAR(20),
    confidence  FLOAT,
    severity    VARCHAR(50),
    report      TEXT,
    image_path  VARCHAR(255)
);
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- MySQL 8.0+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/IshitaPadhi/Pneumonia-AI-Agent.git
cd Pneumonia-AI-Agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=pneumonia_db
```

### 4. Set up the database

```bash
mysql -u root -p < database/schema.sql
```

### 5. Add the trained model

Place `best_pneumonia_model.keras` inside the `model/` directory.

### 6. Run the application

```bash
streamlit run app.py
```

---

## 🖥 Usage

1. Navigate to `http://localhost:8501` in your browser
2. Open the **Diagnostic** tab
3. Enter patient details (ID, name, age)
4. Upload a chest X-ray (JPG / PNG / JPEG)
5. Click **⚡ Run AI Diagnostic**
6. View prediction, Grad-CAM heatmap, probability bars, and AI recommendation
7. Download the PDF diagnostic report
8. Switch to **Patient Records** to view history and search patients
9. Switch to **Analytics** to view case distribution and model performance trends

---

## 📦 Requirements

```
streamlit
tensorflow
opencv-python
numpy
scikit-learn
seaborn
matplotlib
reportlab
python-dotenv
mysql-connector-python
```

---

## 🔮 Future Scope

- [ ] Feedback loop — let clinicians mark predictions as correct/incorrect to build a retraining dataset
- [ ] Class-weighted resampling to address VIRUS underrepresentation
- [ ] Test-time augmentation (TTA) for improved inference confidence
- [ ] DICOM file support for direct hospital system integration
- [ ] Multi-GPU training pipeline
- [ ] REST API wrapper for integration with hospital EMR systems
- [ ] Mobile-responsive UI

---

## 👩‍💻 Team

Developed by a 5-member team as part of the **CSS 2203 – AI Lab**, Semester IV
**Manipal Institute of Technology, Manipal**

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ at MIT Manipal · For research and educational use only</sub>
</div>
