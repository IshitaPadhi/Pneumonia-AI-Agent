# 🫁 XPneumoNet — AI-Powered Pneumonia Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat&logo=mysql&logoColor=white)
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
├── temp/                              # Runtime: uploaded images + heatmaps
│
├── app.py                             # Main Streamlit application
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

### Model Comparison

| Model | Val Accuracy | Notes |
|---|---|---|
| DenseNet-121 | **Best** | Primary model — selected for deployment |
| ResNet-50 | — | Comparison baseline |
| VGG-16 | — | Comparison baseline |
| EfficientNet-B3 | — | Comparison baseline |

---

## 📂 Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) — Kaggle (Paul Mooney)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Original Size:** ~1.2 GB
- **Reorganised into 3 classes:**
  - `BACTERIA/` — Bacterial pneumonia X-rays
  - `VIRUS/` — Viral pneumonia X-rays
  - `NORMAL/` — Healthy lungs
- **Preprocessing:** Resize to 224×224, normalise to [0, 1], augmentation applied during training

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
git clone https://github.com/<your-username>/xpneumonet.git
cd xpneumonet
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

## 📊 Results

> *Update with your actual evaluation metrics after training*

| Metric | BACTERIA | VIRUS | NORMAL |
|---|---|---|---|
| Precision | — | — | — |
| Recall | — | — | — |
| F1-Score | — | — | — |
| Avg Confidence | ~86% | ~75% | ~94% |

The VIRUS class consistently shows lower confidence — suggesting this class benefits most from additional training data.

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

