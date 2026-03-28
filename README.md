# Pneumonia AI Agent

An AI-based system for detecting pneumonia from chest X-ray images using deep learning, with integrated explainability, database storage, and automated report generation.

---

## Overview

This project presents a complete end-to-end solution for pneumonia detection using medical imaging. The system not only performs classification but also provides interpretability using Grad-CAM, stores patient data in a database, and generates structured diagnostic reports.

---

## Features

- Pneumonia classification using DenseNet121
- Explainable AI using Grad-CAM visualization
- Smart AI-based medical advice generation
- MySQL database integration for storing patient data
- Automated PDF report generation
- Web interface built using Streamlit

---

## Model Performance

- Accuracy: approximately 87%
- Precision, Recall, and F1-score evaluated
- Confusion matrix analysis performed

The model achieves competitive performance while focusing on usability and explainability.

---

## System Architecture

User → Upload X-ray → Model Prediction → Grad-CAM Visualization → AI Advice → Database Storage → PDF Report Generation

---

## Technologies Used

- Deep Learning: TensorFlow, Keras
- Frontend: Streamlit
- Backend: Python
- Database: MySQL (XAMPP)
- Libraries: OpenCV, NumPy, Matplotlib, Seaborn

---

## Project Structure
Pneumonia-AI-Agent/
│
├── app/ # Streamlit application
├── agent/ # AI decision logic
├── database/ # MySQL connection logic
├── model/ # Trained model files
├── notebooks/ # Training and experimentation
├── temp/ # Temporary images and reports
├── README.md
├── requirements.txt


---

## Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/IshitaPadhi/Pneumonia-AI-Agent.git
cd Pneumonia-AI-Agent

Step 2: Create virtual environment
python -m venv .venv
.venv\Scripts\activate
Step 3: Install dependencies
pip install -r requirements.txt
Database Setup
Step 1: Start MySQL using XAMPP

Ensure that MySQL service is running.

Step 2: Create database
CREATE DATABASE pneumonia_db;
Step 3: Create table
CREATE TABLE patients (
    patient_id VARCHAR(50),
    name VARCHAR(100),
    age INT,
    date DATE,
    prediction VARCHAR(50),
    confidence FLOAT,
    severity VARCHAR(50),
    report TEXT,
    image_path TEXT
);
Step 4: Configure environment variables

Create a .env file in the root directory:

DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=pneumonia_db
Running the Application
streamlit run app/app_pro.py

Then open in browser:

http://localhost:8501
Usage
Enter patient details
Upload a chest X-ray image
Run prediction
View:
Predicted class
Confidence score
Grad-CAM heatmap
AI-generated medical advice
Download PDF report
Data is automatically stored in MySQL database
Future Work
Improve model accuracy using EfficientNet or ResNet
Train on larger and more diverse datasets
Deploy system on cloud platforms
Develop mobile or API-based interface
Disclaimer

This system is intended for academic and research purposes only. It is not a substitute for professional medical diagnosis.

Author

Ishita Padhi


---

