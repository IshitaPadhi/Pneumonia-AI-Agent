# Pneumonia-AI-agent
FULL MODEL : 
User Upload X-ray
        ↓
CNN Model → Prediction
        ↓
Grad-CAM → Show infected area
        ↓
AI Agent → Decision + Advice
        ↓
MySQL → Store patient history
        ↓
Streamlit → Show report


How the CNN model works : X-ray Image → Detect edges → Detect shapes → Detect lung patterns → Detect infection → Classify

Grad-CAM shows: Where the model is looking in the X-ray before making decision
So output becomes:  Input	Output
                    X-ray	Heatmap on infected region
 

MySQL :  Patients table to store patient medical history of all scans

| Column     | Meaning               |
| ---------- | --------------------- |
| patient_id | Unique ID             |
| name       | Patient name          |
| age        | Age                   |
| date       | Visit date            |
| prediction | Bacteria/Virus/Normal |
| confidence | Model confidence      |
| severity   | High/Medium/Low       |
| report     | AI agent advice       |
| image_path | X-ray file            |

To run : streamlit run app/app.py
