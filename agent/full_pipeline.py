from agent.predict import predict_image
from agent.agent import agent_decision
from database.mysql_connect import insert_patient
from datetime import date

def run_pipeline(patient_id, name, age, image_path):
    prediction, confidence = predict_image(image_path)
    severity, report = agent_decision(prediction, confidence)

    data = (
        patient_id,
        name,
        age,
        date.today(),
        prediction,
        float(confidence),
        severity,
        report,
        image_path
    )

    insert_patient(data)

    return prediction, confidence, severity, report