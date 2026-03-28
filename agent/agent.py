def agent_decision(prediction, confidence):
    if confidence < 0.6:
        severity = "Uncertain"
        report = "Low confidence. Retake X-ray."

    elif prediction == "BACTERIA" and confidence > 0.85:
        severity = "High"
        report = "High risk bacterial pneumonia. Consult doctor immediately."

    elif prediction == "BACTERIA":
        severity = "Medium"
        report = "Bacterial pneumonia detected. Medication required."

    elif prediction == "VIRUS":
        severity = "Medium"
        report = "Viral pneumonia detected. Rest and medication advised."

    else:
        severity = "Low"
        report = "No pneumonia detected."

    return severity, report