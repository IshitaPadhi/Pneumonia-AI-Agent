import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    try:
        db = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        return db
    except Exception as e:
        print("❌ MySQL Connection Error:", e)
        return None


def insert_patient(data):
    db = get_connection()

    if db is None:
        raise Exception("Database connection failed")

    cursor = db.cursor()

    query = """
    INSERT INTO patients
    (patient_id, name, age, date, prediction, confidence, severity, report, image_path)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    cursor.execute(query, data)
    db.commit()

    cursor.close()
    db.close()