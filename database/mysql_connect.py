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


def get_recent_patients(limit=20):
    """Return most recent N patient records."""
    db = get_connection()
    if db is None:
        return []
    cursor = db.cursor()
    cursor.execute("""
        SELECT patient_id, name, age, date, prediction, confidence, severity, report
        FROM patients
        ORDER BY date DESC
        LIMIT %s
    """, (limit,))
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return rows


def get_prediction_stats():
    """Count of each prediction class."""
    db = get_connection()
    if db is None:
        return []
    cursor = db.cursor()
    cursor.execute("""
        SELECT prediction, COUNT(*) as cnt
        FROM patients
        GROUP BY prediction
    """)
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return rows


def get_avg_confidence_by_class():
    """Average model confidence per prediction class."""
    db = get_connection()
    if db is None:
        return []
    cursor = db.cursor()
    cursor.execute("""
        SELECT prediction, AVG(confidence) as avg_conf
        FROM patients
        GROUP BY prediction
    """)
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return rows


def get_total_count():
    """Total number of scans performed."""
    db = get_connection()
    if db is None:
        return 0
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM patients")
    result = cursor.fetchone()
    cursor.close()
    db.close()
    return result[0] if result else 0


def get_severity_distribution():
    """Count per severity level."""
    db = get_connection()
    if db is None:
        return []
    cursor = db.cursor()
    cursor.execute("""
        SELECT severity, COUNT(*) as cnt
        FROM patients
        GROUP BY severity
    """)
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return rows


def search_patient(query_str):
    """Search patients by name or patient_id."""
    db = get_connection()
    if db is None:
        return []
    cursor = db.cursor()
    like = f"%{query_str}%"
    cursor.execute("""
        SELECT patient_id, name, age, date, prediction, confidence, severity, report
        FROM patients
        WHERE name LIKE %s OR patient_id LIKE %s
        ORDER BY date DESC
    """, (like, like))
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return rows