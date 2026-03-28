from mysql_connect import insert_patient

data = (
    "P001",
    "Test Patient",
    30,
    "2026-03-26",
    "NORMAL",
    0.95,
    "Low",
    "No pneumonia detected",
    "test.jpg"
)

insert_patient(data)
print("Data inserted successfully")