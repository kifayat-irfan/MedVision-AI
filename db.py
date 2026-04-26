import sqlite3
import datetime

class PatientDB:
    def __init__(self):
        self.conn = sqlite3.connect("medvision_vault.db", check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            modality TEXT,
            report TEXT,
            date TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def save_report(self, patient_id, modality, report):
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.conn.execute("INSERT INTO reports (patient_id, modality, report, date) VALUES (?, ?, ?, ?)", 
                          (patient_id, modality, report, date))
        self.conn.commit()

    def get_history(self, patient_id):
        cursor = self.conn.execute("SELECT modality, report, date FROM reports WHERE patient_id = ? ORDER BY id DESC", (patient_id,))
        return cursor.fetchall()

db = PatientDB()
