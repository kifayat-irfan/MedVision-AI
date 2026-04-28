import streamlit as st
from supabase import create_client, Client
import sqlite3
import datetime

class PatientDB:
    def __init__(self):
        # Logic: Agar secrets mein SUPABASE_URL hai, toh Cloud use karo.
        # Warna Local SQLite use karo.
        try:
            if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
                url = st.secrets["SUPABASE_URL"]
                key = st.secrets["SUPABASE_KEY"]
                self.supabase: Client = create_client(url, key)
                self.mode = "CLOUD"
                print("Connected to Cloud Database (Supabase)")
            else:
                self.mode = "LOCAL"
                self.setup_local()
        except Exception as e:
            print(f"Cloud connection failed, switching to Local: {e}")
            self.mode = "LOCAL"
            self.setup_local()

    def setup_local(self):
        self.conn = sqlite3.connect("medvision_vault.db", check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, modality TEXT, report TEXT, date TEXT)")
        self.conn.commit()

    def save_report(self, patient_id, modality, report):
        if self.mode == "CLOUD":
            try:
                data = {"patient_id": patient_id, "modality": modality, "report": report}
                self.supabase.table("reports").insert(data).execute()
                return True
            except Exception as e:
                print(f"Cloud Save Error: {e}")
                return False
        else:
            try:
                date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                self.conn.execute("INSERT INTO reports (patient_id, modality, report, date) VALUES (?, ?, ?, ?)", 
                                  (patient_id, modality, report, date))
                self.conn.commit()
                return True
            except: return False

    def get_history(self, patient_id):
        if self.mode == "CLOUD":
            try:
                response = self.supabase.table("reports").select("modality, report, created_at").eq("patient_id", patient_id).order("created_at", desc=True).execute()
                return response.data
            except Exception as e:
                print(f"Cloud Fetch Error: {e}")
                return []
        else:
            cursor = self.conn.execute("SELECT modality, report, date FROM reports WHERE patient_id = ? ORDER BY id DESC", (patient_id,))
            return cursor.fetchall()

db = PatientDB()
