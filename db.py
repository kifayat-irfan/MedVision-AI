import streamlit as st
from supabase import create_client, Client
import sqlite3
import datetime

class PatientDB:
    def __init__(self):
        try:
            # Priority: Try Cloud first
            if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
                url = st.secrets["SUPABASE_URL"]
                key = st.secrets["SUPABASE_KEY"]
                self.supabase: Client = create_client(url, key)
                self.mode = "CLOUD"
            else:
                self.mode = "LOCAL"
                self.setup_local()
        except Exception as e:
            st.error(f"🚨 Critical DB Init Error: {e}")
            self.mode = "LOCAL"
            self.setup_local()

    def setup_local(self):
        self.conn = sqlite3.connect("medvision_vault.db", check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, modality TEXT, report TEXT, date TEXT)")
        self.conn.commit()

    def save_report(self, patient_id, modality, report):
        if self.mode == "CLOUD":
            try:
                # EXACT column names matching the Supabase Table
                data = {
                    "patient_id": str(patient_id), 
                    "modality": str(modality), 
                    "report": str(report)
                }
                # We use .execute() to force the request
                self.supabase.table("reports").insert(data).execute()
                return True
            except Exception as e:
                # WE WANT TO SEE THIS ERROR!
                st.error(f"❌ SUPABASE ERROR: {str(e)}") 
                return False
        else:
            try:
                date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                self.conn.execute("INSERT INTO reports (patient_id, modality, report, date) VALUES (?, ?, ?, ?)", 
                                  (patient_id, modality, report, date))
                self.conn.commit()
                return True
            except Exception as e:
                st.error(f"Local DB Error: {e}")
                return False

    def get_history(self, patient_id):
        if self.mode == "CLOUD":
            try:
                response = self.supabase.table("reports").select("modality, report, created_at").eq("patient_id", patient_id).order("created_at", desc=True).execute()
                return response.data
            except Exception as e:
                st.error(f"Cloud Fetch Error: {e}")
                return []
        else:
            cursor = self.conn.execute("SELECT modality, report, date FROM reports WHERE patient_id = ? ORDER BY id DESC", (patient_id,))
            return cursor.fetchall()

db = PatientDB()
