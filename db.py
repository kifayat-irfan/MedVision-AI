import streamlit as st
from supabase import create_client, Client
import sqlite3
import datetime
import traceback

# ==============================================================================
# CRITICAL FIX #1: @st.cache_resource
# Without this, Streamlit creates a NEW Supabase client on EVERY interaction
# (button click, text input, etc.), causing connection exhaustion and race
# conditions. This decorator ensures ONE shared client for the entire session.
# ==============================================================================
@st.cache_resource
def _init_supabase_client() -> Client | None:
    """
    Creates and CACHES a single Supabase client for the entire app session.
    Returns None if secrets are missing (local dev mode).
    """
    try:
        url = st.secrets.get("SUPABASE_URL", "").strip()
        key = st.secrets.get("SUPABASE_KEY", "").strip()

        if not url or not key:
            st.warning(
                "⚠️ [DB] SUPABASE_URL or SUPABASE_KEY is empty in Streamlit Secrets. "
                "Falling back to local SQLite."
            )
            return None

        # Validate URL format to catch common config mistakes early
        if not url.startswith("https://"):
            st.error(
                "❌ [DB] SUPABASE_URL does not start with 'https://'. "
                f"Check your Streamlit Secrets. Value received: '{url[:30]}...'"
            )
            return None

        client: Client = create_client(url, key)
        return client

    except KeyError:
        # st.secrets raises KeyError if the key doesn't exist at all
        return None  # Silently fall back to local — normal for dev environment
    except Exception as e:
        st.error(
            f"❌ [DB] Supabase client creation FAILED.\n"
            f"Error type: `{type(e).__name__}`\n"
            f"Message: `{e}`\n\n"
            f"Full traceback:\n```\n{traceback.format_exc()}\n```"
        )
        return None


class PatientDB:
    """
    Hybrid database class.
    - CLOUD mode: Supabase (PostgreSQL) — used when secrets are present.
    - LOCAL mode: SQLite — used for local development or as a fallback.

    Expected Supabase table schema:
        CREATE TABLE reports (
            id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            patient_id  TEXT NOT NULL,
            modality    TEXT NOT NULL,
            report      TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
    """

    def __init__(self):
        self.supabase: Client | None = None
        self.conn: sqlite3.Connection | None = None
        self.mode = "LOCAL"  # Default to LOCAL; upgrade to CLOUD if client succeeds

        try:
            # Only attempt Supabase if secrets namespace is available
            if hasattr(st, "secrets") and ("SUPABASE_URL" in st.secrets):
                client = _init_supabase_client()
                if client is not None:
                    self.supabase = client
                    self.mode = "CLOUD"

        except Exception as e:
            # Broad catch: never let DB init crash the entire app
            st.warning(
                f"⚠️ [DB] Could not initialize Supabase, using local SQLite. "
                f"Reason: {e}"
            )
            self.mode = "LOCAL"

        # Always set up local SQLite — used as fallback even in CLOUD mode
        self._setup_local()

    # -------------------------------------------------------------------------
    # LOCAL SQLITE SETUP
    # -------------------------------------------------------------------------
    def _setup_local(self):
        """Initialize the local SQLite database and ensure the schema exists."""
        try:
            self.conn = sqlite3.connect(
                "medvision_vault.db", check_same_thread=False
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id  TEXT NOT NULL,
                    modality    TEXT NOT NULL,
                    report      TEXT NOT NULL,
                    date        TEXT NOT NULL
                )
                """
            )
            self.conn.commit()
        except Exception as e:
            st.error(f"❌ [DB] SQLite setup failed: {e}")

    # -------------------------------------------------------------------------
    # SAVE REPORT
    # -------------------------------------------------------------------------
    def save_report(self, patient_id: str, modality: str, report: str) -> bool:
        """
        Saves a diagnostic report.
        - Tries Supabase first in CLOUD mode.
        - Falls back to SQLite on any Supabase failure.
        Returns True on success, False on total failure.
        """
        if self.mode == "CLOUD" and self.supabase is not None:
            return self._save_cloud(patient_id, modality, report)
        else:
            return self._save_local(patient_id, modality, report)

    def _save_cloud(self, patient_id: str, modality: str, report: str) -> bool:
        """
        Inserts a record into the Supabase 'reports' table.
        Includes explicit response validation and verbose error logging.
        """
        try:
            payload = {
                "patient_id": str(patient_id).strip(),
                "modality":   str(modality).strip(),
                "report":     str(report).strip(),
                # 'created_at' is omitted — Supabase fills it with NOW()
            }

            response = self.supabase.table("reports").insert(payload).execute()

            # ==================================================================
            # CRITICAL FIX #2: Validate the response.
            # A failed insert (wrong column, RLS ghost, type mismatch) can
            # return an empty response.data WITHOUT raising an exception.
            # We must check explicitly.
            # ==================================================================
            if response.data and len(response.data) > 0:
                inserted_id = response.data[0].get("id", "N/A")
                st.success(
                    f"✅ [DB] Report synced to Supabase Cloud. "
                    f"Row ID: `{inserted_id}` | Patient: `{patient_id}` | Modality: `{modality}`"
                )
                return True
            else:
                # Insert didn't raise an error but returned nothing — log everything
                st.error(
                    f"❌ [DB] Supabase INSERT returned empty data — insert likely failed silently.\n\n"
                    f"**Full API Response:** `{response}`\n\n"
                    f"**Possible causes:**\n"
                    f"- Column name mismatch (check `patient_id`, `modality`, `report` exist in your table)\n"
                    f"- A ghost RLS policy is active (run `SELECT * FROM pg_policies WHERE tablename='reports';` in Supabase SQL Editor)\n"
                    f"- Wrong API key (use `service_role` key for server-side inserts)\n"
                    f"- `report` column has a length constraint being violated\n"
                )
                # Fallback to local so data is not lost
                st.warning("⚠️ [DB] Falling back to local SQLite to preserve data.")
                return self._save_local(patient_id, modality, report)

        except Exception as e:
            # Full diagnostic dump — this is intentionally verbose
            st.error(
                f"❌ [DB] SUPABASE SAVE EXCEPTION\n\n"
                f"**Error Type:** `{type(e).__name__}`\n"
                f"**Message:** `{str(e)}`\n\n"
                f"**Diagnostic Checklist:**\n"
                f"1. Is `SUPABASE_URL` correct? (Should end with `.supabase.co`)\n"
                f"2. Are you using the `service_role` key, not the `anon` key?\n"
                f"3. Is the `reports` table named exactly `reports` (lowercase)?\n"
                f"4. Do columns `patient_id`, `modality`, `report` all exist?\n\n"
                f"**Full Traceback:**\n```\n{traceback.format_exc()}\n```"
            )
            st.warning("⚠️ [DB] Falling back to local SQLite to preserve data.")
            return self._save_local(patient_id, modality, report)

    def _save_local(self, patient_id: str, modality: str, report: str) -> bool:
        """Saves a report to the local SQLite database."""
        if self.conn is None:
            self._setup_local()
        try:
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            self.conn.execute(
                "INSERT INTO reports (patient_id, modality, report, date) VALUES (?, ?, ?, ?)",
                (patient_id, modality, report, date),
            )
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"❌ [DB] SQLite save failed: {e}\n{traceback.format_exc()}")
            return False

    # -------------------------------------------------------------------------
    # GET HISTORY
    # -------------------------------------------------------------------------
    def get_history(self, patient_id: str) -> list[tuple[str, str, str]]:
        """
        Returns a list of (modality, report, date_string) tuples for a patient.
        Returns a CONSISTENT TUPLE FORMAT regardless of CLOUD or LOCAL mode.

        CRITICAL FIX #3: The original CLOUD path returned raw dicts from Supabase,
        but app.py unpacked them as tuples — causing a silent ValueError crash.
        This method now always returns list[tuple[str, str, str]].
        """
        if self.mode == "CLOUD" and self.supabase is not None:
            return self._get_history_cloud(patient_id)
        else:
            return self._get_history_local(patient_id)

    def _get_history_cloud(self, patient_id: str) -> list[tuple[str, str, str]]:
        try:
            response = (
                self.supabase.table("reports")
                .select("modality, report, created_at")
                .eq("patient_id", str(patient_id).strip())
                .order("created_at", desc=True)
                .execute()
            )

            results = []
            for row in response.data:
                modality    = row.get("modality", "Unknown")
                report_text = row.get("report", "")
                raw_date    = row.get("created_at", "")
                # Trim ISO timestamp to "YYYY-MM-DD HH:MM" for display
                display_date = raw_date[:16].replace("T", " ") if raw_date else "N/A"
                results.append((modality, report_text, display_date))

            return results

        except Exception as e:
            st.error(
                f"❌ [DB] Cloud history fetch failed.\n"
                f"Error: `{e}`\n\n"
                f"Traceback:\n```\n{traceback.format_exc()}\n```"
            )
            return []

    def _get_history_local(self, patient_id: str) -> list[tuple[str, str, str]]:
        if self.conn is None:
            self._setup_local()
        try:
            cursor = self.conn.execute(
                "SELECT modality, report, date FROM reports "
                "WHERE patient_id = ? ORDER BY id DESC",
                (patient_id,),
            )
            return cursor.fetchall()
        except Exception as e:
            st.error(f"❌ [DB] SQLite history fetch failed: {e}")
            return []


# Singleton instance — imported by app.py as `from db import db`
db = PatientDB()