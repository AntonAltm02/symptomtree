"""
SymptomSense MVP - High Fidelity Version
Backend: FastAPI + SQLite + Gemini (MedGemma Logic)
"""
import sqlite3
import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai

# -- Config --
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "symptoms.db"
# Replace with your actual key or set environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_ACTUAL_API_KEY"))

# -- Database Architecture --
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            symptom_name TEXT NOT NULL,
            category TEXT NOT NULL,
            severity INTEGER NOT NULL,
            duration TEXT,
            location TEXT,
            notes TEXT,
            source TEXT DEFAULT 'manual',
            hrv_stat INTEGER, -- Heart Rate Variability
            rhr_stat INTEGER  -- Resting Heart Rate
        );
    """)
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="SymptomSense MVP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -- Pydantic Models --
class IntakeRequest(BaseModel):
    message: str
    conversation_history: List[dict] = []
    biometrics: Optional[dict] = None # e.g., {"hrv": 52, "rhr": 68}

class SymptomSave(BaseModel):
    symptom_name: str
    category: str
    severity: int
    duration: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    source: str = "manual"
    hrv_stat: Optional[int] = None
    rhr_stat: Optional[int] = None

# -- AI System Prompt (The "Agentic" Logic) --
SYSTEM_PROMPT = """You are the SymptomSense Clinical Agent. 
Extract structured medical data from the user's narrative.

ALWAYS respond with valid JSON only:
{
  "parsed_symptoms": [
    {
      "symptom_name": "Scientific name",
      "category": "Pain|Respiratory|Digestive|Neurological|Cardiovascular|Mental Health|Other",
      "severity": 1-10,
      "duration": "string",
      "location": "string",
      "notes": "context"
    }
  ],
  "summary": "Brief empathetic clinical summary",
  "follow_up_question": "ONE targeted question to rule out red flags or find triggers"
}

Clinical Rules:
- If chest pain/shortness of breath: severity 9-10.
- Use medical terminology but keep the summary warm.
"""

@app.post("/api/ai/parse")
async def ai_parse(body: IntakeRequest):
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat(history=[{"role": m["role"], "parts": [m["content"]]} for m in body.conversation_history])
    
    response = chat.send_message(body.message)
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(clean_json)
        # Inject Biometrics if provided by the 'wearable' simulation
        if body.biometrics:
            for s in data["parsed_symptoms"]:
                s["hrv_stat"] = body.biometrics.get("hrv")
                s["rhr_stat"] = body.biometrics.get("rhr")
        return data
    except:
        return {"summary": "I'm having trouble parsing that. Could you describe your symptoms again?", "parsed_symptoms": []}

@app.post("/api/symptoms")
async def save_symptom(s: SymptomSave):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now()
    conn.execute("""
        INSERT INTO symptoms (timestamp, date, symptom_name, category, severity, duration, location, notes, source, hrv_stat, rhr_stat)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (now.isoformat(), now.date().isoformat(), s.symptom_name, s.category, s.severity, s.duration, s.location, s.notes, s.source, s.hrv_stat, s.rhr_stat))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/symptoms/stats")
async def get_stats(days: int = 30):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Category Distribution
    by_cat = cursor.execute("SELECT category, COUNT(*) as count FROM symptoms WHERE date >= date('now', ?) GROUP BY category", (f"-{days} days",)).fetchall()
    
    # Severity Trend
    by_day = cursor.execute("SELECT date, AVG(severity) as avg_sev, AVG(hrv_stat) as avg_hrv FROM symptoms WHERE date >= date('now', ?) GROUP BY date ORDER BY date ASC", (f"-{days} days",)).fetchall()
    
    conn.close()
    return {
        "by_category": [dict(r) for r in by_cat],
        "trend": [dict(r) for r in by_day]
    }

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return (BASE_DIR / "index.html").read_text()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)