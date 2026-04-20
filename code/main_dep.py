"""
SymptomSense — AI-powered symptom tracker
Backend: FastAPI + SQLite + Claude AI + ReportLab PDF
"""
import sqlite3
import json
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.parse

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import google.generativeai as genai

import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "symptoms.db"
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            symptom_name TEXT NOT NULL,
            category TEXT NOT NULL,
            severity INTEGER NOT NULL CHECK(severity BETWEEN 1 AND 10),
            duration TEXT,
            location TEXT,
            notes TEXT,
            tags TEXT,
            source TEXT DEFAULT 'manual'
        );
        CREATE TABLE IF NOT EXISTS ai_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_input TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            extracted_symptoms TEXT
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="SymptomSense")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Pydantic Models ───────────────────────────────────────────────────────────
class SymptomEntry(BaseModel):
    symptom_name: str
    category: str
    severity: int
    duration: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[str] = None
    source: str = "manual"

class AIMessage(BaseModel):
    message: str
    conversation_history: list = []

class ConfirmSymptoms(BaseModel):
    symptoms: list
    original_message: str
    ai_response: str

# ── Symptom Categories ────────────────────────────────────────────────────────
CATEGORIES = [
    "Pain", "Respiratory", "Digestive", "Neurological", "Cardiovascular",
    "Skin", "Musculoskeletal", "Mental Health", "Sleep", "Energy", "Other"
]

# ── AI Integration ─────────────────────────────────────────────────────────────
def call_claude(messages: list, system: str) -> str:
    """Call Anthropic Claude API directly via HTTP."""
    if not ANTHROPIC_API_KEY:
        return json.dumps({
            "parsed_symptoms": [],
            "follow_up_question": "Please set your ANTHROPIC_API_KEY environment variable to enable AI features.",
            "needs_clarification": False,
            "summary": "AI not configured"
        })
    
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system": system,
        "messages": messages
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data["content"][0]["text"]
    except Exception as e:
        return json.dumps({
            "parsed_symptoms": [],
            "follow_up_question": f"AI error: {str(e)}",
            "needs_clarification": False,
            "summary": "Error"
        })
    
def call_medgemma(messages: list, system: str) -> str:
    import requests, certifi, os, time

    HF_TOKEN = os.environ.get("HF_TOKEN", "")

    # Manually apply the chat template as a string
    prompt = f"<start_of_turn>system\n{system}<end_of_turn>\n"
    for m in messages:
        role = m["role"]
        content = m["content"]
        prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"

    for attempt in range(3):
        r = requests.post(
            "https://api-inference.huggingface.co/models/google/medgemma-4b-it",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.3,
                    "return_full_text": False  # only return new tokens
                }
            },
            timeout=60,
            verify=certifi.where()
        )
        if r.status_code == 503:
            wait = r.json().get("estimated_time", 20)
            print(f"Model loading, retrying in {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()[0]["generated_text"]
    
genai.configure(api_key='AQ.Ab8RN6LXgIrI0DgsgaZIUIhyLNqYWPaGAnxJtz5cQM-ynsDoqg')

def call_gemini(messages: list, system: str) -> str:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system
    )
    # Convert your messages format to Gemini's
    history = [
        {"role": m["role"], "parts": [m["content"]]}
        for m in messages[:-1]  # all but last
    ]
    chat = model.start_chat(history=history)
    response = chat.send_message(messages[-1]["content"])
    return response.text

SYSTEM_PROMPT = """You are SymptomSense AI, a compassionate and precise medical symptom parsing assistant. 
Your role is to extract structured symptom data from natural language descriptions.

ALWAYS respond with valid JSON only (no markdown, no preamble):
{
  "parsed_symptoms": [
    {
      "symptom_name": "string",
      "category": "one of: Pain|Respiratory|Digestive|Neurological|Cardiovascular|Skin|Musculoskeletal|Mental Health|Sleep|Energy|Other",
      "severity": number 1-10,
      "duration": "string or null",
      "location": "string or null",
      "notes": "string or null",
      "tags": "comma-separated string or null"
    }
  ],
  "needs_clarification": boolean,
  "follow_up_question": "string or null — ask ONE specific question if clarification improves data quality",
  "summary": "brief friendly summary of what you understood"
}

Guidelines:
- Infer severity from language: 'mild/slight'=2-3, 'moderate/some'=4-6, 'severe/strong'=7-8, 'worst/unbearable'=9-10
- If severity is unclear, ask about it
- Be empathetic and health-focused
- Never diagnose — only categorize and record
- Ask follow-up only if it significantly improves data quality"""

@app.post("/api/ai/parse")
async def ai_parse_symptoms(body: AIMessage):
    messages = body.conversation_history + [{"role": "user", "content": body.message}]
    # raw = call_claude(messages, SYSTEM_PROMPT)
    raw = call_gemini(messages, SYSTEM_PROMPT)
    
    # Clean up potential markdown fences
    clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    
    try:
        parsed = json.loads(clean)
    except Exception:
        parsed = {
            "parsed_symptoms": [],
            "needs_clarification": False,
            "follow_up_question": None,
            "summary": "I had trouble understanding that. Could you describe your symptoms more specifically?"
        }
    
    # Log session
    conn = get_db()
    conn.execute(
        "INSERT INTO ai_sessions (timestamp, user_input, ai_response, extracted_symptoms) VALUES (?,?,?,?)",
        (datetime.now().isoformat(), body.message, raw,
         json.dumps(parsed.get("parsed_symptoms", [])))
    )
    conn.commit()
    conn.close()
    
    return parsed

@app.post("/api/symptoms")
async def add_symptom(entry: SymptomEntry):
    conn = get_db()
    now = datetime.now()
    conn.execute(
        """INSERT INTO symptoms 
           (timestamp, date, symptom_name, category, severity, duration, location, notes, tags, source)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (now.isoformat(), now.date().isoformat(),
         entry.symptom_name, entry.category, entry.severity,
         entry.duration, entry.location, entry.notes, entry.tags, entry.source)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": f"Logged: {entry.symptom_name}"}

@app.post("/api/symptoms/batch")
async def add_symptoms_batch(body: ConfirmSymptoms):
    conn = get_db()
    now = datetime.now()
    added = []
    for s in body.symptoms:
        conn.execute(
            """INSERT INTO symptoms 
               (timestamp, date, symptom_name, category, severity, duration, location, notes, tags, source)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (now.isoformat(), now.date().isoformat(),
             s.get("symptom_name", "Unknown"),
             s.get("category", "Other"),
             max(1, min(10, int(s.get("severity", 5)))),
             s.get("duration"), s.get("location"),
             s.get("notes"), s.get("tags"), "ai")
        )
        added.append(s.get("symptom_name"))
    conn.commit()
    conn.close()
    return {"status": "ok", "added": added}

@app.get("/api/symptoms")
async def get_symptoms(days: int = 30, category: str = None):
    conn = get_db()
    if category:
        rows = conn.execute(
            "SELECT * FROM symptoms WHERE date >= date('now', ?) AND category=? ORDER BY timestamp DESC",
            (f"-{days} days", category)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM symptoms WHERE date >= date('now', ?) ORDER BY timestamp DESC",
            (f"-{days} days",)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/symptoms/stats")
async def get_stats(days: int = 30):
    conn = get_db()
    
    # By category
    by_cat = conn.execute("""
        SELECT category, COUNT(*) as count, AVG(severity) as avg_severity
        FROM symptoms WHERE date >= date('now', ?)
        GROUP BY category ORDER BY count DESC
    """, (f"-{days} days",)).fetchall()
    
    # By day (last 14 days)
    by_day = conn.execute("""
        SELECT date, COUNT(*) as count, AVG(severity) as avg_severity
        FROM symptoms WHERE date >= date('now', '-14 days')
        GROUP BY date ORDER BY date ASC
    """).fetchall()
    
    # Top symptoms
    top = conn.execute("""
        SELECT symptom_name, COUNT(*) as count, AVG(severity) as avg_severity
        FROM symptoms WHERE date >= date('now', ?)
        GROUP BY symptom_name ORDER BY count DESC LIMIT 10
    """, (f"-{days} days",)).fetchall()
    
    # Severity distribution
    severity_dist = conn.execute("""
        SELECT severity, COUNT(*) as count
        FROM symptoms WHERE date >= date('now', ?)
        GROUP BY severity ORDER BY severity
    """, (f"-{days} days",)).fetchall()
    
    conn.close()
    return {
        "by_category": [dict(r) for r in by_cat],
        "by_day": [dict(r) for r in by_day],
        "top_symptoms": [dict(r) for r in top],
        "severity_distribution": [dict(r) for r in severity_dist]
    }

@app.delete("/api/symptoms/{symptom_id}")
async def delete_symptom(symptom_id: int):
    conn = get_db()
    conn.execute("DELETE FROM symptoms WHERE id=?", (symptom_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/api/export/json")
async def export_json(days: int = 90):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM symptoms WHERE date >= date('now', ?) ORDER BY timestamp DESC",
        (f"-{days} days",)
    ).fetchall()
    conn.close()
    
    data = {
        "export_date": datetime.now().isoformat(),
        "period_days": days,
        "total_entries": len(rows),
        "symptoms": [dict(r) for r in rows],
        "fhir_note": "Data structured for FHIR R4 Observation resource compatibility"
    }
    
    path = EXPORT_DIR / f"symptoms_{date.today().isoformat()}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    return FileResponse(path, filename=f"symptom_export_{date.today().isoformat()}.json",
                        media_type="application/json")

@app.get("/api/export/pdf")
async def export_pdf(days: int = 30):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM symptoms WHERE date >= date('now', ?) ORDER BY timestamp DESC",
        (f"-{days} days",)
    ).fetchall()
    stats = conn.execute("""
        SELECT category, COUNT(*) as count, AVG(severity) as avg_sev
        FROM symptoms WHERE date >= date('now', ?)
        GROUP BY category ORDER BY count DESC
    """, (f"-{days} days",)).fetchall()
    conn.close()

    path = EXPORT_DIR / f"symptom_report_{date.today().isoformat()}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    TEAL = colors.HexColor("#0D9488")
    LIGHT = colors.HexColor("#F0FDFA")
    DARK  = colors.HexColor("#134E4A")
    GRAY  = colors.HexColor("#6B7280")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", fontSize=24, textColor=DARK,
                                  spaceAfter=4, fontName="Helvetica-Bold", alignment=TA_LEFT)
    sub_style   = ParagraphStyle("sub", fontSize=11, textColor=GRAY,
                                  spaceAfter=12, fontName="Helvetica")
    h2_style    = ParagraphStyle("h2", fontSize=14, textColor=TEAL,
                                  spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")
    body_style  = ParagraphStyle("body", fontSize=10, textColor=DARK,
                                  spaceAfter=6, fontName="Helvetica", leading=14)

    story = []
    story.append(Paragraph("SymptomSense", title_style))
    story.append(Paragraph(f"Health Report — Last {days} days | Generated {datetime.now().strftime('%B %d, %Y')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL))
    story.append(Spacer(1, 8*mm))

    # Summary
    story.append(Paragraph("Summary", h2_style))
    story.append(Paragraph(f"Total symptom entries: <b>{len(rows)}</b>", body_style))
    if rows:
        avg_sev = sum(r["severity"] for r in rows) / len(rows)
        story.append(Paragraph(f"Average severity: <b>{avg_sev:.1f} / 10</b>", body_style))
    story.append(Spacer(1, 6*mm))

    # Category breakdown
    if stats:
        story.append(Paragraph("By Category", h2_style))
        tdata = [["Category", "Entries", "Avg Severity"]]
        for s in stats:
            tdata.append([s["category"], str(s["count"]), f"{s['avg_sev']:.1f}"])
        t = Table(tdata, colWidths=[80*mm, 40*mm, 40*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), TEAL),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, colors.white]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D1FAF6")),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 6*mm))

    # Recent entries
    if rows:
        story.append(Paragraph("Recent Entries", h2_style))
        tdata = [["Date", "Symptom", "Category", "Severity", "Notes"]]
        for r in rows[:30]:
            note = (r["notes"] or "")[:40]
            tdata.append([
                r["date"],
                r["symptom_name"][:25],
                r["category"],
                str(r["severity"]),
                note
            ])
        t = Table(tdata, colWidths=[28*mm, 45*mm, 35*mm, 22*mm, 40*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), TEAL),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, colors.white]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D1FAF6")),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(t)

    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=GRAY))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "This report is for personal health tracking only and does not constitute medical advice. "
        "Data exported in FHIR-compatible structure. Please consult a healthcare professional for medical guidance.",
        ParagraphStyle("footer", fontSize=8, textColor=GRAY, fontName="Helvetica-Oblique")
    ))

    doc.build(story)
    return FileResponse(str(path), filename=f"symptom_report_{date.today().isoformat()}.pdf",
                        media_type="application/pdf")

# ── Serve Frontend ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = BASE_DIR / "index.html"
    return HTMLResponse(html_path.read_text())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
