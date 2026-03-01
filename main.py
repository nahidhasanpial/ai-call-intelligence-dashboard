import os
import re
from datetime import datetime
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import whisper

# OPTIONAL (if ffmpeg path issues happen again):
# os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load Whisper model once (offline)
# Options: tiny, base, small, medium, large
model = whisper.load_model("base")

LEADS = []  # temporary in-memory store (later you can move to SQLite)


def safe_filename(name: str) -> str:
    """Make filename safe for Windows paths."""
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    if not name:
        name = "audio"
    return name


def transcribe_audio(file_path: str) -> str:
    """Offline speech-to-text using Whisper."""
    result = model.transcribe(file_path)
    return (result.get("text") or "").strip()


def extract_money(text: str) -> str:
    """
    Extract budget-like info from transcript.
    Examples it catches: 10000, 10,000, 10k, 10 k, 10 thousand
    """
    t = text.lower()

    # 10k / 10 k
    m = re.search(r"\b(\d{1,3})\s*k\b", t)
    if m:
        return f"{m.group(1)}k"

    # 10,000 / 10000 / 120000
    m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{4,})\b", t)
    if m:
        return m.group(1).replace(",", "")

    # "ten thousand" basic (simple)
    if "thousand" in t:
        m = re.search(r"\b(\d{1,3})\s*thousand\b", t)
        if m:
            return f"{m.group(1)}000"

    return "unknown"


def extract_timeline(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["today", "aj", "aaj"]):
        return "today"
    if any(x in t for x in ["tomorrow", "kal"]):
        return "tomorrow"
    if "next week" in t or "7 days" in t or "within a week" in t:
        return "within_7_days"
    if "next month" in t or "within a month" in t:
        return "within_30_days"
    if any(x in t for x in ["later", "pore", "por"]):
        return "later"
    return "unknown"


def extract_intent(text: str, product: str = "") -> str:
    t = (product + " " + text).lower()

    # Education/coaching examples
    if "ielts" in t:
        return "IELTS_inquiry"
    if any(x in t for x in ["admission", "apply", "university", "visa", "study abroad"]):
        return "admission_or_study_inquiry"

    # Real estate examples
    if any(x in t for x in ["flat", "apartment", "plot", "land", "rent", "basha"]):
        return "real_estate_inquiry"

    # Complaints
    if any(x in t for x in ["complain", "problem", "issue", "refund", "bad", "worst"]):
        return "complaint"

    return "general_inquiry"


def lead_scoring(transcript: str, product: str, form_budget: str, form_timeline: str) -> dict:
    """
    Simple explainable scoring:
    - Budget mentioned -> +25
    - Urgency within 7 days -> +20
    - Product fit clear -> +20
    - Decision maker signals -> +15
    - Positive sentiment -> +10
    - Asked specific questions -> +10
    """
    t = transcript.lower()

    score = 0
    reasons = []

    # Product fit
    intent = extract_intent(transcript, product)
    if intent != "general_inquiry":
        score += 20
        reasons.append("Product fit is clear (+20)")

    # Budget
    extracted_budget = extract_money(transcript)
    budget_value = form_budget.strip() if form_budget.strip() else extracted_budget
    if budget_value != "unknown":
        score += 25
        reasons.append("Budget mentioned (+25)")

    # Urgency
    extracted_timeline = extract_timeline(transcript)
    timeline_value = form_timeline.strip() if form_timeline.strip() else extracted_timeline
    if timeline_value in ["today", "tomorrow", "within_7_days"]:
        score += 20
        reasons.append("Urgent timeline (+20)")
    elif timeline_value in ["within_30_days"]:
        score += 10
        reasons.append("Near timeline (+10)")

    # Decision maker (simple heuristic)
    if any(x in t for x in ["i will", "i want", "i need", "ami chai", "ami nibo", "i am ready"]):
        score += 15
        reasons.append("Likely decision maker (+15)")

    # Positive sentiment
    if any(x in t for x in ["please", "thanks", "thank you", "great", "good", "valo", "dhonnobad"]):
        score += 10
        reasons.append("Positive sentiment (+10)")

    # Specific questions
    if any(x in t for x in ["price", "fee", "cost", "schedule", "time", "batch", "class", "documents", "location"]):
        score += 10
        reasons.append("Asked specific questions (+10)")

    score = min(score, 100)

    label = "cold"
    if score >= 70:
        label = "hot"
    elif score >= 40:
        label = "warm"

    # Next action suggestion
    next_action = "Send details and follow up."
    if label == "hot":
        next_action = "Call within 1 hour and send payment/admission steps."
    elif label == "warm":
        next_action = "Send brochure/details and follow up within 24 hours."
    else:
        next_action = "Send basic info; follow up if they respond."

    # Short summary bullets (simple)
    bullets = []
    if intent != "general_inquiry":
        bullets.append(f"Intent: {intent.replace('_', ' ')}")
    if budget_value != "unknown":
        bullets.append(f"Budget: {budget_value}")
    if timeline_value != "unknown":
        bullets.append(f"Timeline: {timeline_value}")
    bullets.append("Customer asked questions and wants information.")

    return {
        "intent": intent,
        "lead_score_0_100": score,
        "lead_label": label,
        "budget": budget_value,
        "timeline": timeline_value,
        "summary_bullets": bullets[:5],
        "next_action": next_action,
        "reasons": reasons
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(
    audio: UploadFile,
    name: str = Form(""),
    phone: str = Form(""),
    product: str = Form(""),
    source: str = Form("call"),
    budget: str = Form(""),
    timeline: str = Form("")
):
    os.makedirs("uploads", exist_ok=True)

    # Make filename safe
    original = audio.filename or "audio"
    filename = safe_filename(original)
    # Add timestamp to avoid overwrite
    stamped_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    path = os.path.join("uploads", stamped_name)

    # Save file
    with open(path, "wb") as f:
        f.write(await audio.read())

    # Transcribe + Analyze
    transcript = ""
    analysis = {}
    try:
        transcript = transcribe_audio(path)
        analysis = lead_scoring(transcript, product, budget, timeline)
    except Exception as e:
        transcript = f"TRANSCRIPTION ERROR: {str(e)}"
        analysis = {"error": str(e)}

    LEADS.append({
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "phone": phone,
        "product": product,
        "source": source,
        "budget": budget,
        "timeline": timeline,
        "filename": stamped_name,
        "transcript": transcript,
        "analysis": analysis
    })

    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    # newest first
    leads = list(reversed(LEADS))
    return templates.TemplateResponse("dashboard.html", {"request": request, "leads": leads})