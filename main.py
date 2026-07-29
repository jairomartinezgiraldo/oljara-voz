from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import httpx
import uuid
import os
import json
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors

# ── Configuración ──────────────────────────────────────────────────────────────
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_DEVICE  = os.environ.get("WHISPER_DEVICE", "auto")   # auto | cpu | cuda
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3")
MAX_FILE_MB     = int(os.environ.get("MAX_FILE_MB", "200"))
UPLOAD_DIR      = Path("/app/uploads")
RESULTS_DIR     = Path("/app/results")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="OljaraVoz")
app.state.limiter = limiter

# Almacén de jobs en memoria
jobs: dict[str, dict] = {}

# Modelo Whisper (se carga una vez al inicio)
_whisper_model: WhisperModel | None = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        device = WHISPER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    return _whisper_model

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"error": "Demasiadas solicitudes. Espera un momento."})

# ── Frontend ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

# ── Upload y procesamiento ─────────────────────────────────────────────────────
ALLOWED_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/m4a", "audio/mp4", "video/mp4",
    "audio/webm", "video/webm", "audio/flac", "audio/x-flac",
}
ALLOWED_EXT = {".mp3", ".wav", ".ogg", ".m4a", ".mp4", ".webm", ".flac", ".opus"}

@app.post("/upload")
@limiter.limit("10/minute")
async def upload(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validar extensión
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Formato no soportado. Usa: {', '.join(ALLOWED_EXT)}")

    # Leer y validar tamaño
    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"El archivo supera los {MAX_FILE_MB} MB.")

    job_id = str(uuid.uuid4())
    audio_path = UPLOAD_DIR / f"{job_id}{ext}"
    audio_path.write_bytes(content)

    jobs[job_id] = {
        "status": "processing",
        "step": "Transcribiendo audio...",
        "progress": 10,
        "filename": file.filename,
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }

    background_tasks.add_task(process_audio, job_id, audio_path)
    return JSONResponse({"job_id": job_id})

async def process_audio(job_id: str, audio_path: Path):
    try:
        # ── 1. Transcripción ───────────────────────────────────────────────────
        _update_job(job_id, step="Transcribiendo audio con Whisper...", progress=20)
        loop = asyncio.get_event_loop()
        transcripcion = await loop.run_in_executor(None, transcribe, audio_path)

        # ── 2. Análisis con LLM ────────────────────────────────────────────────
        _update_job(job_id, step="Analizando con IA...", progress=60)
        analisis = await analizar_con_ollama(transcripcion)

        # ── 3. Generar PDF ─────────────────────────────────────────────────────
        _update_job(job_id, step="Generando PDF...", progress=85)
        filename = jobs[job_id]["filename"]
        pdf_path = await loop.run_in_executor(
            None, generar_pdf, job_id, filename, transcripcion, analisis
        )

        jobs[job_id].update({
            "status": "done",
            "step": "Listo",
            "progress": 100,
            "result": {
                "transcripcion": transcripcion,
                "resumen": analisis.get("resumen", []),
                "compromisos": analisis.get("compromisos", []),
                "acta": analisis.get("acta", ""),
            },
            "pdf": str(pdf_path),
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "step": "Error",
            "error": str(e),
        })
    finally:
        if audio_path.exists():
            audio_path.unlink()

def _update_job(job_id, **kwargs):
    jobs[job_id].update(kwargs)

def transcribe(audio_path: Path) -> str:
    model = get_whisper()
    segments, _ = model.transcribe(
        str(audio_path),
        language="es",
        beam_size=1,           # más rápido, calidad similar para español
        vad_filter=True,       # salta silencios automáticamente
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,  # evita repeticiones, más rápido
        temperature=0,         # determinístico, sin retries
    )
    return " ".join(seg.text.strip() for seg in segments)

async def analizar_con_ollama(transcripcion: str) -> dict:
    prompt = f"""Eres un asistente experto en análisis de reuniones de negocios en Colombia.
Analiza la siguiente transcripción y responde ÚNICAMENTE con un JSON válido con estas claves:

- "resumen": lista de 3 a 5 strings con los puntos clave discutidos
- "compromisos": lista de strings con tareas o compromisos acordados (incluye responsable y fecha si se menciona)
- "acta": string con un acta formal de la reunión lista para ser firmada, en formato de texto plano con saltos de línea

Transcripción:
\"\"\"
{transcripcion[:6000]}
\"\"\"

Responde SOLO el JSON, sin texto adicional."""

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            # Extraer JSON del response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                if isinstance(data, dict):
                    return data
            # Ollama respondió pero sin JSON válido
            return {
                "resumen": [f"RAW OLLAMA: {raw[:500]}"],
                "compromisos": [],
                "acta": f"TRANSCRIPCIÓN:\n\n{transcripcion}",
            }
    except Exception as e:
        raise RuntimeError(f"Ollama ({OLLAMA_URL}): {e}")

def generar_pdf(job_id: str, filename: str, transcripcion: str, analisis: dict) -> Path:
    output = RESULTS_DIR / f"{job_id}.pdf"
    doc = SimpleDocTemplate(
        str(output), pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    AZUL = colors.HexColor("#1A56DB")
    GRIS = colors.HexColor("#6B6B65")

    titulo = ParagraphStyle("titulo", parent=styles["Title"],
        fontSize=22, textColor=AZUL, spaceAfter=6, fontName="Helvetica-Bold")
    subtitulo = ParagraphStyle("subtitulo", parent=styles["Heading2"],
        fontSize=13, textColor=AZUL, spaceBefore=16, spaceAfter=6, fontName="Helvetica-Bold")
    normal = ParagraphStyle("normal", parent=styles["Normal"],
        fontSize=10, leading=16, textColor=colors.HexColor("#111110"))
    meta = ParagraphStyle("meta", parent=styles["Normal"],
        fontSize=9, textColor=GRIS)
    bullet_style = ParagraphStyle("bullet", parent=normal,
        leftIndent=16, bulletIndent=0, spaceBefore=3)

    story = []
    fecha = datetime.now().strftime("%d de %B de %Y, %H:%M")

    # Encabezado
    story.append(Paragraph("OljaraVoz", titulo))
    story.append(Paragraph("Transcripción y Análisis de Reunión con IA", meta))
    story.append(Paragraph(f"Archivo: {filename} · Generado: {fecha}", meta))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=AZUL, spaceAfter=12))

    # Resumen
    story.append(Paragraph("Resumen ejecutivo", subtitulo))
    resumen = analisis.get("resumen", [])
    if isinstance(resumen, list):
        for punto in resumen:
            story.append(Paragraph(f"• {punto}", bullet_style))
    else:
        story.append(Paragraph(str(resumen), normal))

    # Compromisos
    story.append(Paragraph("Compromisos y tareas", subtitulo))
    compromisos = analisis.get("compromisos", [])
    if compromisos:
        for c in compromisos:
            story.append(Paragraph(f"☐ {c}", bullet_style))
    else:
        story.append(Paragraph("No se identificaron compromisos explícitos.", meta))

    # Acta
    story.append(Paragraph("Acta de reunión", subtitulo))
    acta = analisis.get("acta", "")
    for linea in acta.split("\n"):
        if linea.strip():
            story.append(Paragraph(linea.strip(), normal))
        else:
            story.append(Spacer(1, 0.2*cm))

    # Transcripción completa
    story.append(Paragraph("Transcripción completa", subtitulo))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRIS, spaceAfter=8))
    for linea in transcripcion.split(". "):
        if linea.strip():
            story.append(Paragraph(linea.strip() + ".", normal))

    # Firma
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRIS, spaceAfter=6))
    story.append(Paragraph("Generado por OljaraVoz · oljara.com", meta))

    doc.build(story)
    return output

# ── Diagnóstico ───────────────────────────────────────────────────────────────
@app.get("/health/ollama")
async def health_ollama():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # 1. Verifica que Ollama responda
            tags = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in tags.json().get("models", [])]

            # 2. Prueba rápida de generación
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "Di solo: OK", "stream": False},
            )
            resp.raise_for_status()
            reply = resp.json().get("response", "")

            return JSONResponse({
                "ok": True,
                "ollama_url": OLLAMA_URL,
                "model": OLLAMA_MODEL,
                "models_available": models,
                "test_reply": reply.strip(),
            })
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "ok": False,
            "ollama_url": OLLAMA_URL,
            "model": OLLAMA_MODEL,
            "error": str(e),
        })

# ── Estado y descarga ──────────────────────────────────────────────────────────
@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    return JSONResponse({
        "status":   job["status"],
        "step":     job["step"],
        "progress": job.get("progress", 0),
        "error":    job.get("error"),
    })

@app.get("/result/{job_id}")
async def result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    if job["status"] != "done":
        raise HTTPException(400, "El trabajo aún no está listo")
    return JSONResponse(job["result"])

@app.get("/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "PDF no disponible")
    pdf_path = Path(job["pdf"])
    if not pdf_path.exists():
        raise HTTPException(404, "Archivo PDF no encontrado")
    filename = Path(job["filename"]).stem + "_acta.pdf"
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=filename)
