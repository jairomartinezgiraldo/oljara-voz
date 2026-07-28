FROM python:3.12-slim

# ffmpeg es requerido por Whisper para procesar audio
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descarga el modelo Whisper al construir la imagen (evita espera en primer uso)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"

COPY . .

RUN mkdir -p /app/uploads /app/results

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
