FROM python:3.11-slim

# ffmpeg is required to convert arbitrary uploaded audio into the mono/16kHz
# PCM WAV format the hosted Parakeet model requires.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "300", "--workers", "2", "app:app"]
