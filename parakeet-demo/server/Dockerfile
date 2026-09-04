FROM python:3.11-slim

# ffmpeg is required to convert arbitrary uploaded audio into the mono/16kHz
# PCM WAV format the hosted Parakeet model requires.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

# Render injects its own $PORT at runtime and routes to whatever port the
# container actually listens on -- binding to a hardcoded port instead of
# reading $PORT is what caused "no-server" routing failures. Shell form
# (not exec-form CMD) is required here so $PORT gets expanded.
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 300 --workers 2 app:app
