"""
Parakeet demo proxy.

The browser can't speak gRPC and shouldn't hold the NVIDIA API key, so this
tiny Flask app sits in between:

  browser --(HTTP multipart upload)--> this server --(gRPC)--> NVIDIA Parakeet NIM

It also runs ffmpeg to convert whatever audio format the browser sends into
the mono/16kHz PCM WAV that Parakeet requires (confirmed via a live test
call -- the hosted model rejects anything else with "input format doesn't
match with header format").

Env vars required (set in Render's dashboard, not committed to git):
    NVIDIA_API_KEY   -- your build.nvidia.com API key (starts with "nvapi-")
"""

import os
import subprocess
import tempfile
import wave
from pathlib import Path

import riva.client
from flask import Flask, jsonify, request
from flask_cors import CORS

FUNCTION_ID = "d3fe9151-442b-4204-a70d-5fcc597fd610"  # ai-parakeet-tdt-0_6b-v2
GRPC_URI = "grpc.nvcf.nvidia.com:443"

# Parakeet's documented bounds for this hosted endpoint.
MIN_AUDIO_SECONDS = 3
MAX_AUDIO_SECONDS = 5 * 60

app = Flask(__name__)
CORS(app)  # demo only -- allows the static frontend (any origin) to call this API


def get_asr_service() -> riva.client.ASRService:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY environment variable is not set on the server.")
    auth = riva.client.Auth(
        uri=GRPC_URI,
        use_ssl=True,
        metadata_args=[
            ["function-id", FUNCTION_ID],
            ["authorization", f"Bearer {api_key}"],
        ],
    )
    return riva.client.ASRService(auth)


def convert_to_parakeet_wav(input_path: Path, output_path: Path) -> None:
    """Mono, 16kHz, 16-bit PCM WAV -- the only format the hosted model accepts."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio conversion failed: {result.stderr[-2000:]}")


def get_wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def transcribe(wav_path: Path) -> dict:
    asr_service = get_asr_service()
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    audio_bytes = wav_path.read_bytes()
    response = asr_service.offline_recognize(audio_bytes, config)

    # The API returns one result per internal audio segment for longer
    # files; concatenate them into a single transcript, and keep per-segment
    # detail for the UI in case it wants to show timing.
    segments = []
    full_text_parts = []
    for result in response.results:
        if not result.alternatives:
            continue
        alt = result.alternatives[0]
        full_text_parts.append(alt.transcript)
        segments.append({
            "text": alt.transcript,
            "confidence": alt.confidence,
            "start_ms": alt.words[0].start_time if alt.words else None,
            "end_ms": alt.words[-1].end_time if alt.words else None,
        })

    return {
        "text": "".join(full_text_parts).strip(),
        "segments": segments,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/transcribe", methods=["POST"])
def transcribe_endpoint():
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file in request."}), 400

    uploaded = request.files["audio"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / "input_raw"
        converted_path = tmp_dir_path / "converted.wav"

        uploaded.save(input_path)

        try:
            convert_to_parakeet_wav(input_path, converted_path)
        except RuntimeError as exc:
            return jsonify({"error": f"Could not process this audio file: {exc}"}), 400

        duration_s = get_wav_duration_seconds(converted_path)
        if duration_s < MIN_AUDIO_SECONDS:
            return jsonify({
                "error": f"Audio is {duration_s:.1f}s, but the model requires at least "
                         f"{MIN_AUDIO_SECONDS}s."
            }), 400
        if duration_s > MAX_AUDIO_SECONDS:
            return jsonify({
                "error": f"Audio is {duration_s:.1f}s, but the model accepts at most "
                         f"{MAX_AUDIO_SECONDS}s ({MAX_AUDIO_SECONDS // 60} minutes)."
            }), 400

        try:
            result = transcribe(converted_path)
        except Exception as exc:  # surface a clean error to the demo UI
            return jsonify({"error": f"Transcription failed: {exc}"}), 502

        result["duration_seconds"] = round(duration_s, 2)
        return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
