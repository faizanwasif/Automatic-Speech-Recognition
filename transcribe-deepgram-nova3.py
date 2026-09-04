# pip install deepgram-sdk python-dotenv jiwer
# optional but recommended: pip install openai-whisper   (for EnglishTextNormalizer)
#
# Benchmarks Deepgram's nova-3 model (pre-recorded/batch STT, English-only)
# on the same fixed dataset manifest used for gemini-3.5-transcribe, and
# reports per-sample WER/RTFx plus corpus-level global WER/RTFx.
#
# This file is intentionally self-contained (see transcribe-3.5-google.py
# for the Gemini counterpart) -- the scoring/manifest/report logic below is
# the same shape by design so results_*.txt files from different providers
# stay directly comparable.
#
# IMPORTANT ON RTFx: processing_time is end-to-end REST request latency
# (network + queueing + server compute), not on-device compute time. Not
# comparable to a locally-run model's RTFx without saying so.

import os
import statistics
import time
import wave
from pathlib import Path

import jiwer
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
RESULTS_FILE = Path(__file__).parent / "results_deepgram.txt"

MODEL_UNDER_TEST = "nova-3"

# English-only, pre-recorded (not live-streaming) transcription.
LANGUAGE = "en"

# smart_format punctuates/capitalizes and formats numbers, dates, currency
# etc. Left on: normalize() below (Whisper's EnglishTextNormalizer) already
# strips formatting differences before scoring, so this doesn't distort WER,
# and it's how Deepgram is used in production.
SMART_FORMAT = True
PUNCTUATE = True

# Number of timed runs per sample -- see transcribe-3.5-google.py for the
# reasoning (pooled RTFx over a large corpus already averages out jitter).
RUNS_PER_SAMPLE = 1

# The SDK's default HTTP timeout is too short for large files (a 56MB / 331s
# recording hit WriteTimeout uploading the raw bytes in one shot). Long
# enough for the largest file in this corpus with margin.
REQUEST_TIMEOUT_S = 180.0

# Filenames accepted as ground truth, in priority order.
REFERENCE_NAMES = ("Hand written.txt", "reference.txt")
REFERENCE_GLOBS = ("*_reference.txt", "*.ref.txt")

# ----------------------------------------------------------------------------
# Dataset subset -- SAME manifest file as the Gemini script, so both
# providers are benchmarked on the exact same audio files.
# ----------------------------------------------------------------------------
MANIFEST_FILE = Path(__file__).parent / "dataset_manifest.txt"
MAX_FILES_PER_RUN = 24
TARGET_MINUTES = 60

MAX_RATE_LIMIT_RETRIES = 8
RATE_LIMIT_FALLBACK_WAIT_S = 20.0


# -----------------------------------------------------------------------------
# Normalization (identical policy to transcribe-3.5-google.py)
# -----------------------------------------------------------------------------
try:
    from whisper.normalizers import EnglishTextNormalizer

    _english_normalizer = EnglishTextNormalizer()

    def normalize(text: str) -> str:
        return _english_normalizer(text)

    NORMALIZER_NAME = "whisper EnglishTextNormalizer"
except ImportError:
    _fallback = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])

    def normalize(text: str) -> str:
        return _fallback(text)

    NORMALIZER_NAME = "basic lowercase/punctuation (install openai-whisper for the real one)"


# -----------------------------------------------------------------------------
# STEP 1: call the model
# -----------------------------------------------------------------------------

def get_audio_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _is_rate_limit_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "429" in str(exc)


def generate_transcript(client, model: str, audio_path: Path,
                        runs: int = 1) -> tuple[str, float, list[float]]:
    """
    Transcribes audio_path `runs` times via Deepgram's pre-recorded REST
    endpoint (listen.v1.media.transcribe_file) and returns
    (hypothesis_text_from_first_run, median_latency_s, all_latencies).

    Deepgram's pre-recorded API takes raw audio bytes in one request and
    returns the full transcript synchronously (no polling, unlike Gemini's
    file-upload flow) -- so the timed window is just the single API call.
    """
    audio_bytes = audio_path.read_bytes()

    latencies = []
    first_text = None
    for attempt_group in range(max(1, runs)):
        last_exc = None
        for retry in range(MAX_RATE_LIMIT_RETRIES):
            try:
                start = time.monotonic()
                response = client.listen.v1.media.transcribe_file(
                    request=audio_bytes,
                    model=model,
                    language=LANGUAGE,
                    smart_format=SMART_FORMAT,
                    punctuate=PUNCTUATE,
                    request_options={"timeout": REQUEST_TIMEOUT_S},
                )
                latencies.append(time.monotonic() - start)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_error(exc) or retry == MAX_RATE_LIMIT_RETRIES - 1:
                    raise
                print(f"    [rate limited] waiting {RATE_LIMIT_FALLBACK_WAIT_S:.0f}s "
                      f"(attempt {retry + 1}/{MAX_RATE_LIMIT_RETRIES})...")
                time.sleep(RATE_LIMIT_FALLBACK_WAIT_S)
        if last_exc:
            raise last_exc

        if first_text is None:
            channel = response.results.channels[0]
            first_text = (channel.alternatives[0].transcript or "").strip()

    return first_text, statistics.median(latencies), latencies


# -----------------------------------------------------------------------------
# STEP 2: score any (reference, hypothesis) pair -- model-agnostic
# -----------------------------------------------------------------------------

def score_transcript(reference: str, hypothesis: str, audio_duration_s: float,
                     processing_time_s: float) -> dict:
    """
    WER = (S + I + D) / N on normalized text. RTFx = audio_s / processing_s.
    Same policy as transcribe-3.5-google.py's score_transcript -- kept
    identical so numbers from both providers are directly comparable.
    """
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    if not ref_norm.strip():
        raise ValueError("empty reference transcript -- cannot compute WER")

    ref_words = ref_norm.split()
    n = len(ref_words)

    if not hyp_norm.strip():
        s, i, d, wer = 0, 0, n, 1.0
    else:
        measures = jiwer.process_words(ref_norm, hyp_norm)
        s, i, d = measures.substitutions, measures.insertions, measures.deletions
        wer = measures.wer

    return {
        "wer": wer,
        "substitutions": s,
        "insertions": i,
        "deletions": d,
        "ref_word_count": n,
        "audio_duration_s": audio_duration_s,
        "processing_time_s": processing_time_s,
        "rtfx": (audio_duration_s / processing_time_s) if processing_time_s > 0 else None,
    }


def aggregate(results: list[dict]) -> dict | None:
    scored = [r["stats"] for r in results if r.get("stats")]
    if not scored:
        return None
    total_n = sum(s["ref_word_count"] for s in scored)
    total_err = sum(s["substitutions"] + s["insertions"] + s["deletions"] for s in scored)
    total_audio = sum(s["audio_duration_s"] for s in scored)
    total_proc = sum(s["processing_time_s"] for s in scored)
    return {
        "micro_wer": total_err / total_n if total_n else None,
        "macro_wer": sum(s["wer"] for s in scored) / len(scored),
        "total_ref_words": total_n,
        "total_audio_s": total_audio,
        "total_proc_s": total_proc,
        "overall_rtfx": (total_audio / total_proc) if total_proc > 0 else None,
        "n_scored": len(scored),
        "n_total": len(results),
    }


# -----------------------------------------------------------------------------
# Data discovery + manifest (identical to transcribe-3.5-google.py, same file)
# -----------------------------------------------------------------------------

def find_reference(sample_dir: Path) -> Path | None:
    for name in REFERENCE_NAMES:
        candidate = sample_dir / name
        if candidate.exists():
            return candidate
    for pattern in REFERENCE_GLOBS:
        matches = sorted(sample_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_paired_corpus_pairs(audio_dir: Path, transcript_dir: Path, label_prefix: str):
    if not audio_dir.exists() or not transcript_dir.exists():
        return []

    transcript_by_stem = {}
    for txt_path in transcript_dir.glob("*.txt"):
        stem = txt_path.name.split(".")[0]
        transcript_by_stem[stem] = txt_path

    pairs = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        stem = wav_path.stem
        reference_path = transcript_by_stem.get(stem)
        if reference_path is None:
            print(f"[warn] {label_prefix}/{wav_path.name}: no matching transcript, skipping")
            continue
        pairs.append({
            "sample_id": f"{label_prefix}/{stem}",
            "audio_path": wav_path,
            "reference_path": reference_path,
        })
    return pairs


def find_sample_pairs():
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    pairs = []
    for sample_dir in sorted(DATA_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue

        paired_audio_dirs = sorted(sample_dir.glob("*audio*"))
        paired_transcript_dirs = sorted(sample_dir.glob("*transcript*"))
        if paired_audio_dirs and paired_transcript_dirs:
            for audio_dir in paired_audio_dirs:
                if not audio_dir.is_dir():
                    continue
                for transcript_dir in paired_transcript_dirs:
                    if transcript_dir.is_dir():
                        pairs.extend(find_paired_corpus_pairs(
                            audio_dir, transcript_dir, label_prefix=sample_dir.name
                        ))
                        break
            continue

        wav_files = sorted(sample_dir.glob("*.wav"))
        if not wav_files:
            print(f"[warn] {sample_dir.name}: no .wav found, skipping")
            continue
        reference_path = find_reference(sample_dir)
        if reference_path is None:
            print(f"[warn] {sample_dir.name}: no reference transcript found, skipping")
            continue
        pairs.append({
            "sample_id": sample_dir.name,
            "audio_path": wav_files[0],
            "reference_path": reference_path,
        })
    return pairs


def load_or_build_manifest(all_pairs: list[dict]) -> list[dict]:
    """
    Loads the SAME dataset_manifest.txt built by transcribe-3.5-google.py.
    If it doesn't exist yet, builds it the same way (longest files first, up
    to MAX_FILES_PER_RUN) so either script can be run first and both end up
    testing the identical file set.
    """
    by_id = {p["sample_id"]: p for p in all_pairs}

    if MANIFEST_FILE.exists():
        ids = [line.strip() for line in MANIFEST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing = [i for i in ids if i not in by_id]
        if missing:
            print(f"[warn] manifest references {len(missing)} sample(s) no longer found under {DATA_DIR}: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        selected = [by_id[i] for i in ids if i in by_id]
        print(f"[manifest] loaded {len(selected)} sample(s) from {MANIFEST_FILE.name} (fixed dataset, reused as-is)")
        return selected

    ranked = sorted(all_pairs, key=lambda p: -get_audio_duration_seconds(p["audio_path"]))
    selected = ranked[:MAX_FILES_PER_RUN]
    total_min = sum(get_audio_duration_seconds(p["audio_path"]) for p in selected) / 60

    MANIFEST_FILE.write_text(
        "\n".join(p["sample_id"] for p in selected) + "\n", encoding="utf-8"
    )
    print(f"[manifest] built new manifest: {len(selected)} file(s), {total_min:.1f} min total "
          f"(target was {TARGET_MINUTES} min) -> saved to {MANIFEST_FILE.name}")
    return selected


def main():
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise SystemExit("DEEPGRAM_API_KEY not set (add it to your .env file).")

    client = DeepgramClient(api_key=api_key)

    all_pairs = find_sample_pairs()
    if not all_pairs:
        raise SystemExit(f"No usable audio/reference pairs found under {DATA_DIR}")

    pairs = load_or_build_manifest(all_pairs)
    if not pairs:
        raise SystemExit(f"Manifest {MANIFEST_FILE.name} resolved to zero usable samples.")

    results = []
    for pair in pairs:
        sample_id = pair["sample_id"]
        audio_path = pair["audio_path"]
        duration_s = get_audio_duration_seconds(audio_path)
        reference = pair["reference_path"].read_text(encoding="utf-8").strip()

        print(f"[{sample_id}] [{MODEL_UNDER_TEST}] Transcribing: {audio_path.name} ({duration_s:.1f}s)")

        hypothesis, processing_time_s, latencies, stats, error = "", 0.0, [], None, None
        try:
            hypothesis, processing_time_s, latencies = generate_transcript(
                client, MODEL_UNDER_TEST, audio_path, runs=RUNS_PER_SAMPLE
            )
            stats = score_transcript(reference, hypothesis, duration_s, processing_time_s)
            spread = f" (runs: {', '.join(f'{x:.2f}s' for x in latencies)})" if len(latencies) > 1 else ""
            print(f"[{sample_id}] WER={stats['wer']*100:.1f}%  RTFx={stats['rtfx']:.2f}x{spread}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[{sample_id}] ERROR: {error}")

        results.append({
            "sample_id": sample_id,
            "audio_file": audio_path.name,
            "reference_file": pair["reference_path"].name,
            "duration_s": duration_s,
            "reference": reference,
            "hypothesis": hypothesis,
            "latencies": latencies,
            "error": error,
            "stats": stats,
        })

    write_results(results)
    print(f"\nDone. Results written to: {RESULTS_FILE}")


def write_results(results):
    totals = aggregate(results)

    lines = []
    lines.append("=" * 100)
    lines.append(f"ASR TEST RESULTS  (model: {MODEL_UNDER_TEST}, lang: {LANGUAGE}, "
                 f"smart_format: {SMART_FORMAT})")
    lines.append("=" * 100)
    lines.append("")

    lines.append("DATASET (fixed manifest -- same files used for every model/provider tested)")
    lines.append("-" * 100)
    lines.append(f"Manifest file: {MANIFEST_FILE.name}")
    lines.append(f"Files in this run: {len(results)}")
    if totals:
        lines.append(f"Total audio duration: {totals['total_audio_s']:.1f}s  ({totals['total_audio_s']/60:.2f} min)")
    lines.append("")

    if totals:
        lines.append("*" * 100)
        lines.append(f"  GLOBAL WER:   {totals['micro_wer']*100:.2f}%")
        lines.append(f"  GLOBAL RTFx:  {totals['overall_rtfx']:.2f}x")
        lines.append("*" * 100)
        lines.append("")
        lines.append(f"Normalizer: {NORMALIZER_NAME}")
        lines.append(f"Timing: end-to-end API request latency, {RUNS_PER_SAMPLE} run(s)/file.")
        lines.append("NOTE: RTFx here is API latency, not on-device compute. Do not compare directly")
        lines.append("      against locally-run models without saying so.")
        lines.append("")

        lines.append("CORPUS TOTALS (how the headline numbers are derived)")
        lines.append("-" * 100)
        lines.append(f"Global WER (micro-averaged, pooled S+I+D over pooled N): {totals['micro_wer']*100:.2f}%  "
                     f"over {totals['total_ref_words']} reference words")
        lines.append(f"  (naive per-file mean would read {totals['macro_wer']*100:.2f}% -- not used, it "
                     f"over-weights short clips)")
        lines.append(f"Global RTFx (pooled total audio / pooled total processing time): {totals['overall_rtfx']:.2f}x  "
                     f"({totals['total_audio_s']:.1f}s audio / {totals['total_proc_s']:.1f}s processing)")
        lines.append(f"Samples scored: {totals['n_scored']}/{totals['n_total']}")
        lines.append("")

    lines.append("PER-SAMPLE (WER lower is better | RTFx higher is faster)")
    lines.append("-" * 100)
    lines.append(f"{'SAMPLE':<10} {'WER':>8} {'S':>5} {'I':>5} {'D':>5} {'N':>6} {'DUR(s)':>8} {'PROC(s)':>9} {'RTFx':>8}")
    lines.append("-" * 100)
    for r in results:
        if not r["stats"]:
            lines.append(f"{r['sample_id']:<10} {'ERROR':>8} {'-':>5} {'-':>5} {'-':>5} {'-':>6} "
                         f"{r['duration_s']:>8.1f} {'-':>9} {'-':>8}")
            continue
        s = r["stats"]
        lines.append(
            f"{r['sample_id']:<10} {s['wer']*100:>7.1f}% {s['substitutions']:>5} {s['insertions']:>5} "
            f"{s['deletions']:>5} {s['ref_word_count']:>6} {s['audio_duration_s']:>8.1f} "
            f"{s['processing_time_s']:>8.2f}s {s['rtfx']:>7.2f}x"
        )
    lines.append("-" * 100)
    lines.append("")
    lines.append("")

    for r in results:
        lines.append("#" * 100)
        lines.append(f"### SAMPLE: {r['sample_id']}")
        lines.append(f"### AUDIO:      {r['audio_file']}  ({r['duration_s']:.1f}s)")
        lines.append(f"### REFERENCE:  {r['reference_file']}")
        if r["latencies"]:
            lines.append(f"### LATENCIES:  {', '.join(f'{x:.2f}s' for x in r['latencies'])}")
        lines.append("#" * 100)
        lines.append("")

        if r["error"]:
            lines.append(f"[ERROR]: {r['error']}")
            lines.append("")

        lines.append("--- REFERENCE (ground truth) ---")
        lines.append(r["reference"])
        lines.append("")

        lines.append(f"--- HYPOTHESIS ({MODEL_UNDER_TEST}) ---")
        if r["stats"]:
            s = r["stats"]
            lines.append(
                f"[METRICS] WER={s['wer']*100:.2f}%  "
                f"(S={s['substitutions']}, I={s['insertions']}, D={s['deletions']}, N={s['ref_word_count']})  |  "
                f"Latency={s['processing_time_s']:.2f}s  |  RTFx={s['rtfx']:.2f}x"
            )
            lines.append("")
        lines.append(r["hypothesis"] if r["hypothesis"] else "(no output)")
        lines.append("")
        lines.append("")

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
