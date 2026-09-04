# pip install google-genai python-dotenv jiwer
# optional but recommended: pip install openai-whisper   (for EnglishTextNormalizer)
#
# Benchmarks one ASR system on a folder of (audio, reference) pairs and reports
# per-sample WER/RTFx plus a corpus-level micro-averaged WER.
#
#   generate_transcript()  -- Gemini-specific: one audio file -> (text, latency_s)
#   score_transcript()     -- model-agnostic: (reference, hypothesis) -> metrics
#
# IMPORTANT ON RTFx: for a hosted API, processing_time is request latency
# (queueing + server compute + network), not compute time. It is NOT
# comparable to the RTFx of a locally-run model on your own GPU. Report it
# in a separate column of your doc, labelled as end-to-end API latency.

import os
import re
import statistics
import time
import wave
from pathlib import Path

import jiwer
from dotenv import load_dotenv
from google import genai

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
RESULTS_FILE = Path(__file__).parent / "results_gemini.txt"

MODEL_UNDER_TEST = "gemini-3.5-transcribe"

# Scope is English-only; giving the language hint avoids autodetect drift and
# is what you would do in production anyway.
LANGUAGE_CODES = ["en-US"]

# "verbatim" (default) vs "smart". Keep verbatim: smart mode removes
# disfluencies and reformats, which is a WER catastrophe against verbatim
# ground truth (AMI references keep "um", "uh", false starts).
TRANSCRIPTION_MODE = {"type": "verbatim"}

# Number of timed runs per sample. Latency on a shared API is noisy, but with
# a large corpus (250+ files) a single run per file is enough -- the global
# RTFx is pooled over the whole set, which already averages out per-call
# jitter. Bump this back up only for small corpora where per-file noise
# wouldn't otherwise wash out.
RUNS_PER_SAMPLE = 1

# Filenames accepted as ground truth, in priority order. Deliberately NOT a
# blind *.txt glob -- that can pick up a previous run's output.
REFERENCE_NAMES = ("Hand written.txt", "reference.txt")
REFERENCE_GLOBS = ("*_reference.txt", "*.ref.txt")

FILE_PROCESSING_TIMEOUT_S = 300

# ----------------------------------------------------------------------------
# Dataset subset (quota-constrained runs)
# ----------------------------------------------------------------------------
# The full corpus (259 files, ~71 min) exceeds gemini-3.5-transcribe's free
# tier of 25 requests/day. MANIFEST_FILE freezes a fixed subset -- selected
# once as the longest-duration files available, to pack the most audio
# minutes into a limited request budget -- so every model/provider tested
# later runs against the exact same files for a fair comparison. If the
# manifest file doesn't exist yet, one is built and saved on first run;
# after that it is loaded as-is and MAX_FILES_PER_RUN/TARGET_MINUTES below
# are ignored, so the dataset stays fixed even as request budgets change.
MANIFEST_FILE = Path(__file__).parent / "dataset_manifest.txt"
MAX_FILES_PER_RUN = 24       # today's remaining free-tier request budget
TARGET_MINUTES = 60          # aspirational; free tier can't reach this in a day

# This API key is on the free tier, which caps gemini-3.5-transcribe at a few
# requests/minute (429 RateLimitError otherwise). On a 429, the server tells
# us how long to wait ("...Please retry in 12.2s."); we sleep for exactly
# that plus a small margin, rather than a fixed/exponential guess, and try
# again. That wait time is NOT counted in RTFx -- it is quota throttling,
# not model inference time -- same treatment as upload time.
MAX_RATE_LIMIT_RETRIES = 20
RATE_LIMIT_FALLBACK_WAIT_S = 20.0  # used only if the server doesn't give a hint
_RETRY_SECONDS_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
# Lowercasing + punctuation stripping is NOT enough here. Gemini applies
# inverse text normalization ("twenty six million dollars" -> "$26M",
# "two o'clock" -> "2:00 PM"), while AMI references are spelled out. Without a
# real normalizer those become substitutions and you end up measuring
# formatting conventions, not recognition. Whisper's EnglishTextNormalizer
# handles numbers, currency, contractions and British/American spelling; it is
# what the HF Open ASR Leaderboard uses, so your numbers stay comparable.

try:
    from whisper.normalizers import EnglishTextNormalizer

    _english_normalizer = EnglishTextNormalizer()

    def normalize(text: str) -> str:
        return _english_normalizer(text)

    NORMALIZER_NAME = "whisper EnglishTextNormalizer"
except ImportError:  # fallback: weaker, but at least consistent
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


def _rate_limit_wait_s(exc: Exception) -> float:
    """Extracts the server-suggested retry delay from a 429 error message
    (e.g. "...Please retry in 12.2s."), falling back to a fixed wait if the
    message doesn't include one."""
    match = _RETRY_SECONDS_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 0.5  # small margin
    return RATE_LIMIT_FALLBACK_WAIT_S


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    True if exc represents an HTTP 429. Deliberately NOT type-checked against
    a specific exception class: client.models.* raises google.genai.errors
    .APIError (with a .code attribute), while client.interactions.* raises a
    completely separate internal hierarchy (google.genai._gaos...RateLimit
    Error, with .status_code instead) for the same HTTP status. Checking both
    common attribute names, plus the message text as a last resort, means
    this keeps working across genai-SDK internals without importing a
    private module path.
    """
    if getattr(exc, "code", None) == 429:
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    return "429" in str(exc) and "retry in" in str(exc).lower()


def _call_with_rate_limit_retry(fn, *, on_wait=None) -> tuple:
    """
    Calls fn() and retries on HTTP 429 (rate limit), sleeping for the
    duration the server tells us to wait. Any other error propagates
    immediately. Raises the last error if MAX_RATE_LIMIT_RETRIES is exceeded.

    `on_wait(wait_s, attempt)` is called right before each sleep, so callers
    can log progress -- otherwise a run against a free-tier key looks stuck.

    Returns (result, total_wait_s) so callers can exclude the throttling
    wait from their own timing (e.g. RTFx latency should not include time
    spent sleeping off a quota limit).
    """
    total_wait_s = 0.0
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return fn(), total_wait_s
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            wait_s = _rate_limit_wait_s(exc)
            if on_wait:
                on_wait(wait_s, attempt)
            time.sleep(wait_s)
            total_wait_s += wait_s
    raise RuntimeError("unreachable")  # loop always returns or raises


def _wait_for_file_active(client, uploaded_file):
    deadline = time.monotonic() + FILE_PROCESSING_TIMEOUT_S
    while str(uploaded_file.state).endswith("PROCESSING"):
        if time.monotonic() > deadline:
            raise TimeoutError(f"File {uploaded_file.name} stuck in PROCESSING")
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    if str(uploaded_file.state).endswith("FAILED"):
        raise RuntimeError(f"File upload failed: {uploaded_file.name}")
    return uploaded_file


def generate_transcript(client, model: str, audio_path: Path,
                        runs: int = 1) -> tuple[str, float, list[float]]:
    """
    Transcribes audio_path `runs` times and returns
    (hypothesis_text_from_first_run, median_latency_s, all_latencies).

    gemini-3.5-transcribe is served by the Interactions API, not
    generate_content. The transcript comes back in interaction.output_text.
    Upload time is excluded from the timing (it is network, not inference);
    everything else in the request is included and cannot be separated out.

    On a free-tier key, requests are commonly rate-limited (HTTP 429). Those
    are retried automatically (see _call_with_rate_limit_retry) and the wait
    is excluded from the measured latency -- it's quota throttling, not
    inference time, so it would distort RTFx if counted.
    """
    def log_wait(wait_s, attempt):
        print(f"    [rate limited] waiting {wait_s:.1f}s (attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})...")

    uploaded_file, _ = _call_with_rate_limit_retry(
        lambda: client.files.upload(file=str(audio_path)), on_wait=log_wait
    )
    try:
        uploaded_file = _wait_for_file_active(client, uploaded_file)

        latencies = []
        first_text = None
        for _ in range(max(1, runs)):
            start = time.monotonic()
            interaction, rate_limit_wait_s = _call_with_rate_limit_retry(
                lambda: client.interactions.create(
                    model=model,
                    input=[{
                        "type": "audio",
                        "uri": uploaded_file.uri,
                        "mime_type": uploaded_file.mime_type,
                    }],
                    generation_config={
                        "transcription_config": {
                            "language_codes": LANGUAGE_CODES,
                            "mode": TRANSCRIPTION_MODE,
                        }
                    },
                ),
                on_wait=log_wait,
            )
            elapsed_s = (time.monotonic() - start) - rate_limit_wait_s
            latencies.append(max(elapsed_s, 0.0))
            if first_text is None:
                first_text = (interaction.output_text or "").strip()

        return first_text, statistics.median(latencies), latencies
    finally:
        # Always clean up, even if the request blew up mid-way.
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# STEP 2: score any (reference, hypothesis) pair -- model-agnostic
# -----------------------------------------------------------------------------

def score_transcript(reference: str, hypothesis: str, audio_duration_s: float,
                     processing_time_s: float) -> dict:
    """
    WER = (S + I + D) / N on normalized text. RTFx = audio_s / processing_s.

    Raises ValueError on an empty reference -- WER is undefined with N=0, and
    silently scoring it as 0% or 100% would poison the corpus average.
    Empty hypotheses are valid (counted as N deletions, WER = 100%).

    To score another system:
        hypothesis, elapsed = my_asr(audio_path)
        stats = score_transcript(ref_text, hypothesis, duration_s, elapsed)
    """
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    if not ref_norm.strip():
        raise ValueError("empty reference transcript -- cannot compute WER")

    ref_words = ref_norm.split()
    n = len(ref_words)

    if not hyp_norm.strip():
        # jiwer rejects empty hypotheses; handle explicitly rather than
        # passing " " (which the transforms strip back to empty anyway).
        measures = None
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
    """
    Corpus-level micro-average: sum errors / sum reference words. This is the
    number to put in the doc -- averaging per-file WERs over-weights short
    files and is not what published leaderboards report.
    """
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
# Data discovery + orchestration
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
    """
    Handles a "paired corpus" layout: one flat folder of .wav files and one
    flat folder of transcript .txt files, matched by basename (audio.wav <->
    audio.<anything>.txt, e.g. clip_00000.wav <-> clip_00000.flac.txt).
    Used for data/Audio-Team-English/{peoples_speech_audio_wav,peoples_speech_transcriptions}.
    """
    if not audio_dir.exists() or not transcript_dir.exists():
        return []

    # Map "clip_00000" -> transcript path, regardless of the extra suffix
    # (.flac.txt) hung off the stem.
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

        # Paired-corpus layout: this dir has its own audio_wav/transcriptions
        # subfolders (e.g. Audio-Team-English/peoples_speech_*).
        paired_audio_dirs = sorted(sample_dir.glob("*audio*"))
        paired_transcript_dirs = sorted(sample_dir.glob("*transcript*"))
        if paired_audio_dirs and paired_transcript_dirs:
            for audio_dir in paired_audio_dirs:
                if not audio_dir.is_dir():
                    continue
                # match e.g. "..._audio_wav" with "..._transcriptions" by
                # taking whichever transcript dir exists alongside it
                for transcript_dir in paired_transcript_dirs:
                    if transcript_dir.is_dir():
                        pairs.extend(find_paired_corpus_pairs(
                            audio_dir, transcript_dir, label_prefix=sample_dir.name
                        ))
                        break
            continue

        # Simple layout: one .wav + one reference .txt directly in this dir.
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
    Returns the fixed subset of pairs to run this session, keyed by
    sample_id so it's stable across re-runs and across which model is under
    test. First call writes MANIFEST_FILE (one sample_id per line); every
    call after that reads the same file back, ignoring MAX_FILES_PER_RUN --
    this is what keeps the dataset identical across different models/
    providers tested on different days/quota budgets.
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

    # No manifest yet: build one. Rank by duration (longest first) to pack
    # the most audio minutes into a limited per-day request budget, then
    # take the top MAX_FILES_PER_RUN and freeze that as the manifest.
    ranked = sorted(all_pairs, key=lambda p: -get_audio_duration_seconds(p["audio_path"]))
    selected = ranked[:MAX_FILES_PER_RUN]
    total_min = sum(get_audio_duration_seconds(p["audio_path"]) for p in selected) / 60

    MANIFEST_FILE.write_text(
        "\n".join(p["sample_id"] for p in selected) + "\n", encoding="utf-8"
    )
    print(f"[manifest] built new manifest: {len(selected)} file(s), {total_min:.1f} min total "
          f"(target was {TARGET_MINUTES} min; free-tier request budget capped it) -> saved to {MANIFEST_FILE.name}")
    return selected


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set (check your .env file).")

    client = genai.Client(api_key=api_key)

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
    lines.append(f"ASR TEST RESULTS  (model: {MODEL_UNDER_TEST}, mode: {TRANSCRIPTION_MODE}, lang: {LANGUAGE_CODES})")
    lines.append("=" * 100)
    lines.append("")

    lines.append("DATASET (fixed manifest -- same files used for every model/provider tested)")
    lines.append("-" * 100)
    lines.append(f"Manifest file: {MANIFEST_FILE.name}")
    lines.append(f"Files in this run: {len(results)}")
    if totals:
        lines.append(f"Total audio duration: {totals['total_audio_s']:.1f}s  ({totals['total_audio_s']/60:.2f} min)")
    lines.append(f"Target was {TARGET_MINUTES} min; gemini-3.5-transcribe's free-tier daily request quota "
                 f"(~{MAX_FILES_PER_RUN} requests/day used to build this manifest) capped it below that --")
    lines.append("see per-sample table below for the exact files and durations included.")
    lines.append("")

    if totals:
        # ---- THE HEADLINE NUMBERS ----
        # These two are the answer to "how accurate and how fast is this
        # model on our dataset": one WER, one RTFx, pooled over every file
        # (not averaged per-file -- see CORPUS TOTALS below for why).
        lines.append("*" * 100)
        lines.append(f"  GLOBAL WER:   {totals['micro_wer']*100:.2f}%")
        lines.append(f"  GLOBAL RTFx:  {totals['overall_rtfx']:.2f}x")
        lines.append("*" * 100)
        lines.append("")
        lines.append(f"Normalizer: {NORMALIZER_NAME}")
        lines.append(f"Timing: end-to-end API request latency, {RUNS_PER_SAMPLE} run(s)/file, upload excluded.")
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


# -----------------------------------------------------------------------------
# Template: scoring a different ASR system with the same metrics
# -----------------------------------------------------------------------------

def score_only_example():
    """
    Reuse for any other system. Time only the transcribe call, and for local
    models exclude model load time and discard a warm-up run first.
    """
    reference_text = Path("data/s1/Hand written.txt").read_text(encoding="utf-8")
    audio_path = Path("data/s1/sample.wav")
    audio_duration_s = get_audio_duration_seconds(audio_path)

    start = time.monotonic()
    hypothesis_text = "...output from some other ASR system..."
    processing_time_s = time.monotonic() - start

    print(score_transcript(reference_text, hypothesis_text, audio_duration_s, processing_time_s))


if __name__ == "__main__":
    main()