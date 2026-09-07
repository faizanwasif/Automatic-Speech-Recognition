# pip install qwen-asr torch python-dotenv jiwer
# optional but recommended: pip install openai-whisper   (for EnglishTextNormalizer)
#
# Benchmarks Qwen3-ASR-1.7B (local GPU inference, multilingual, Apache-2.0)
# on the same fixed dataset manifest used for the other providers, and
# reports per-sample WER/RTFx plus corpus-level global WER/RTFx.
#
# UNLIKE the other transcribe-*.py scripts in this repo, this one runs
# entirely locally -- no API key, no network call. Requires a CUDA GPU
# (confirmed available: RTX 3080 Ti Mobile, 16GB VRAM). Model loading
# happens ONCE, outside the timed per-file loop -- RTFx measures pure
# inference time, not one-time model-load overhead, which is the correct
# comparison against the other providers' network-latency-based RTFx (both
# measure "time to get a transcript once the system is ready to work").
#
# This file is intentionally self-contained (see transcribe-3.5-google.py,
# transcribe-deepgram-nova3.py, transcribe-assemblyai-u35pro.py for the
# hosted-API counterparts) -- the scoring/manifest/report logic below is
# the same shape by design so results_*.txt files stay directly comparable.

import os
import time
import wave
from pathlib import Path

import jiwer
import torch
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
RESULTS_FILE = Path(__file__).parent / "results_qwen3asr.txt"

MODEL_UNDER_TEST = "Qwen/Qwen3-ASR-1.7B"

# None = auto-detect language (the model's native behavior). All our
# manifest audio is English, so this is equivalent to forcing "en" for our
# purposes, but leaving it None exercises the model's real auto-detect path
# rather than assuming -- catches a mis-detection as a genuine WER hit.
LANGUAGE = None

RUNS_PER_SAMPLE = 1

REFERENCE_NAMES = ("Hand written.txt", "reference.txt")
REFERENCE_GLOBS = ("*_reference.txt", "*.ref.txt")

# Same fixed manifest as every other provider script in this repo.
MANIFEST_FILE = Path(__file__).parent / "dataset_manifest.txt"
MAX_FILES_PER_RUN = 24
TARGET_MINUTES = 60


# -----------------------------------------------------------------------------
# Normalization (identical policy to the other provider scripts)
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
# STEP 1: load the model once, then call it per file
# -----------------------------------------------------------------------------

def get_audio_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def load_model():
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. Qwen3-ASR-1.7B on CPU would be extremely slow; "
            "this script assumes local GPU inference. Check `nvidia-smi`."
        )

    from qwen_asr import Qwen3ASRModel

    print(f"Loading {MODEL_UNDER_TEST} onto GPU (one-time cost, excluded from RTFx)...")
    load_start = time.monotonic()
    model = Qwen3ASRModel.from_pretrained(
        MODEL_UNDER_TEST,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        # 256 (an earlier value here) silently truncates anything past ~1-2
        # sentences -- confirmed via a first run where our longest sample
        # (331s / 781 reference words) came back with 606 deleted words,
        # cut off mid-sentence at roughly the token budget. Speech runs
        # ~2-3 tokens/word at ~150 wpm, so our longest manifest file
        # (~5.5 min) needs on the order of ~1500-2500 tokens; this is set
        # with real headroom above that.
        max_new_tokens=4096,
    )
    print(f"Model loaded in {time.monotonic() - load_start:.1f}s.")
    return model


def generate_transcript(model, audio_path: Path, runs: int = 1) -> tuple[str, float, list[float]]:
    """
    Transcribes audio_path `runs` times and returns
    (hypothesis_text_from_first_run, median_inference_s, all_inference_times).

    Only the model.transcribe() call itself is timed -- this is pure
    on-device inference time, analogous to what the hosted-API scripts
    measure as "processing time" (their equivalent of "how long did it take
    once the system started working on this file").
    """
    latencies = []
    first_text = None
    for _ in range(max(1, runs)):
        start = time.monotonic()
        results = model.transcribe(audio=str(audio_path), language=LANGUAGE)
        latencies.append(time.monotonic() - start)
        if first_text is None:
            # qwen_asr's transcribe() returns a list of results (one per
            # input); we pass one file, so take the first.
            result = results[0] if isinstance(results, list) else results
            first_text = (getattr(result, "text", None) or str(result)).strip()

    import statistics
    return first_text, statistics.median(latencies), latencies


# -----------------------------------------------------------------------------
# STEP 2: score any (reference, hypothesis) pair -- model-agnostic
# -----------------------------------------------------------------------------

def score_transcript(reference: str, hypothesis: str, audio_duration_s: float,
                     processing_time_s: float) -> dict:
    """
    WER = (S + I + D) / N on normalized text. RTFx = audio_s / processing_s.
    Same policy as the other provider scripts' score_transcript.
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
# Data discovery + manifest (identical to the other provider scripts, same file)
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
    Loads the SAME dataset_manifest.txt built by the other provider scripts.
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
    model = load_model()

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
                model, audio_path, runs=RUNS_PER_SAMPLE
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
    lines.append(f"ASR TEST RESULTS  (model: {MODEL_UNDER_TEST}, local GPU inference, lang: auto-detect)")
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
        lines.append(f"Timing: pure on-device inference time (model.transcribe() call only), "
                     f"{RUNS_PER_SAMPLE} run(s)/file. One-time model-load cost excluded.")
        lines.append("NOTE: This is LOCAL GPU compute time, not hosted-API network latency --")
        lines.append("      the opposite measurement basis from the other results_*.txt files in this")
        lines.append("      repo. A higher RTFx here reflects real inference speed on this GPU")
        lines.append("      (RTX 3080 Ti Mobile, 16GB), not network/queueing conditions.")
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
