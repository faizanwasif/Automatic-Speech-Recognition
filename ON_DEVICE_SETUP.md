# On-Device ASR Benchmarking: Setup Guide (Qwen3-ASR-1.7B)

This documents how `transcribe-qwen3-asr.py` was set up and debugged, so the
same environment (or a similar one) can be reproduced without repeating the
same failures. Unlike the other `transcribe-*.py` scripts in this repo
(Gemini, Deepgram, AssemblyAI — all hosted APIs), this one runs the model
entirely locally on a GPU. No API key, no network call for inference.

## Reference environment

This was built and verified on:

| Component | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop, 16 GB VRAM |
| Driver | 595.84 |
| CUDA (driver-reported) | 13.2 |
| PyTorch | 2.13.0+cu130 |
| cuDNN (final, working) | 9.25.1.1 |
| Pillow | 12.3.0 |
| `qwen-asr` | 0.0.6 |

Your versions don't need to match exactly, but the *shape* of the setup
below (recent driver + recent PyTorch + explicit cuDNN pin) is what mattered.

## 1. Confirm you have a usable GPU first

Don't install anything until this passes — every failure mode below is
easier to debug once you know the GPU itself is fine:

```bash
nvidia-smi                                   # confirms driver + GPU are visible to the OS
python3 -c "import torch; print(torch.cuda.is_available())"   # confirms PyTorch sees it
```

If `torch.cuda.is_available()` is `False`, stop here — that's a PyTorch/CUDA
install problem to fix before touching the ASR model itself.

**VRAM sizing**: Qwen3-ASR-1.7B in bfloat16 needs roughly 3.4 GB for weights
alone, plus inference overhead. 16 GB has comfortable headroom; an 8 GB card
should still work but leaves less margin for longer audio batches.

## 2. Install the model package

```bash
pip install --user qwen-asr torch python-dotenv jiwer
pip install --user openai-whisper   # optional, for accurate WER scoring (see repo README)
```

`qwen-asr` pulls in `transformers`, `torch` (if not already present), and
several other dependencies. This step alone took several minutes — normal.

## 3. Fix the Pillow conflict (if you hit `AttributeError: module 'PIL.Image' has no attribute 'Resampling'`)

**What happened**: this machine had an old system-wide Pillow (9.0.1, from
`/usr/lib/python3/dist-packages/`, likely pulled in by unrelated tools like
`gradio`/`deepface`) that predates the `Resampling` API `transformers`
requires. Importing `qwen_asr` transitively imports `transformers`, which
imports `PIL.Image.Resampling`, and crashes on old Pillow.

**Why the fix is scoped the way it is**: don't touch the system Pillow —
other installed tools depend on that exact old version, and upgrading it
system-wide could break them. Upgrade Pillow in *user* site-packages
instead, which Python resolves ahead of the system copy for `pip install
--user` installs:

```bash
pip install --user --upgrade "Pillow>=10"
```

Verify the fix:
```bash
python3 -c "import PIL; print(PIL.__version__, PIL.__file__)"
# should print a version >= 10, from a path under ~/.local/lib/...
```

If your system doesn't have this old-Pillow problem, you'll simply never
hit this error — skip straight to step 4.

## 4. Fix the cuDNN sublibrary mismatch (if you hit `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`)

**Symptom**: model loads fine, but the first real inference call crashes
deep in a `conv2d` operation (inside the model's audio encoder) with:

```
RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed...
cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

**How this was diagnosed** (worth doing in this order rather than guessing):

1. Isolated the failure to a minimal repro — a bare `torch.nn.functional.conv2d`
   call on a random tensor, with no model involved:
   ```python
   import torch, torch.nn.functional as F
   x = torch.randn(1, 3, 8, 8, device="cuda", dtype=torch.bfloat16)
   w = torch.randn(4, 3, 3, 3, device="cuda", dtype=torch.bfloat16)
   F.conv2d(x, w)   # crashed with the same error
   ```
   This confirmed the bug was in the PyTorch/cuDNN/driver stack, not in
   `qwen_asr` or the model itself.

2. Confirmed cuDNN specifically was the cause by disabling it and re-running
   the same repro:
   ```python
   torch.backends.cudnn.enabled = False
   F.conv2d(x, w)   # succeeded (fell back to a non-cuDNN kernel)
   ```

3. Checked what cuDNN version was actually loaded:
   ```python
   import torch
   print(torch.backends.cudnn.version())   # printed 92000, i.e. cuDNN 9.2.0
   ```
   Cross-referenced against the driver (595.84 / CUDA 13.2, a very recent
   combination) — cuDNN 9.2.0 predates that driver/CUDA pairing and isn't
   validated against it. This is a real compatibility gap, not a
   misconfiguration.

**The fix** — upgrade cuDNN to the latest available build:
```bash
pip install --user --upgrade nvidia-cudnn-cu13
```
This installed 9.25.1.1, which resolved the crash. Re-running the same
minimal `conv2d` repro (with `cudnn.enabled` left at its default `True`)
succeeded afterward.

You'll likely see a pip warning like:
```
torch 2.13.0 requires nvidia-cudnn-cu13==9.20.0.48, but you have
nvidia-cudnn-cu13 9.25.1.1 which is incompatible.
```
This is expected and safe to ignore here — it only means PyTorch's declared
pin is stricter than what's actually required; the newer cuDNN is backward
compatible and the fix was verified to work, not just installed and assumed
fine.

**If the upgrade doesn't fix it for your combination**: fall back to
disabling cuDNN for this script specifically (add `torch.backends.cudnn.enabled
= False` near the top of `transcribe-qwen3-asr.py`). Confirmed working, but
slower per-file inference since it loses cuDNN's optimized conv kernels —
acceptable if you just need correct results over raw speed.

## 5. Sanity-check the full pipeline before running the full benchmark

Model weights download from Hugging Face on first use (~3.4 GB) and get
cached under `~/.cache/huggingface/hub/`. **This can be slow** — on this
setup, the initial download took about 56 minutes over a modest connection.
Subsequent loads from cache take ~10 seconds.

Run a single-file smoke test first so a slow network or a broken environment
doesn't eat an hour before you find out:

```python
import torch, time
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=4096,   # see the note below -- do not use the small default
)

start = time.monotonic()
results = model.transcribe(audio="path/to/some/short/clip.wav", language=None)
print(f"{time.monotonic()-start:.2f}s:", results[0].text)
```

A working setup should transcribe a 15-second clip in a few seconds once the
model is loaded and cached.

## 6. The `max_new_tokens` trap — silent truncation on longer audio

**This is the most important gotcha, and it produces no error at all** — it
just silently returns an incomplete transcript, which then looks like bad
model quality unless you know to look for it.

The model card's example code uses `max_new_tokens=256`. That's enough for
a couple of sentences. On anything longer, generation simply stops there —
no exception, no warning, just a transcript that quietly stops mid-sentence.

**How this was caught**: after a first full benchmark run, the two longest
audio files in the test set scored dramatically worse (WER 83% and 21%) than
every short clip (WER 2–14%). That pattern — long files bad, short files
fine, same model — is the signature of a length-related truncation bug, not
a genuine accuracy problem. Inspecting the actual hypothesis text confirmed
it: the transcript for the 331-second file stopped after roughly one-third
of the audio, with the scoring breakdown showing 606 of 781 reference words
counted as deletions (i.e., simply missing from the output).

**The fix**: scale `max_new_tokens` to your actual audio lengths. Speech
transcribes at roughly 2–3 tokens per word, and spoken English runs around
150 words/minute, so:

```
max_new_tokens ≈ (max_audio_minutes × 150 words/min) × 3 tokens/word,
                  with real headroom on top
```

For a 5-minute-max dataset, `max_new_tokens=4096` gives comfortable margin.
After this fix, the same two long files improved to WER 35% and 11%
respectively — the 331-second file still trails the short clips (a real,
smaller accuracy gap on long-form audio, not a bug), but is no longer
catastrophically truncated.

**Takeaway for replication**: whatever `max_new_tokens` value you pick,
verify it against your longest actual audio file, not just the model card's
short-clip example. If you see suspiciously high WER concentrated on your
longest files specifically, check hypothesis length against reference length
before assuming it's a model-quality problem.

## 7. How WER and RTFx are calculated, and how the report is built

This section documents the scoring/reporting pipeline in
`transcribe-qwen3-asr.py` (`score_transcript()`, `aggregate()`,
`write_results()`) — it's identical in shape to the same functions in the
other three `transcribe-*.py` scripts in this repo, by design, so results
are comparable across providers and not just internally consistent.

### WER (Word Error Rate)

Both reference and hypothesis text are run through `normalize()` first
(Whisper's `EnglishTextNormalizer`) — lowercasing, contraction expansion,
spelled-out numbers converted to digits, punctuation stripped — so
formatting differences aren't counted as transcription errors. Then:

```python
measures = jiwer.process_words(ref_norm, hyp_norm)
s, i, d = measures.substitutions, measures.insertions, measures.deletions
wer = measures.wer
```

`jiwer` performs word-level edit-distance alignment between the two texts.
`wer = (S + I + D) / N`, where `N` is the reference word count — the
standard WER formula.

### RTFx (Real-Time Factor)

```python
rtfx = audio_duration_s / processing_time_s
```

Higher is faster than real-time (e.g. a 15-second clip processed in 3
seconds → RTFx 5x). For this script specifically, `processing_time_s` is
**pure on-device inference time** — only the `model.transcribe()` call,
timed with `time.monotonic()`, with one-time model loading excluded. This
is a different basis than the hosted-API scripts in this repo, which time
network round-trip instead of local compute — the report file notes this
explicitly so the two aren't compared as if they were the same
measurement.

### Corpus-level aggregation (the headline numbers)

The report's "Global WER" and "Global RTFx" are **not** an average of each
file's own WER/RTFx. They're computed by pooling raw counts across every
file first, then dividing once:

```python
total_n     = sum(s["ref_word_count"] for s in scored)
total_err   = sum(s["substitutions"] + s["insertions"] + s["deletions"] for s in scored)
micro_wer   = total_err / total_n            # -> "Global WER"

total_audio = sum(s["audio_duration_s"] for s in scored)
total_proc  = sum(s["processing_time_s"] for s in scored)
overall_rtfx = total_audio / total_proc       # -> "Global RTFx"
```

This is **micro-averaging**: every word across the whole dataset counts
once, as if all 24 files were one concatenated transcript. The alternative
— averaging each file's own WER% (**macro-averaging**) — is also computed
(`macro_wer`) and shown in the report for comparison, but is deliberately
*not* used as the headline. Macro-averaging over-weights short clips: a
15-second file with 1 error in 10 words (10% WER) would count exactly as
much as a 331-second file with 100 errors in 800 words (12.5% WER),
despite the long file containing 80x more actual speech. Micro-averaging
weights every word equally regardless of which file it came from, which is
what a single "how accurate is this system on this dataset" number should
reflect.

### How `write_results()` assembles the report file

One plain-text file, four sections in order:

1. **Headline block** — Global WER / Global RTFx in a bordered box, plus a
   one-line note on what RTFx is measuring (on-device vs. network).
2. **Corpus totals** — the same two numbers spelled out with their raw
   inputs (total errors, total reference words, total audio/processing
   seconds), plus the macro-average shown alongside so it's clear *why*
   the headline differs from a naive per-file average.
3. **Per-sample table** — one row per file: WER%, S/I/D counts, N,
   duration, processing time, RTFx.
4. **Full transcripts** — reference and hypothesis text printed side by
   side per file, so any number in the table above can be manually spot-
   checked against the actual text (this is what surfaced the
   `max_new_tokens` truncation bug in section 6 — the aggregate numbers
   alone wouldn't have shown *why* two files scored badly, only that they
   did).

## Summary: known-good install sequence

```bash
# 1. Confirm GPU is visible
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# 2. Install
pip install --user qwen-asr torch python-dotenv jiwer openai-whisper

# 3. Fix Pillow if needed (only if you hit the Resampling AttributeError)
pip install --user --upgrade "Pillow>=10"

# 4. Fix cuDNN if needed (only if you hit SUBLIBRARY_VERSION_MISMATCH)
pip install --user --upgrade nvidia-cudnn-cu13

# 5. Smoke test one short file before running a full batch

# 6. Set max_new_tokens based on your longest audio file, not the model
#    card's default -- verify against real output, not just "it ran"
```
