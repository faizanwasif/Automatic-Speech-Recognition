# ASR Provider Benchmark & Parakeet Demo

Research and a live demo comparing speech-to-text (ASR) providers on a fixed
audio dataset, plus a customer-facing demo app built on NVIDIA's Parakeet.

## Repo layout

```
.
├── data/                          Test audio + ground-truth reference transcripts
│   ├── s1/, s5/                   Two hand-picked long-form phone-call recordings
│   └── Audio-Team-English/        People's Speech corpus clips (audio + transcripts)
├── dataset_manifest.txt           Fixed 24-file / ~12.5-min subset used for every
│                                  provider benchmark below, so results are comparable
├── transcribe-3.5-google.py       Benchmark script: Gemini (gemini-3.5-transcribe)
├── transcribe-deepgram-nova3.py   Benchmark script: Deepgram (nova-3)
├── transcribe-assemblyai-u35pro.py Benchmark script: AssemblyAI (universal-3-5-pro)
├── transcribe-qwen3-asr.py        Benchmark script: Qwen3-ASR-1.7B (LOCAL GPU, no API key)
├── results_gemini.txt             Output: WER / RTFx results, Gemini
├── results_deepgram.txt           Output: WER / RTFx results, Deepgram
├── results_assemblyai.txt         Output: WER / RTFx results, AssemblyAI
├── results_qwen3asr.txt           Output: WER / RTFx results, Qwen3-ASR
├── ON_DEVICE_SETUP.md             How the local-GPU (Qwen3-ASR) setup was built and
│                                  debugged -- read this before running that script
└── parakeet-demo/                 Standalone customer demo app (see its own README)
    ├── public/                    Static frontend (sample picker, upload, playback)
    ├── server/                    Flask proxy -> NVIDIA Parakeet gRPC API
    └── render.yaml                One-shot deploy config for Render
```

## Benchmark methodology

Each provider script transcribes the same fixed 24-file manifest
(`dataset_manifest.txt`) and reports two numbers:

- **Global WER** (Word Error Rate) — `Σ(Substitutions + Insertions + Deletions) / Σ(Reference Words)`,
  pooled across every file (not averaged per-file, which over-weights short clips).
  Text is normalized (Whisper's `EnglishTextNormalizer`) before scoring so formatting
  differences (numbers, punctuation, capitalization) aren't counted as errors.
- **Global RTFx** (Real-Time Factor) — `Total Audio Duration / Total Processing Time`.
  Higher = faster than real-time. This measures end-to-end API latency, not
  on-device compute — see the NOTE in each results file for what's included.

Run any script directly (`python3 transcribe-deepgram-nova3.py`) — each is
self-contained; see the top-of-file comments for setup and required env vars
(`.env`, based on `.env.example`).

### Results summary (24-file / 12.45-min manifest)

| Provider | Model | Global WER | Global RTFx | Runs |
|---|---|---|---|---|
| AssemblyAI | universal-3-5-pro | 10.40% | 2.74x | hosted API |
| Deepgram | nova-3 | 11.98% | 6.18x | hosted API |
| Qwen3-ASR | 1.7B | 18.53% | 15.86x | local GPU |
| Gemini | gemini-3.5-transcribe | 15.23% | 5.53x | hosted API |

Qwen3-ASR's RTFx is not directly comparable to the others: it measures pure
on-device inference time (no network round-trip), while the hosted APIs'
RTFx includes network + queueing latency. See `ON_DEVICE_SETUP.md` for how
that benchmark was set up (GPU driver/cuDNN issues hit and fixed along the
way, plus a silent-truncation bug worth knowing about if you replicate it).

Full per-file breakdowns and transcripts are in each `results_*.txt` file.

## The demo app

`parakeet-demo/` is a separate, self-contained deployable app — not part of
the benchmark suite above. It lets a customer pick a sample audio clip (or
upload their own) and see it transcribed live by NVIDIA's hosted
Parakeet-TDT 0.6B v2 model. See [`parakeet-demo/README.md`](parakeet-demo/README.md)
for what it is and how to deploy it (Render, free tier).

**Only `parakeet-demo/` gets deployed** — the rest of this repo is research
history kept for reference alongside it.

## Setup

Hosted-API scripts (Gemini, Deepgram, AssemblyAI):
```bash
cp .env.example .env   # fill in your own API keys
pip install google-genai python-dotenv jiwer deepgram-sdk assemblyai openai-whisper
```

Local-GPU script (Qwen3-ASR) needs no API key but a CUDA GPU and some
version-pinning care — see [`ON_DEVICE_SETUP.md`](ON_DEVICE_SETUP.md) before
running `transcribe-qwen3-asr.py`.
