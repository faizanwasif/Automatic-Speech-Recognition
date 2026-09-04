# Parakeet Speech-to-Text Demo

A customer-facing demo: pick a sample audio clip (or upload your own), get it
transcribed live by NVIDIA's hosted Parakeet-TDT 0.6B v2 model.

## How it's wired together

```
Browser (public/index.html)
   |  HTTP multipart upload
   v
Flask proxy (server/app.py), on Render
   |  1. ffmpeg: convert upload -> mono/16kHz PCM WAV
   |  2. gRPC call (riva.client) to grpc.nvcf.nvidia.com
   v
NVIDIA Parakeet NIM (hosted, build.nvidia.com)
```

The proxy exists for two reasons:
1. **NVIDIA's hosted Parakeet API is gRPC, not REST** — browsers can't call it directly.
2. **Keeps your `NVIDIA_API_KEY` off the client** — it never appears in page source.

## Deploying (one-time setup, ~10 minutes)

You'll need a free [Render](https://render.com) account (GitHub login is fine, no credit card required for the free tier).

### 1. Push this folder to a GitHub repo

```bash
cd parakeet-demo
git init
git add .
git commit -m "Parakeet demo"
git remote add origin <your-new-repo-url>
git push -u origin main
```

### 2. Deploy via Render Blueprint

- Go to [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints) → **New Blueprint Instance**
- Connect the GitHub repo you just pushed
- Render reads `render.yaml` and creates two services automatically:
  - `parakeet-demo-api` — the Python/ffmpeg/gRPC proxy (Docker)
  - `parakeet-demo-site` — the static frontend

### 3. Set your API key

- Open the `parakeet-demo-api` service in the Render dashboard → **Environment**
- Add `NVIDIA_API_KEY` = your key from build.nvidia.com (starts with `nvapi-`)
- Save — Render redeploys automatically

### 4. Confirm the frontend points at the right API URL

`public/index.html` already assumes your API service is named `parakeet-demo-api`
(giving it the URL `https://parakeet-demo-api.onrender.com`). If you named the
Render service something else, edit this line in `public/index.html`:

```js
: "https://parakeet-demo-api.onrender.com");
```

### 5. Test it

Open the `parakeet-demo-site` URL Render gives you. Pick a sample, hit
Transcribe. First request after idle may take ~30-50s (Render's free tier
spins services down when unused, then cold-starts) — normal, just a
one-time wait per idle period, not per request.

## Local development

Terminal 1 (proxy):
```bash
cd server
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-...
python app.py            # listens on :8000 by default; the frontend expects :8091 locally
PORT=8091 python app.py  # match what public/index.html expects on localhost
```

Terminal 2 (static site):
```bash
cd public
python3 -m http.server 8092
```

Then open `http://localhost:8092`.

## Notes / known limits

- **Audio bounds**: NVIDIA's hosted endpoint requires 3 seconds–5 minutes of audio.
  The proxy checks this after conversion and returns a clear error outside that range.
- **ffmpeg dependency**: the Docker image installs it; if you ever move off
  Docker-based hosting, make sure the target runtime has `ffmpeg` on `PATH`.
- **Free tier cold starts**: Render's free web services sleep after ~15 min
  idle and take a bit to wake up. Fine for a demo; mention it if you're doing
  a live walkthrough so a first cold request isn't mistaken for a bug.
- **Sample files**: `public/samples/` are pre-converted to mono/16kHz WAV
  already (smaller downloads, and it's the exact format the model wants).
  `sample-2` was trimmed from ~5m31s to 4m55s to fit under the 5-minute cap.
