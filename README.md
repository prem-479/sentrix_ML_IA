# SENTRIX — Sentiment Analysis System

A hybrid-cloud sentiment analysis deployment. The model weights are hosted on Hugging Face Hub and downloaded automatically on first run. The frontend is hosted on GitHub Pages. The backend runs locally on any machine with Python installed — no manual model file transfers required.

Live Frontend: https://prem-479.github.io/sentrix_ML_IA/
Model on Hugging Face: https://huggingface.co/prem79/sentrix_roberta_V2

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Running the Server](#running-the-server)
- [Deployment](#deployment)
- [Mobile Access](#mobile-access)
- [API Reference](#api-reference)
- [NLP Capabilities](#nlp-capabilities)
- [Cross-Platform Notes](#cross-platform-notes)
- [Hardware Acceleration](#hardware-acceleration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project demonstrates three core NLP tasks applied to user-generated text:

- **Aspect-Based Sentiment Analysis** — Extracts specific product/service aspects (camera, battery, price) and identifies sentiment toward each.
- **Multilingual Sentiment Analysis** — Detects and processes text in English, French, Spanish, German, and Portuguese using the cross-lingual capabilities of the RoBERTa architecture.
- **Emoji-Aware Sentiment Analysis** — Extracts emoji signals from input text and incorporates them as supplementary sentiment indicators alongside the model's confidence scores.

The model (`prem79/sentrix_roberta_V2`) is a fine-tuned RoBERTa checkpoint hosted on Hugging Face Hub. The `transformers` library downloads and caches it automatically — no manual file handling required.

---

## Architecture

```
Browser / Mobile Device
        |
        | HTTP (local network)
        v
+---------------------------+
|   GitHub Pages            |  index.html  (static frontend)
|   prem-479.github.io      |
+---------------------------+
        |
        | POST /analyze  (configured IP)
        v
+---------------------------+
|   Flask REST API          |  app.py  (runs on your machine)
|   localhost:5000          |
+---------------------------+
        |
        v
+---------------------------+
|   HuggingFace Hub         |  prem79/sentrix_roberta_V2
|   (first run only)        |  downloaded and cached locally
+---------------------------+
        |
        v
+---------------------------+
|   PyTorch Inference       |  CPU / CUDA / Apple MPS
|   twitter-roberta-base    |
+---------------------------+
```

**Model caching behavior:**

| Run | Behavior |
|---|---|
| First run | Downloads ~500MB from Hugging Face Hub, saves to local cache |
| Subsequent runs | Loads from cache instantly, no network required |
| Offline (after first run) | Works fully offline from the local cache |

---

## Requirements

- Python 3.9 or higher
- Internet connection on first run only (to download the model)

Python packages (`requirements.txt`):

```
flask>=3.0.0
flask-cors>=4.0.0
torch>=2.0.0
transformers>=4.35.0
safetensors>=0.4.0
```

No model files need to be copied or moved between machines. The `transformers` library handles all downloading and caching automatically.

---

## Quickstart

This is all you need to do on any machine, including a new Mac, Windows PC, or another Linux system:

```bash
# 1. Clone the repository
git clone https://github.com/prem-479/sentrix_ML_IA.git
cd sentrix_ML_IA

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# .\venv\Scripts\activate         # Windows PowerShell

# 3. Install dependencies (CPU-only PyTorch — smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Start the server
python3 app.py
```

On first run, the model downloads automatically:

```
Model source: Hugging Face Hub (prem79/sentrix_roberta_V2)
First run will download ~500MB and cache locally.
Subsequent runs load from cache instantly.

INFO: Loading model: prem79/sentrix_roberta_V2 (Hugging Face Hub or local cache)
Downloading model.safetensors: 100%  499M/499M
INFO: Model loaded on cpu

  Device:   cpu
  API:      http://localhost:5000
  Frontend: open index.html in browser
```

Open https://prem-479.github.io/sentrix_ML_IA/ and enter your machine's IP in the Server Address field.

---

## Running the Server

### Linux / macOS

```bash
cd sentrix_ML_IA
source venv/bin/activate
python3 app.py
```

### Windows (PowerShell)

```powershell
cd sentrix_ML_IA
.\venv\Scripts\activate
python app.py
```

The server starts on `0.0.0.0:5000` — accessible from the host machine at `http://localhost:5000` and from other devices on the same network at `http://192.168.x.x:5000`.

To stop the server: `Ctrl + C`

---

## Deployment

This project uses a three-layer hybrid deployment:

```
Layer 1: Model weights    — Hugging Face Hub  (prem79/sentrix_roberta_V2)
Layer 2: Frontend         — GitHub Pages      (prem-479.github.io/sentrix_ML_IA)
Layer 3: Backend/inference — Your local machine (python3 app.py)
```

---

### Layer 1 — Model on Hugging Face Hub

The model is already uploaded at `prem79/sentrix_roberta_V2`. The `app.py` backend references it by this Hub ID:

```python
MODEL_PATH = "prem79/sentrix_roberta_V2"
```

The `transformers` library resolves this string automatically — if the model is not in the local cache, it downloads it from Hugging Face. If it is cached, it loads from disk. No code changes are needed when moving to a new machine.

The local cache location is:

| Platform | Cache Path |
|---|---|
| Linux | `~/.cache/huggingface/hub/` |
| macOS | `~/.cache/huggingface/hub/` |
| Windows | `C:\Users\username\.cache\huggingface\hub\` |

---

### Layer 2 — Frontend on GitHub Pages

The frontend is a single static HTML file (`index.html`) with no build step.

#### Updating and redeploying

```bash
# Edit index.html, then:
git add index.html
git commit -m "Update frontend"
git push
```

GitHub Pages automatically redeploys within 1-2 minutes of every push to `main`.

#### Enable GitHub Pages (if setting up a new fork)

1. Go to repository **Settings** > **Pages**
2. Under Source, select **Deploy from a branch**
3. Set branch to `main`, folder to `/ (root)`
4. Click **Save**

The site will be published at `https://your-username.github.io/your-repo-name/`.

#### .gitignore

The following should not be committed:

```
venv/
__pycache__/
*.pyc
```

Model files are not stored in this repository — they live on Hugging Face Hub.

---

### Layer 3 — Backend on Your Machine

The backend must be running locally whenever you use the frontend. It cannot be deployed to GitHub Pages or any static host.

For a permanent production deployment, the backend would need a server (VPS, AWS EC2, etc.) with a domain and SSL certificate. This is outside the scope of a local demo. For presentation purposes, running `python3 app.py` on a laptop connected to the same WiFi as the demo device is sufficient.

---

### Serving the Frontend Locally (Recommended for Presentations)

GitHub Pages is served over HTTPS. Browsers block requests from HTTPS pages to plain HTTP backends (Mixed Content policy). To avoid this during a demo:

```bash
# In the project folder (venv does not need to be active)
python3 -m http.server 8080
```

Then open `http://localhost:8080` instead of the GitHub Pages URL. Requests from `http` to `http://localhost:5000` are not blocked.

For mobile access on the same network:

```
Frontend:  http://192.168.x.x:8080
Backend:   http://192.168.x.x:5000   (enter this in Server Address field)
```

---

## Mobile Access

1. Run `python3 app.py` on your machine.
2. The terminal displays two addresses — use the `192.168.x.x` one:
   ```
   Running on http://127.0.0.1:5000      <- host machine only
   Running on http://192.168.x.x:5000    <- any device on the same network
   ```
3. On your phone, open `http://192.168.x.x:8080` (local server) or the GitHub Pages URL.
4. In the **Server Address** field on the Analyze tab, enter `http://192.168.x.x:5000` and tap **Set**.
5. The status indicator turns green when connected.

The configured address is saved to `localStorage` and persists between sessions.

---

## API Reference

### GET /health

Returns server, model, and hardware status.

```json
{
  "status": "online",
  "model_loaded": true,
  "model_source": "prem79/sentrix_roberta_V2",
  "num_labels": 3,
  "labels": ["negative", "neutral", "positive"],
  "device": "cpu"
}
```

---

### POST /analyze

Analyzes a single text input.

**Request:**
```json
{
  "text": "The camera is absolutely stunning at night"
}
```

**Response:**
```json
{
  "sentiment": "POSITIVE",
  "pos": 95.05,
  "neg": 4.21,
  "neu": 0.74,
  "lang": "EN",
  "aspects": ["camera", "night"],
  "emojis": [],
  "analysis": "Very high confidence positive sentiment (95.1%). Key aspects: camera, night. Subject expresses satisfaction or approval.",
  "raw_text": "The camera is absolutely stunning at night",
  "debug": {
    "num_labels": 3,
    "labels": ["negative", "neutral", "positive"],
    "raw_probs": [4.21, 0.74, 95.05],
    "override_applied": false
  }
}
```

**curl example:**
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Worst product I have ever bought"}'
```

---

### POST /batch

Analyzes up to 20 texts in a single request.

**Request:**
```json
{
  "texts": [
    "Great product, highly recommend",
    "Terrible quality, broke in one day",
    "It is okay, nothing special"
  ]
}
```

**Response:**
```json
{
  "count": 3,
  "results": [
    { "sentiment": "POSITIVE", "pos": 91.2, "neg": 6.1, "neu": 2.7 },
    { "sentiment": "NEGATIVE", "pos": 3.4,  "neg": 95.1, "neu": 1.5 },
    { "sentiment": "NEUTRAL",  "pos": 22.1, "neg": 18.3, "neu": 59.6 }
  ]
}
```

---

## NLP Capabilities

### Preprocessing

| Pattern | Replacement | Reason |
|---|---|---|
| URLs (`http://...`) | `http` | Standard RoBERTa tweet token |
| Mentions (`@username`) | `@user` | Standard RoBERTa tweet token |

### Aspect Extraction

Rule-based stopword filtering returns up to 4 content words representing product or service aspects (camera, battery, display, price, delivery).

### Language Detection

Heuristic regex matching against closed-class vocabulary:

| Language | Example trigger words |
|---|---|
| French | `très`, `qualité`, `satisfait`, `incroyable` |
| Spanish | `muy`, `producto`, `calidad`, `gracias` |
| German | `sehr`, `gut`, `danke`, `schlecht` |
| Portuguese | `muito`, `produto`, `qualidade`, `bom` |

### Lexical Override

When model confidence is below 65% and the input contains unambiguous hostile vocabulary (profanity, explicit threats), the classification is overridden to NEGATIVE. The response includes `"override_applied": true` and the analysis string notes the trigger words. This handles cases where the model — trained on casual Twitter data — may misclassify aggressive language as positive.

---

## Cross-Platform Notes

| Platform | Activate virtual environment |
|---|---|
| Linux / macOS | `source venv/bin/activate` |
| Windows PowerShell | `.\venv\Scripts\activate` |
| Windows CMD | `venv\Scripts\activate.bat` |

**Model path** — `app.py` uses the Hugging Face Hub ID `"prem79/sentrix_roberta_V2"` which is OS-agnostic. The `transformers` library resolves caching paths correctly on all platforms.

**Windows Firewall** — On first run, Windows prompts to allow Python through the firewall. Both Private and Public network boxes must be checked for local network access to work.

**macOS Firewall** — macOS may prompt to allow incoming network connections. Click Allow.

---

## Hardware Acceleration

`app.py` automatically selects the best available compute device at startup:

| Platform | Hardware | Device | Notes |
|---|---|---|---|
| Linux / Windows | NVIDIA GPU | CUDA | Fastest |
| macOS | M1 / M2 / M3 | MPS | Fast — Metal Performance Shaders |
| Any | CPU only | CPU | Default fallback |

The selected device is reported in the startup banner and in the `/health` response.

To force CPU inference:
```python
# In app.py, after the get_device() call:
DEVICE = torch.device("cpu")
```

---

## Project Structure

```
sentrix_ML_IA/
|
|-- app.py              # Flask backend — model inference, API routes, device detection
|-- index.html          # Frontend — mobile-responsive single-file app (GitHub Pages)
|-- requirements.txt    # Python dependencies
|-- README.md           # This file
|
|-- venv/               # Virtual environment — do not commit, recreate on each machine
```

The model weights are not stored in this repository. They are hosted at:
`https://huggingface.co/prem79/sentrix_roberta_V2`

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: flask` | Running outside virtual environment | Activate venv before running |
| `OSError: prem79/sentrix_roberta_V2 is not a local folder` | No internet, model not cached yet | Connect to internet for first run to download the model |
| `IndexError: index 2 out of bounds` | Code assumed 3 classes, model has 2 | Labels are auto-detected from `config.json` in current version |
| Model not loading | Wrong Hub ID or repository is private | Verify at `huggingface.co/prem79/sentrix_roberta_V2` that the repo is public |
| Mobile cannot connect | Firewall blocking port 5000 | Allow Python through firewall; confirm both devices on same WiFi |
| Status stays Offline on GitHub Pages | Mixed Content block (HTTPS to HTTP) | Serve locally with `python3 -m http.server 8080`, open `http://localhost:8080` |
| Works on PC but not on phone | Phone loading GitHub Pages URL (HTTPS) | Use `http://192.168.x.x:8080` on the phone instead |
| Download very slow on first run | 500MB model over network | Normal — subsequent runs load from cache instantly |
| `Disk quota exceeded` during install | Full GPU PyTorch too large | Use: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Wrong language detected | Ambiguous vocabulary | Known limitation of heuristic detection; does not affect sentiment scores |
| Low confidence on profanity | Model trained on casual Twitter data | Lexical override handles this automatically |

---

*Built with Flask, PyTorch, and HuggingFace Transformers.*
*Model: prem79/sentrix_roberta_V2 — hosted on Hugging Face Hub.*
