# SENTRIX — Sentiment Analysis System

A local sentiment analysis deployment using the `twitter_roberta_v2` model. The system exposes a REST API via Flask and serves a mobile-responsive frontend that connects to it over a local network. The frontend is hosted on GitHub Pages and communicates with a locally running backend server.

Live Frontend: https://prem-479.github.io/sentrix_ML_IA/

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
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

The system uses `cardiffnlp/twitter-roberta-base-sentiment-latest` (stored locally as `twitter_roberta_v2`) and runs entirely on-device with no external API calls required during inference.

---

## Architecture

```
Browser / Mobile Device
        |
        | HTTP (local network)
        v
  Flask REST API (app.py)
        |
        v
  HuggingFace Transformers
        |
        v
  twitter_roberta_v2 (SafeTensors)
        |
        v
  PyTorch (CPU / CUDA / MPS)
```

**Request flow:**

1. Frontend sends `POST /analyze` with raw text
2. Backend preprocesses (URL masking, mention masking)
3. RoBERTa tokenizer encodes the input (max 128 tokens)
4. Model produces logits, softmax yields confidence scores
5. Post-processing: language detection, emoji extraction, aspect extraction, lexical override
6. JSON response returned to frontend

---

## Requirements

- Python 3.9 or higher
- The `twitter_roberta_v2` model folder (must be placed in the project root)

Python packages are listed in `requirements.txt`:

```
flask>=3.0.0
flask-cors>=4.0.0
torch>=2.0.0
transformers>=4.35.0
safetensors>=0.4.0
```

---

## Installation

### Step 1 — Copy project files

Copy the project folder to your machine. Do not transfer the `venv` or `__pycache__` directories — these are environment-specific and must be recreated.

Required files:
```
app.py
index.html
requirements.txt
twitter_roberta_v2/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    training_args.bin
```

### Step 2 — Create a virtual environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3 — Install dependencies

**Standard (CPU-only, works on all platforms):**
```bash
pip install -r requirements.txt
```

**CPU-only PyTorch (smaller download, no GPU libraries):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Running the Server

```bash
python app.py
```

On startup, the server will:

1. Auto-detect available hardware (CUDA / MPS / CPU)
2. Load the model from `./twitter_roberta_v2`
3. Start listening on `0.0.0.0:5000`

Expected output:
```
=======================================================
  SENTRIX // Neural Sentiment Engine
  twitter_roberta_v2 - Local Deployment
=======================================================
INFO: Hardware: CPU         (or CUDA / Apple Silicon MPS)
INFO: Loading model from: /your/path/twitter_roberta_v2
INFO: Model loaded on cpu
INFO: Labels from config: ['negative', 'neutral', 'positive']

  Device:   cpu
  API:      http://localhost:5000
  Health:   http://localhost:5000/health
  Frontend: open index.html in browser
=======================================================
```

Open `index.html` in your browser. The status indicator will turn green when connected.

---

## Deployment

This project uses a split deployment model:

- **Frontend** — `index.html` is a single static HTML file hosted on GitHub Pages. It requires no build step and has no dependencies beyond the Google Fonts CDN.
- **Backend** — `app.py` runs on your local machine. The frontend connects to it over your local network via the IP address you configure in the Server Address field.

The backend cannot be hosted on GitHub Pages or any static hosting service because it requires Python, PyTorch, and file system access to the model weights.

---

### Frontend — GitHub Pages

#### Initial Setup

1. Create a GitHub repository (public or private).

2. Push the project to GitHub:
   ```bash
   git init
   git add index.html README.md requirements.txt app.py
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/your-username/your-repo.git
   git push -u origin main
   ```
   Note: do not commit the `twitter_roberta_v2/` folder (model weights) or the `venv/` folder. Add them to `.gitignore`:
   ```
   venv/
   __pycache__/
   twitter_roberta_v2/
   *.bin
   *.safetensors
   ```

3. Enable GitHub Pages:
   - Go to your repository on GitHub
   - Navigate to **Settings** > **Pages**
   - Under **Source**, select **Deploy from a branch**
   - Set branch to `main` and folder to `/ (root)`
   - Click **Save**

4. GitHub will publish the site within 1-2 minutes. The URL will be:
   ```
   https://your-username.github.io/your-repo-name/
   ```

#### Updating the Frontend

After editing `index.html`:
```bash
git add index.html
git commit -m "Update frontend"
git push
```
GitHub Pages automatically rebuilds and redeploys on every push to `main`.

---

### Backend — Local Server

The backend must be running on your machine whenever you use the frontend. GitHub Pages only hosts `index.html` — all sentiment analysis happens on your device.

#### Full Setup and Run (first time)

**Linux / macOS:**
```bash
# Clone or download the project
cd /path/to/project

# Place the model folder here
# twitter_roberta_v2/ must be in this directory

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CPU-only PyTorch — avoids large GPU downloads)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Start the server
python3 app.py
```

**Windows (PowerShell):**
```powershell
# Navigate to project folder
cd C:\path\to\project

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Start the server
python app.py
```

#### Subsequent Runs (after first-time setup)

**Linux / macOS:**
```bash
cd /path/to/project
source venv/bin/activate
python3 app.py
```

**Windows (PowerShell):**
```powershell
cd C:\path\to\project
.\venv\Scripts\activate
python app.py
```

#### Connecting the Frontend to the Backend

Once the server is running:

1. Open https://prem-479.github.io/sentrix_ML_IA/ in your browser or on your phone.
2. Look at the terminal output for the server address:
   ```
   Running on http://127.0.0.1:5000      <- PC only
   Running on http://192.168.x.x:5000    <- use this for mobile / GitHub Pages
   ```
3. On the Analyze tab, paste the address into the **Server Address** field and click **Set**.
4. The status indicator will turn green: `Online // 3 classes`.

The address is saved in the browser's `localStorage` and persists between sessions. You only need to set it once per device.

---

### Why the Frontend Cannot Connect Automatically

GitHub Pages is served over HTTPS. Browsers block requests from an HTTPS page to a plain HTTP backend (this is called a Mixed Content block). This means:

- `http://localhost:5000` will be blocked when accessed from the GitHub Pages URL
- `http://192.168.x.x:5000` will also be blocked from the GitHub Pages URL in most browsers
- **Solution:** Open `index.html` directly from your file system (`file:///...`) or serve it locally with `python -m http.server 8080`, then use `http://localhost:8080`. Requests from `http` to `http` are not blocked.

Alternatively, for a production deployment, you would place the backend behind a reverse proxy (nginx or Caddy) with a valid SSL certificate, making it accessible as `https://your-domain.com/analyze`. This is beyond the scope of a local demo but is documented in the Troubleshooting section.

---

### Serving the Frontend Locally (Recommended for Demo)

To avoid the HTTPS Mixed Content issue during a presentation:

```bash
# In the project folder (venv does not need to be active for this)
python3 -m http.server 8080
# or on Windows:
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser. The backend at `http://localhost:5000` will connect without any Mixed Content restrictions.

For mobile access on the same network, open `http://192.168.x.x:8080` on your phone and set the Server Address to `http://192.168.x.x:5000`.

---



The server binds to `0.0.0.0`, meaning it is reachable from any device on the same local network.

1. Run `python app.py` — the terminal will display two addresses:
   - `http://127.0.0.1:5000` — accessible only from the host machine
   - `http://192.168.x.x:5000` — accessible from any device on the same network

2. On your mobile device, open `index.html` in a browser (or serve it with `python -m http.server 8080` and navigate to `http://192.168.x.x:8080`).

3. In the Server Address field on the Analyze tab, enter the `192.168.x.x:5000` address and tap **Set**.

The address is saved to `localStorage` and persists across sessions.

---

## API Reference

### GET /health

Returns server and model status.

```json
{
  "status": "online",
  "model_loaded": true,
  "model_path": "./twitter_roberta_v2",
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
    { "text": "...", "sentiment": "POSITIVE", "pos": 91.2, "neg": 6.1, "neu": 2.7, ... },
    { "text": "...", "sentiment": "NEGATIVE", "pos": 3.4, "neg": 95.1, "neu": 1.5, ... },
    { "text": "...", "sentiment": "NEUTRAL",  "pos": 22.1, "neg": 18.3, "neu": 59.6, ... }
  ]
}
```

---

## NLP Capabilities

### Preprocessing

Before tokenization, the following transformations are applied to normalize Twitter-style text:

| Pattern | Replacement | Reason |
|---|---|---|
| URLs (`http://...`) | `http` | Model token |
| Mentions (`@username`) | `@user` | Model token |

### Aspect Extraction

Noun-phrase extraction using rule-based stopword filtering. Returns up to 4 content words from the input that represent potential product or service aspects.

### Language Detection

Heuristic regex matching against vocabulary patterns for:

| Language | Detection Method |
|---|---|
| French | Closed-class words: `est`, `très`, `qualité`, `satisfait` |
| Spanish | Closed-class words: `muy`, `producto`, `calidad`, `gracias` |
| German | Closed-class words: `ist`, `sehr`, `gut`, `danke` |
| Portuguese | Closed-class words: `muito`, `produto`, `qualidade`, `bom` |

### Lexical Override

When the model confidence is below 65% and the input contains unambiguous hostile vocabulary (profanity, threats, explicit negative phrases), the classification is overridden to NEGATIVE. The API response includes `"override_applied": true` and the analysis string notes the trigger.

This handles edge cases where the model trained on casual Twitter data may interpret offensive language as positive (e.g., colloquial friendly insults).

---

## Cross-Platform Notes

| Platform | Environment Activation | Notes |
|---|---|---|
| Linux / macOS | `source venv/bin/activate` | Standard |
| Windows PowerShell | `.\venv\Scripts\activate` | Run as standard user |
| Windows CMD | `venv\Scripts\activate.bat` | Alternative |

**Path handling** — `app.py` uses `os.path.join(".", "twitter_roberta_v2")` which resolves correctly on all operating systems regardless of slash convention.

**Windows Firewall** — On first run, Windows will prompt to allow Python through the firewall. Both Private and Public network options must be checked for local network access to work.

**macOS Firewall** — macOS may prompt to allow incoming network connections. Click Allow.

---

## Hardware Acceleration

`app.py` automatically selects the best available compute device at startup. No configuration is required.

| Platform | Hardware | Device | Notes |
|---|---|---|---|
| Linux / Windows | NVIDIA GPU | CUDA | Fastest — requires CUDA toolkit |
| macOS | M1 / M2 / M3 | MPS | Fast — Metal Performance Shaders |
| Any | CPU only | CPU | Default fallback — fully functional |

The selected device is reported at startup and included in the `/health` response.

To force CPU inference regardless of available hardware, add this line to `app.py` after the `get_device()` call:
```python
DEVICE = torch.device("cpu")
```

---

## Project Structure

```
project-root/
|
|-- app.py                    # Flask backend, model inference, API routes
|-- index.html                # Mobile-responsive frontend (single file)
|-- requirements.txt          # Python dependencies
|-- README.md                 # This file
|
|-- twitter_roberta_v2/       # Model directory (not included in source control)
|   |-- config.json           # Model config, label mapping
|   |-- model.safetensors     # Model weights
|   |-- tokenizer.json        # Tokenizer vocabulary
|   |-- tokenizer_config.json # Tokenizer configuration
|   |-- training_args.bin     # Training metadata
|
|-- venv/                     # Virtual environment (do not transfer between machines)
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: flask` | Running outside virtual environment | Activate venv before running |
| `IndexError: index 2 out of bounds` | Model has 2 classes, code assumed 3 | Labels auto-detected from `config.json` in current version |
| `Model folder not found` | Wrong working directory | Run `python app.py` from the project root |
| Mobile cannot connect | Firewall blocking port 5000 | Allow Python through firewall; ensure both devices on same network |
| Wrong language detected | Ambiguous vocabulary overlap | Known limitation of heuristic detection; does not affect sentiment scores |
| Low confidence on profanity | Model trained on casual Twitter data | Lexical override handles this case automatically |
| `Disk quota exceeded` during install | Full GPU PyTorch too large | Install CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Status stays Offline on GitHub Pages | Mixed Content block (HTTPS to HTTP) | Serve `index.html` locally with `python -m http.server 8080` and open `http://localhost:8080` |
| Works on PC browser but not on phone | Phone using GitHub Pages URL (HTTPS) | Open `http://192.168.x.x:8080` on the phone instead of the GitHub Pages URL |
| `CORS` error in browser console | Flask CORS not configured | Current `app.py` uses `flask-cors` with `CORS(app)` — all origins allowed by default |

---

## Dataset

Model trained on Twitter sentiment data sourced from Kaggle. Suitable datasets include:

- Twitter Sentiment Analysis (Kaggle)
- Amazon Product Reviews
- Flipkart Reviews
- IMDB Movie Reviews

---

*Built with Flask, PyTorch, and HuggingFace Transformers. Model: cardiffnlp/twitter-roberta-base-sentiment-latest.*
