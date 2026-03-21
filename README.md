# SENTRIX — Sentiment Analysis System

A local sentiment analysis deployment using the `twitter_roberta_v2` model. The system exposes a REST API via Flask and serves a mobile-responsive frontend that connects to it over a local network.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Server](#running-the-server)
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

## Mobile Access

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

---

## Dataset

Model trained on Twitter sentiment data sourced from Kaggle. Suitable datasets include:

- Twitter Sentiment Analysis (Kaggle)
- Amazon Product Reviews
- Flipkart Reviews
- IMDB Movie Reviews

---

*Built with Flask, PyTorch, and HuggingFace Transformers. Model: cardiffnlp/twitter-roberta-base-sentiment-latest.*
