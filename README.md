# SENTRIX - Sentiment Analysis System

A finetuned RoBERTa model running locally on your machine, served through a Flask backend, with a mobile-responsive frontend hosted on GitHub Pages. The model is hosted on Hugging Face Hub and downloads automatically on first run. No zip files, no USB sticks, no manual weight transfers between machines.

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

This project implements three core NLP tasks on social media text:

- **Aspect-Based Sentiment Analysis** - The model does not just return "negative." It identifies *what* is negative. Battery life? Camera? Delivery? The aspect extraction surfaces that automatically.
- **Multilingual Sentiment Analysis** - Handles English, French, Spanish, German, and Portuguese. The base model's Twitter pretraining gives it genuine cross-lingual capability without any language-specific fine-tuning.
- **Emoji-Aware Sentiment Analysis** - Detects and surfaces emojis as separate sentiment signals. "Great product" and "great product" followed by three skull emojis carry different weight, and the pipeline surfaces that distinction.

The model (`prem79/sentrix_roberta_V2`) is fine-tuned from Cardiff NLP's twitter-roberta checkpoint, trained on Kaggle with GPU acceleration, achieving 88.21% accuracy on 40,000 test samples. It runs entirely on your device with no cloud inference costs.

---

## Architecture

```
Browser / Mobile Device
        |
        | HTTP (local network)
        v
+---------------------------+
|   GitHub Pages            |  index.html  (static frontend, zero build step)
|   prem-479.github.io      |
+---------------------------+
        |
        | POST /analyze
        v
+---------------------------+
|   Flask REST API          |  app.py  (runs locally on your machine)
|   localhost:5000          |
+---------------------------+
        |
        v
+---------------------------+
|   Hugging Face Hub        |  prem79/sentrix_roberta_V2
|   (first run only)        |  downloads once, cached locally on every run after
+---------------------------+
        |
        v
+---------------------------+
|   PyTorch Inference       |  CPU / CUDA / Apple MPS
|   RoBERTa-base            |  inference engine
+---------------------------+
```

Model caching behavior:

| Run | What happens |
|---|---|
| First run | Downloads ~500MB from Hugging Face, saves to your machine's cache |
| Every run after that | Loads from cache in seconds, no internet required |
| Offline after first run | Works completely offline, the model is on your machine |

---

## Requirements

- Python 3.9 or higher
- Internet connection on the first run only, after that you are free

Python packages in `requirements.txt`:

```
flask>=3.0.0
flask-cors>=4.0.0
torch>=2.0.0
transformers>=4.35.0
safetensors>=0.4.0
```

That is it. No model files to download manually. No zip files. No Google Drive links. The `transformers` library handles all of that for you.

---

## Quickstart

This is the part where most READMEs give you fifteen steps and a warning about system dependencies. This one does not.

```bash
# clone the repo
git clone https://github.com/prem-479/sentrix_ML_IA.git
cd sentrix_ML_IA

# set up the environment
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# .\venv\Scripts\activate         # Windows, different syntax, same result

# install dependencies
# cpu-only torch is smaller and unless you have an NVIDIA card it is the same speed
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# start the server
python3 app.py
```

First run output will look like this:

```
Model source: Hugging Face Hub (prem79/sentrix_roberta_V2)
First run will download ~500MB and cache locally.
Subsequent runs load from cache instantly.

INFO: Hardware: CPU
INFO: Loading model: prem79/sentrix_roberta_V2
Downloading model.safetensors: 100%  499M/499M
INFO: Model loaded on cpu

  Device:   cpu
  API:      http://localhost:5000
  Frontend: open index.html in browser
```

Go to https://prem-479.github.io/sentrix_ML_IA/, enter your machine's IP in the Server Address box, and you are done.

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

The server runs on `0.0.0.0:5000`. That means localhost on your machine and `192.168.x.x:5000` from any other device on the same network. The terminal will show you both. Use the `192.168.x.x` one for your phone.

Stop the server with `Ctrl + C`.

---

## Deployment

Three layers. Each one does exactly one job.

```
Layer 1: Model weights    - Hugging Face Hub  (prem79/sentrix_roberta_V2)
Layer 2: Frontend         - GitHub Pages      (prem-479.github.io/sentrix_ML_IA)
Layer 3: Backend/inference - Your laptop      (python3 app.py)
```

---

### Layer 1 - Model on Hugging Face Hub

The model is at `prem79/sentrix_roberta_V2`. The `app.py` references it with one line:

```python
MODEL_PATH = "prem79/sentrix_roberta_V2"
```

When you run `app.py` on a new machine, `transformers` sees that string, goes to Hugging Face, downloads the model, caches it, and loads it. Next time it skips the download. This is the whole point of hosting it on the Hub - you never have to email yourself a zip file again.

Cache locations in case you need to find or delete it:

| Platform | Cache path |
|---|---|
| Linux | `~/.cache/huggingface/hub/` |
| macOS | `~/.cache/huggingface/hub/` |
| Windows | `C:\Users\YourName\.cache\huggingface\hub\` |

---

### Layer 2 - Frontend on GitHub Pages

`index.html` is a single static file with no build step, no npm install, and no bundler configuration. Push it to the repo, GitHub Pages serves it.

Updating the frontend:

```bash
git add index.html
git commit -m "update frontend"
git push
```

GitHub Pages redeploys within a minute.

Setting up Pages on a new fork if needed:

1. Repository Settings > Pages
2. Source: Deploy from a branch
3. Branch: main, Folder: / (root)
4. Save

Your `.gitignore` should look like this. Do not commit your venv or pycache, nobody wants those:

```
venv/
__pycache__/
*.pyc
```

The model files are not in this repo. They live on Hugging Face. That is the whole point.

---

### Layer 3 - Backend on Your Machine

The backend cannot go on GitHub Pages. GitHub Pages is a static file host. It does not run Python. This is a common source of confusion and the answer is always the same: run `python3 app.py` on your laptop before you open the frontend.

For a real production deployment you would need a VPS, a domain, and an SSL certificate, at which point this stops being a student project and starts being infrastructure. That is out of scope here.

---

### Serving Locally for Presentations (Important)

GitHub Pages runs on HTTPS. Your backend runs on HTTP. Browsers block HTTP requests from HTTPS pages. This is called Mixed Content and it will make the status indicator stay red on the GitHub Pages URL.

The fix is simple: do not use the GitHub Pages URL during the demo. Serve the frontend locally instead.

```bash
# terminal 1: backend
source venv/bin/activate
python3 app.py

# terminal 2: frontend
python3 -m http.server 8080
```

Open `http://localhost:8080` in your browser. Status goes green. Everything works.

For mobile on the same WiFi:

```
Frontend on phone:  http://192.168.x.x:8080
Server Address box: http://192.168.x.x:5000
```

---

## Mobile Access

1. Run `python3 app.py`. The terminal shows two addresses.
2. The `192.168.x.x:5000` one works from any device on the same network.
3. Open the frontend on your phone.
4. Paste `http://192.168.x.x:5000` into the Server Address field. Tap Set.
5. Green status. You are connected.

The address saves to localStorage. You only have to do this once per device.

---

## API Reference

### GET /health

Check if the server and model are running.

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

Send text, get sentiment.

**Request:**
```json
{ "text": "The camera is absolutely stunning at night" }
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
  "analysis": "Very high confidence positive sentiment (95.1%). Key aspects: camera, night.",
  "raw_text": "The camera is absolutely stunning at night",
  "debug": {
    "num_labels": 3,
    "labels": ["negative", "neutral", "positive"],
    "raw_probs": [4.21, 0.74, 95.05],
    "override_applied": false
  }
}
```

**curl:**
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Worst product I have ever bought"}'
```

---

### POST /batch

Up to 20 texts at once. Same format per item.

```bash
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great quality", "Terrible experience", "It is fine I guess"]}'
```

---

## NLP Capabilities

### Preprocessing

Before anything reaches the model, two substitutions happen:

| Pattern | Replacement | Why |
|---|---|---|
| URLs | `http` | Standard RoBERTa tweet token |
| @mentions | `@user` | Standard RoBERTa tweet token |

The model was pretrained expecting these tokens. Skipping this step hurts accuracy on anything with links or mentions.

### Aspect Extraction

Stopword filtering pulls up to four content words from the input. These become the aspect chips in the UI. It is not spaCy, it is not a dependency parser, it is a stopword list - and for the purpose of highlighting what the review is about, it works well enough.

### Language Detection

Pattern matching against closed-class vocabulary. If the text has enough French or Spanish function words, it gets flagged. This does not affect the model's inference - the base model handles cross-lingual input natively. It is purely for displaying the correct flag in the UI.

### Lexical Override

When model confidence is below 65% and the input contains explicit hostile language, the classification flips to NEGATIVE. The `debug.override_applied` field in the response tells you when this happened. This exists because RoBERTa was trained on casual Twitter where people use profanity affectionately, and sometimes the model genuinely cannot tell the difference. The override can.

---

## Cross-Platform Notes

| Platform | Activate venv |
|---|---|
| Linux / macOS | `source venv/bin/activate` |
| Windows PowerShell | `.\venv\Scripts\activate` |
| Windows CMD | `venv\Scripts\activate.bat` |

Do not move your `venv` folder between machines. It will not work. Create a new one on each machine with `python3 -m venv venv`. This takes two minutes and saves hours of debugging path issues.

**Windows Firewall** - on first run, Windows will ask if Python can access the network. Allow both Private and Public. If you click no by accident, the mobile connection will fail silently and you will spend twenty minutes wondering why.

**macOS Firewall** - same deal, click Allow when prompted.

---

## Hardware Acceleration

The server auto-detects the best available device at startup. You do not configure anything.

| Machine | Hardware | Device used |
|---|---|---|
| Linux / Windows with NVIDIA GPU | CUDA | `cuda` - fastest |
| Mac with M1 / M2 / M3 chip | Metal | `mps` - fast |
| Anything else | CPU | `cpu` - works fine, just slower |

For a 500MB sentiment model on CPU you are looking at about 0.2-0.5 seconds per inference. That is fine for a demo. If you need it faster, get a GPU or use a smaller model.

---

## Project Structure

```
sentrix_ML_IA/
|
|-- app.py              # Flask backend - inference, API, device detection, lexical override
|-- index.html          # Frontend - everything in one file, no build step required
|-- requirements.txt    # Python dependencies, five lines
|-- README.md           # This file
|
|-- venv/               # Do not commit this. Do not transfer this between machines.
```

Model weights are not in this repository. They are at `huggingface.co/prem79/sentrix_roberta_V2` and download automatically on first run.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: flask` | You forgot to activate the venv | `source venv/bin/activate` then try again |
| `OSError: not a local folder and not a valid model identifier` | First run with no internet | Connect to the internet once so the model downloads and caches |
| Model repo says private or 404 | Repo visibility changed | Go to `huggingface.co/prem79/sentrix_roberta_V2` and make it public |
| Status indicator stays red on GitHub Pages | Mixed Content block, HTTPS cannot call HTTP | Use `python3 -m http.server 8080` and open `http://localhost:8080` instead |
| Works on your laptop but not on your phone | Phone is loading the GitHub Pages HTTPS URL | Give the phone `http://192.168.x.x:8080` instead |
| `Disk quota exceeded` during pip install | PyTorch with GPU support is enormous | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Low confidence on obvious negative text | Model trained on Twitter where sarcasm and slang are common | Lexical override handles explicit cases automatically |
| Wrong language flag shown | Heuristic detection not perfect | Does not affect sentiment scores, purely cosmetic |
| Port 5000 already in use | Something else is on 5000 | Change `port=5000` to `port=5001` in `app.py` and update the Server Address accordingly |

---

*Built with Flask, PyTorch, and HuggingFace Transformers.*  
*Model: prem79/sentrix_roberta_V2 - Hugging Face Hub.*  
*Trained on Kaggle. Deployed on GitHub Pages.*
