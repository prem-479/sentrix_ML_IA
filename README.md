# SENTRIX — Local Sentiment Analysis
## twitter_roberta_v2 · On-Device Deployment

---

## File Structure

```
your-project/
├── app.py                    ← Flask backend (loads your model)
├── requirements.txt          ← Python dependencies
├── index.html                ← Frontend (open in browser)
└── twitter_roberta_v2/       ← Your downloaded model folder
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_args.bin
```

---

## Setup (One Time)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python app.py
```

You should see:
```
=======================================================
  SENTRIX // Neural Sentiment Engine
  twitter_roberta_v2 — Local Deployment
=======================================================
  ✅ Model loaded successfully!
  API: http://localhost:5000
=======================================================
```

### 3. Open the frontend
Open `index.html` in your browser.
The status banner will turn **green** when connected.

---

## API Endpoints

### POST /analyze
Analyze a single text.
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The camera is amazing 📸😍"}'
```
Response:
```json
{
  "sentiment": "POSITIVE",
  "pos": 95.05,
  "neg": 4.95,
  "neu": 0.00,
  "lang": "EN 🇬🇧",
  "aspects": ["camera"],
  "emojis": ["📸", "😍"],
  "analysis": "High confidence positive sentiment..."
}
```

### POST /batch
Analyze up to 20 texts at once.
```bash
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible quality", "It is okay"]}'
```

### GET /health
Check server + model status.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model folder not found` | Make sure `twitter_roberta_v2/` is in the same folder as `app.py` |
| `pip install` fails | Try `pip install --upgrade pip` first |
| Port 5000 in use | Change `port=5000` to `port=5001` in `app.py` and update `const API` in `index.html` |
| Slow first inference | Normal — model loads into memory on first call |

---

Built with Flask + HuggingFace Transformers + PyTorch
