"""
SENTRIX — Local Sentiment Analysis API
twitter_roberta_v2 — Local Deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re, os, json, logging

logging.basicConfig(level=logging.INFO)
# ─── DEVICE AUTO-DETECTION ───────────────────────────────────────────────────
def get_device():
    """Automatically selects the best available compute device."""
    if torch.cuda.is_available():
        logger.info("Hardware: NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Hardware: Apple Silicon (MPS)")
        return torch.device("mps")
    logger.info("Hardware: CPU")
    return torch.device("cpu")

DEVICE = get_device()

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = "prem79/sentrix_roberta_V2"  # Hugging Face Hub — downloads automatically on first run

# ─── STRONG NEGATIVE SIGNAL WORDS ─────────────────────────────────────────────
# When model confidence is low (<65%) and these appear, override to NEGATIVE.
# These are unambiguously hostile/offensive in any tweet context.
STRONG_NEGATIVE_WORDS = {
    "fuck", "shit", "bitch", "bastard", "asshole", "crap", "damn",
    "idiot", "stupid", "hate", "awful", "terrible", "horrible", "disgusting",
    "pathetic", "useless", "trash", "garbage", "moron", "loser", "sucks",
    "worst", "ruined", "broken", "scam", "fraud", "waste", "rip off",
    "never again", "do not buy", "not worth", "piece of shit",
    "fuck off", "go to hell", "shut up",
    # Death threats / wishes
    "die bastard", "may u die", "may you die", "go die", "drop dead",
    "you will die", "u will die", "i hope you die", "kill yourself",
    "rot in hell", "burn in hell", "i hate you", "i hate u",
    "hope u die", "hope you die", "wish you were dead"
}

# Confidence threshold below which we apply lexical override
OVERRIDE_THRESHOLD = 65.0

# ─── LABEL AUTO-DETECTION ────────────────────────────────────────────────────
def get_labels():
    config_path = os.path.join(MODEL_PATH, "config.json")
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {})
        if id2label:
            labels = [id2label[str(i)] for i in range(len(id2label))]
            logger.info(f"Labels from config: {labels}")
            return labels
    except Exception as e:
        logger.warning(f"Could not read labels from config: {e}")
    return ["NEGATIVE", "POSITIVE"]

LANG_PATTERNS = {
    "FR": r"\b(est|les|des|une|que|très|produit|qualité|satisfait|incroyable|bonjour|merci|avec|pas|sur)\b",
    "ES": r"\b(es|los|las|una|que|muy|producto|calidad|gracias|bueno|malo|hola|pero|para)\b",
    "DE": r"\b(ist|das|der|und|sehr|gut|schlecht|produkt|qualität|danke|nicht|auch)\b",
    "PT": r"\b(é|os|as|uma|que|muito|produto|qualidade|obrigado|bom|ruim|não|com)\b",
}

POSITIVE_EMOJIS = {"😍","🥰","😊","😁","🤩","👍","❤️","💯","🔥","✨","🙌","💪","🎉","😀","😄","🌟","⭐","📸"}
NEGATIVE_EMOJIS = {"😡","😤","😠","👎","💔","😞","😢","😭","🤮","🤬","😒","💩","❌","😑"}

tokenizer = None
model = None
LABELS = []

def load_model():
    global tokenizer, model, LABELS
    logger.info(f"Loading model: {MODEL_PATH} (Hugging Face Hub or local cache)")
    try:
        LABELS = get_labels()
        logger.info(f"Using {len(LABELS)} output classes: {LABELS}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, use_safetensors=True
        )
        model.to(DEVICE)
        model.eval()
        logger.info(f"✅ Model loaded on {DEVICE}")
        return True
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        return False

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def detect_language(text):
    for lang, pattern in LANG_PATTERNS.items():
        if re.search(pattern, text.lower(), re.IGNORECASE):
            return f"{lang} " + {"FR":"🇫🇷","ES":"🇪🇸","DE":"🇩🇪","PT":"🇧🇷"}[lang]
    return "EN 🇬🇧"

def extract_emojis(text):
    seen, result = set(), []
    for ch in text:
        if ch not in seen and ch.strip() and ord(ch) > 0x1F300:
            seen.add(ch); result.append(ch)
    return result

def extract_aspects(text):
    stopwords = {
        "i","me","my","we","our","you","your","it","its","the","a","an",
        "and","or","but","in","on","at","to","for","of","is","was","are",
        "be","been","have","has","do","not","this","that","with","very",
        "so","just","really","much","more","most","also","too","quite",
        "off","get","got","let","its","they","them","their",
    }
    words = re.sub(r'[^\w\s]', ' ', text).split()
    return [w for w in words if w.lower() not in stopwords and len(w) > 3][:4]

def preprocess_tweet(text):
    text = re.sub(r'http\S+|www\S+', 'http', text)
    text = re.sub(r'@\w+', '@user', text)
    return text.strip()

def has_strong_negative_signals(text):
    """Return matched negative words/phrases found in text."""
    text_lower = text.lower()
    found = []
    for phrase in STRONG_NEGATIVE_WORDS:
        if phrase in text_lower:
            found.append(phrase)
    return found

def build_scores(probs_list):
    def normalize(label):
        l = label.lower()
        if 'neg' in l or l in ('label_0', '0'): return 'neg'
        if 'pos' in l or l in ('label_2', '2'): return 'pos'
        if 'neu' in l or l in ('label_1', '1'): return 'neu'
        return 'pos'  # 2-class index-1 fallback

    score_map = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
    for i, prob in enumerate(probs_list):
        if i < len(LABELS):
            score_map[normalize(LABELS[i])] = prob * 100

    neg = round(score_map['neg'], 2)
    neu = round(score_map['neu'], 2)
    pos = round(score_map['pos'], 2)

    if pos >= neg and pos >= neu:   sentiment = 'POSITIVE'
    elif neg >= pos and neg >= neu: sentiment = 'NEGATIVE'
    else:                           sentiment = 'NEUTRAL'

    return sentiment, pos, neg, neu

def apply_lexical_override(text, sentiment, pos, neg, neu):
    """
    When model confidence is low and strong negative words are present,
    override to NEGATIVE. Returns (sentiment, pos, neg, neu, override_note).
    """
    confidence = max(pos, neg, neu)
    matched = has_strong_negative_signals(text)

    if matched and confidence < OVERRIDE_THRESHOLD and sentiment != 'NEGATIVE':
        # Flip scores: boost neg, reduce pos
        boost = min(30.0, OVERRIDE_THRESHOLD - confidence + 10)
        neg_new = round(min(neg + boost, 99.0), 2)
        pos_new = round(max(pos - boost, 1.0), 2)
        neu_new = round(max(neu, 0.0), 2)
        note = f"Lexical override applied — strong negative signals detected: [{', '.join(matched)}]."
        return 'NEGATIVE', pos_new, neg_new, neu_new, note

    return sentiment, pos, neg, neu, None

def generate_analysis(text, sentiment, pos, neg, neu, lang, aspects, emojis, override_note=None):
    confidence = max(pos, neg, neu)
    conf_str = "very high" if confidence > 90 else "high" if confidence > 75 else "moderate" if confidence > 60 else "low"

    lang_note = f" Cross-lingual inference ({lang.split()[0]} input)." if not lang.startswith("EN") else ""

    pos_e = [e for e in emojis if e in POSITIVE_EMOJIS]
    neg_e = [e for e in emojis if e in NEGATIVE_EMOJIS]
    emoji_note = ""
    if pos_e: emoji_note += f" Positive emoji signals: {' '.join(pos_e)}."
    if neg_e: emoji_note += f" Negative emoji signals: {' '.join(neg_e)}."

    aspect_note = f" Key aspects: {', '.join(aspects[:3])}." if aspects else ""

    # Context-aware descriptions — don't say "satisfaction" for NEGATIVE
    descriptions = {
        "POSITIVE": "Subject expresses satisfaction, approval, or enthusiasm.",
        "NEGATIVE": "Subject expresses dissatisfaction, hostility, or criticism.",
        "NEUTRAL":  "No strong emotional polarity detected — informational or mixed tone.",
    }

    base = (
        f"{conf_str.capitalize()} confidence {sentiment.lower()} sentiment "
        f"(pos={pos:.1f}% / neg={neg:.1f}% / neu={neu:.1f}%)."
        f"{lang_note}{emoji_note}{aspect_note} {descriptions.get(sentiment, '')}"
    )

    if override_note:
        base += f" ⚠ {override_note}"

    return base

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "model_source": MODEL_PATH,
        "num_labels": len(LABELS),
        "labels": LABELS,
        "device": str(DEVICE)
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()[:512]
    if not text:
        return jsonify({"error": "Empty text"}), 400

    processed = preprocess_tweet(text)
    inputs = tokenizer(processed, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1)[0].tolist()

    sentiment, pos, neg, neu = build_scores(probs)

    # ── Lexical override for ambiguous negative cases ──
    sentiment, pos, neg, neu, override_note = apply_lexical_override(
        text, sentiment, pos, neg, neu
    )

    lang    = detect_language(text)
    emojis  = extract_emojis(text)
    aspects = extract_aspects(text)
    analysis = generate_analysis(text, sentiment, pos, neg, neu, lang, aspects, emojis, override_note)

    return jsonify({
        "sentiment": sentiment,
        "pos": pos, "neg": neg, "neu": neu,
        "lang": lang,
        "aspects": aspects,
        "emojis": emojis,
        "analysis": analysis,
        "raw_text": text,
        "debug": {
            "num_labels": len(LABELS),
            "labels": LABELS,
            "raw_probs": [round(p*100, 2) for p in probs],
            "override_applied": override_note is not None
        }
    })

@app.route("/batch", methods=["POST"])
def batch_analyze():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' array"}), 400

    results = []
    for text in data["texts"][:20]:
        text = text.strip()[:512]
        inputs = tokenizer(preprocess_tweet(text), return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1)[0].tolist()
        sentiment, pos, neg, neu = build_scores(probs)
        sentiment, pos, neg, neu, _ = apply_lexical_override(text, sentiment, pos, neg, neu)
        results.append({
            "text": text, "sentiment": sentiment,
            "pos": pos, "neg": neg, "neu": neu,
            "lang": detect_language(text),
            "aspects": extract_aspects(text),
            "emojis": extract_emojis(text),
        })
    return jsonify({"results": results, "count": len(results)})

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SENTRIX // Neural Sentiment Engine")
    print("  twitter_roberta_v2 — Local Deployment")
    print("="*55)

    print(f"\n  Model source: Hugging Face Hub ({MODEL_PATH})")
    print(f"  First run will download ~500MB and cache locally.")
    print(f"  Subsequent runs load from cache instantly.\n")
    load_model()

    print(f"\n  Device:   {DEVICE}")
    print(f"  API:      http://localhost:5000")
    print(f"  Health:   http://localhost:5000/health")
    print(f"  Frontend: open index.html in browser")
    print("="*55 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
