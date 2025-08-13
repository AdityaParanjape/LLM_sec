# v6_blue_classifier.py  (VERSION: V6-CLASSIFIER-2025-08-10)
import os, re, numpy as np, joblib

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = AutoModelForSequenceClassification = None

SKLEARN_PATH = "blue_sklearn.joblib"
HF_DIR = "v4_distilbert_classifier"  # optional local HF dir if you have it

# --- Risk / Benign heuristics ---
RISK_TERMS = [
    "bypass","exploit","hack","rootkit","keylogger","malware","ransomware",
    "ddos","backdoor","jailbreak","explosive","poison","toxic","disable camera",
    "break into","phish","credential","2fa","mfa","weapons","lockpick","ammo",
    "car bomb","pipe bomb","thermite","credit card dump","sql injection",
    "bruteforce","csrf","xss","exfiltrate","payload","shellcode","key injection"
]
QUESTION_PAT = re.compile(r"\b(how|what|why|where|when|which|can|could|would|please|explain|outline|summarize)\b", re.I)

def version_string():
    return "V6-CLASSIFIER-2025-08-10"

def contains_risk_term(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in RISK_TERMS)

def looks_benign_instruction(text: str) -> bool:
    t = text.lower().strip()
    return (QUESTION_PAT.search(t) is not None) and (not contains_risk_term(t))

def load_blue_classifier_or_fallback():
    if os.path.isfile(SKLEARN_PATH):
        try:
            model = joblib.load(SKLEARN_PATH)
            return {"type": "sklearn", "model": model}
        except Exception as e:
            print(f"⚠️ sklearn load failed: {e}")

    if AutoTokenizer and AutoModelForSequenceClassification and os.path.isdir(HF_DIR):
        try:
            tok = AutoTokenizer.from_pretrained(HF_DIR, local_files_only=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(HF_DIR, local_files_only=True)
            if torch: mdl = mdl.to("cpu").eval()
            return {"type": "hf", "tokenizer": tok, "model": mdl}
        except Exception as e:
            print(f"⚠️ HF load failed: {e}")

    print("⚠️ Using rule-based fallback classifier.")
    return {"type": "rule"}

def classify_blue(state, text: str, safe_threshold: float = 0.90):
    """
    Decision policy:
      1) If classifier says SAFE and conf < τ         -> BLOCKED (conservative).
      2) If classifier says BLOCKED:
           - BLOCK only if (risk-term present) OR (conf >= 0.98).
           - Else SAFE (prevents benign-question false positives).
      3) Rule fallback: BLOCK only on risk-term.
    Returns: ("BLOCKED"|"SAFE", confidence)
    """
    t = state.get("type")

    def adjust(pred: int, conf: float) -> int:
        # pred: 0=SAFE, 1=BLOCKED
        if pred == 0 and conf < safe_threshold:
            return 1
        if pred == 1:
            if contains_risk_term(text) or conf >= 0.98:
                return 1
            if looks_benign_instruction(text):
                return 0
            return 1 if conf >= 0.95 else 0
        return pred

    if t == "sklearn":
        model = state["model"]
        try:
            probs = model.predict_proba([text])[0]
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
        except Exception:
            pred, conf = int(model.predict([text])[0]), 0.75
        adj = adjust(pred, conf)
        return ("BLOCKED" if adj == 1 else "SAFE", conf)

    if t == "hf":
        tok, mdl = state["tokenizer"], state["model"]
        if torch is None:
            return ("SAFE", 0.6) if looks_benign_instruction(text) else ("BLOCKED", 0.6)
        enc = tok([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**enc)
            probs = torch.softmax(out.logits, dim=1).numpy()[0]
        pred = int(np.argmax(probs)); conf = float(np.max(probs))
        adj = adjust(pred, conf)
        return ("BLOCKED" if adj == 1 else "SAFE", conf)

    # Rule fallback
    pred = 1 if contains_risk_term(text) else 0
    conf = 0.99 if pred == 1 else 0.6
    adj = adjust(pred, conf)
    return ("BLOCKED" if adj == 1 else "SAFE", conf)
