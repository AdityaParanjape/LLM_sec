# v6_ollama_helpers.py
import requests
import json
import re

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def ollama_generate(model_name, prompt, max_tokens=256, temperature=0.7):
    """
    Sends a prompt to the local Ollama server and returns the generated text.
    Requires Ollama to be running locally: ollama serve
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()

        full_output = ""
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                full_output += data["response"]
        return full_output.strip()

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return ""


def redact(text):
    """
    Simple redaction of sensitive terms like emails, phone numbers, or API keys.
    """
    # Example regex patterns for common sensitive info
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"(?:api[_-]?key|secret)[=:]\s*[A-Za-z0-9\-_.]+", "[REDACTED_KEY]", text, flags=re.IGNORECASE)
    return text
