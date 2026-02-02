#!/usr/bin/env python3
"""NLLB Translation Server with AMD GPU support"""

import os
import ctranslate2
from transformers import AutoTokenizer
from flask import Flask, request, jsonify

# Essential for RX 6600 (gfx1032 -> gfx1030 compatibility)
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

app = Flask(__name__)

# Global translator and tokenizer
translator = None
tokenizer = None

# NLLB language codes (subset of supported languages)
LANG_CODES = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
}

def load_model():
    global translator, tokenizer
    print("Loading NLLB model on GPU...")

    model_path = "/models/nllb-3.3B-ct2"

    # Load translator with GPU
    translator = ctranslate2.Translator(
        model_path,
        device="cuda",
        compute_type="int8_float16"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-3.3B",
        src_lang="eng_Latn"
    )

    print("Model loaded successfully!")

def translate_text(text, source_lang="en", target_lang="fr"):
    """Translate text from source to target language"""
    src_code = LANG_CODES.get(source_lang, source_lang)
    tgt_code = LANG_CODES.get(target_lang, target_lang)

    # Set source language
    tokenizer.src_lang = src_code

    # Tokenize
    tokens = tokenizer(text, return_tensors=None)["input_ids"]
    token_strs = tokenizer.convert_ids_to_tokens(tokens)

    # Translate
    results = translator.translate_batch(
        [token_strs],
        target_prefix=[[tgt_code]]
    )

    # Decode
    output_tokens = results[0].hypotheses[0][1:]  # Skip language code
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    translated = tokenizer.decode(output_ids, skip_special_tokens=True)

    return translated

@app.route("/translate", methods=["POST"])
def translate_endpoint():
    data = request.json
    text = data.get("text", "")
    source = data.get("source", "en")
    target = data.get("target", "fr")

    try:
        result = translate_text(text, source, target)
        return jsonify({"translation": result, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/languages", methods=["GET"])
def languages():
    return jsonify(LANG_CODES)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "gpu": "AMD RX 6600"})

if __name__ == "__main__":
    load_model()
    print("Starting translation server on port 5000...")
    app.run(host="0.0.0.0", port=5000)
