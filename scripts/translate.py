#!/usr/bin/env python3
"""NLLB Translation service using CTranslate2."""

import ctranslate2
import os
import sys
from transformers import NllbTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Model path
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/nllb-200-3.3B-ct2-int8")

# Global instances
translator = None
tokenizer = None
device_used = None

def get_device():
    """Try to use GPU, fallback to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU detected via PyTorch: {device_name}")
            # Note: pip ctranslate2 only supports NVIDIA CUDA, not AMD ROCm
            # For AMD GPU support, CTranslate2 must be built from source with ROCm
            return "cuda"
    except Exception as e:
        print(f"PyTorch CUDA check failed: {e}")
    return "cpu"

def init_model():
    """Initialize the translation model."""
    global translator, tokenizer, device_used

    device = get_device()
    compute_type = "int8" if device == "cpu" else "int8_float16"

    print(f"Loading model from {MODEL_PATH}")
    print(f"Attempting device: {device}, Compute type: {compute_type}")

    try:
        translator = ctranslate2.Translator(
            MODEL_PATH,
            device=device,
            compute_type=compute_type,
        )
        device_used = device
    except Exception as e:
        print(f"Failed to load on {device}: {e}")
        if device != "cpu":
            print("Falling back to CPU...")
            translator = ctranslate2.Translator(
                MODEL_PATH,
                device="cpu",
                compute_type="int8",
            )
            device_used = "cpu"
        else:
            raise

    # Load NLLB tokenizer
    tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH)
    print(f"Model loaded successfully on {device_used}!")
    return device_used

def translate(text, source_lang="eng_Latn", target_lang="ukr_Cyrl"):
    """Translate text using NLLB model."""
    global translator, tokenizer

    # Set source language
    tokenizer.src_lang = source_lang

    # Tokenize
    encoded = tokenizer(text, return_tensors=None, truncation=True, max_length=512)
    source_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    # Translate
    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[target_lang]],
        beam_size=4,
        max_decoding_length=256,
    )

    # Get output tokens
    output_tokens = results[0].hypotheses[0]

    # Remove target language token if present
    if output_tokens and output_tokens[0] == target_lang:
        output_tokens = output_tokens[1:]

    # Decode
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text

@app.route("/translate", methods=["POST"])
def translate_endpoint():
    """REST API endpoint for translation."""
    data = request.get_json()

    text = data.get("text", "")
    source_lang = data.get("source_lang", "eng_Latn")
    target_lang = data.get("target_lang", "ukr_Cyrl")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = translate(text, source_lang, target_lang)
        return jsonify({
            "translation": result,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "device": device_used
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "device": device_used})

@app.route("/languages", methods=["GET"])
def languages():
    """Return supported language codes."""
    common_languages = {
        "eng_Latn": "English",
        "ukr_Cyrl": "Ukrainian",
        "rus_Cyrl": "Russian",
        "deu_Latn": "German",
        "fra_Latn": "French",
        "spa_Latn": "Spanish",
        "ita_Latn": "Italian",
        "pol_Latn": "Polish",
        "por_Latn": "Portuguese",
        "nld_Latn": "Dutch",
        "zho_Hans": "Chinese (Simplified)",
        "jpn_Jpan": "Japanese",
        "kor_Hang": "Korean",
        "ara_Arab": "Arabic",
        "tur_Latn": "Turkish",
    }
    return jsonify(common_languages)

def main():
    """Main entry point."""
    device = init_model()

    if len(sys.argv) > 1:
        # CLI mode
        text = " ".join(sys.argv[1:])
        source_lang = os.environ.get("SOURCE_LANG", "eng_Latn")
        target_lang = os.environ.get("TARGET_LANG", "ukr_Cyrl")

        print(f"\nTranslating: {text}")
        print(f"From: {source_lang} -> To: {target_lang}")

        result = translate(text, source_lang, target_lang)
        print(f"\nResult: {result}")
        print(f"Device: {device}")
    else:
        # Server mode
        print("Starting translation server on port 5000...")
        app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
