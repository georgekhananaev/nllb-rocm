#!/usr/bin/env python3
"""Simple test script to verify GPU translation is working"""

import os
import ctranslate2
from transformers import AutoTokenizer

# Essential for RX 6600
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

print("Loading model...")
translator = ctranslate2.Translator("/models/nllb-3.3B-ct2", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="eng_Latn")

print("GPU translation test:")
print("-" * 40)

# Test translations
tests = [
    ("Hello, how are you today?", "fra_Latn", "French"),
    ("The weather is beautiful.", "deu_Latn", "German"),
    ("I love programming.", "spa_Latn", "Spanish"),
]

for text, target_lang, lang_name in tests:
    tokens = tokenizer(text, return_tensors=None)["input_ids"]
    token_strs = tokenizer.convert_ids_to_tokens(tokens)

    results = translator.translate_batch([token_strs], target_prefix=[[target_lang]])

    output_tokens = results[0].hypotheses[0][1:]
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    translated = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"English: {text}")
    print(f"{lang_name}: {translated}")
    print()

print("GPU translation working!")
