#!/usr/bin/env python3
"""NLLB Translation service using CTranslate2 with FastAPI - Optimized for throughput."""

import ctranslate2
import os
import sys
import asyncio
import threading
import secrets
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from transformers import NllbTokenizer
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/nllb-200-3.3B-ct2-int8")
DB_PATH = os.environ.get("DB_PATH", "/data/tokens.db")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "admin-secret-change-me")

# Batching configuration
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", "50"))  # Max wait time for batch
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))  # Max requests per batch
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "1"))  # 1 for speed, 4 for quality
INTER_THREADS = int(os.environ.get("INTER_THREADS", "2"))  # HIP/CUDA streams

# FastAPI app
app = FastAPI(
    title="NLLB Translation API",
    description="Multilingual translation service using Meta's NLLB-200 model on AMD GPU (Optimized)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

security = HTTPBearer()

# Global instances
translator = None
device_used = None

# Thread-local storage for tokenizers
_thread_local = threading.local()

# Thread pool for CPU-bound tokenization
tokenizer_pool = ThreadPoolExecutor(max_workers=4)

# Batching infrastructure
batch_queue: asyncio.Queue = None
batch_processor_task = None


# ============== Pydantic Models ==============

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "eng_Latn"
    target_lang: str = "ukr_Cyrl"

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you today?",
                "source_lang": "eng_Latn",
                "target_lang": "ukr_Cyrl"
            }
        }


class TranslateResponse(BaseModel):
    translation: str
    source_lang: str
    target_lang: str
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    batch_queue_size: int
    config: dict


class TokenCreate(BaseModel):
    name: str
    description: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "my-app",
                "description": "Token for my application"
            }
        }


class TokenResponse(BaseModel):
    id: int
    name: str
    token: str
    description: Optional[str]
    created_at: str
    is_active: bool


class TokenInfo(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: str
    is_active: bool
    usage_count: int


class ErrorResponse(BaseModel):
    error: str


# ============== Batch Request Dataclass ==============

@dataclass
class BatchRequest:
    """A single translation request waiting in the batch queue."""
    text: str
    source_lang: str
    target_lang: str
    future: asyncio.Future
    source_tokens: List[str] = None  # Filled after tokenization


# ============== Database ==============

def init_db():
    """Initialize SQLite database for token management."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            token TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            usage_count INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def verify_token(token: str) -> Optional[dict]:
    """Verify API token and return token info."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM tokens WHERE token = ? AND is_active = 1",
            (token,)
        )
        row = cursor.fetchone()

        if row:
            # Update usage count
            cursor.execute(
                "UPDATE tokens SET usage_count = usage_count + 1 WHERE id = ?",
                (row["id"],)
            )
            conn.commit()
            return dict(row)

    return None


# ============== Authentication ==============

async def get_api_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Validate Bearer token."""
    token = credentials.credentials

    token_info = verify_token(token)
    if not token_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_info


async def get_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Validate admin token."""
    if credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# ============== Model Functions ==============

def get_device():
    """Try to use GPU, fallback to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU detected via PyTorch: {device_name}")
            return "cuda"
    except Exception as e:
        print(f"PyTorch CUDA check failed: {e}")
    return "cpu"


def get_tokenizer():
    """Get or create a thread-local tokenizer instance."""
    if not hasattr(_thread_local, 'tokenizer'):
        _thread_local.tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH)
    return _thread_local.tokenizer


def init_model():
    """Initialize the translation model with optimized settings."""
    global translator, device_used

    device = get_device()
    compute_type = "int8" if device == "cpu" else "int8_float16"

    print(f"Loading model from {MODEL_PATH}")
    print(f"Attempting device: {device}, Compute type: {compute_type}")
    print(f"Optimization settings: beam_size={BEAM_SIZE}, inter_threads={INTER_THREADS}, batch_timeout={BATCH_TIMEOUT_MS}ms")

    try:
        translator = ctranslate2.Translator(
            MODEL_PATH,
            device=device,
            compute_type=compute_type,
            inter_threads=INTER_THREADS,  # Multiple HIP/CUDA streams
            max_queued_batches=-1,  # Unlimited queue for async
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
                inter_threads=4,  # CPU threads
            )
            device_used = "cpu"
        else:
            raise

    # Pre-warm one tokenizer (loads vocab files into memory)
    _ = get_tokenizer()
    print(f"Model loaded successfully on {device_used}!")
    return device_used


def tokenize_text(text: str, source_lang: str) -> List[str]:
    """Tokenize text using a thread-local tokenizer (CPU-bound, can run in parallel)."""
    tokenizer = get_tokenizer()
    tokenizer.src_lang = source_lang

    encoded = tokenizer(text, return_tensors=None, truncation=True, max_length=512)
    source_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    return source_tokens


def decode_tokens(output_tokens: List[str], target_lang: str) -> str:
    """Decode output tokens to text (CPU-bound)."""
    tokenizer = get_tokenizer()

    # Remove target language prefix if present
    if output_tokens and output_tokens[0] == target_lang:
        output_tokens = output_tokens[1:]

    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_text


def translate_batch_sync(requests: List[BatchRequest]) -> List[str]:
    """Translate a batch of pre-tokenized requests synchronously (GPU-bound)."""
    if not requests:
        return []

    # Prepare batch inputs
    source_batch = [req.source_tokens for req in requests]
    target_prefixes = [[req.target_lang] for req in requests]

    # Run batch translation with optimized settings
    results = translator.translate_batch(
        source_batch,
        target_prefix=target_prefixes,
        beam_size=BEAM_SIZE,  # 1 for speed, higher for quality
        max_decoding_length=256,
        return_scores=False,  # Skip softmax computation
        max_batch_size=0,  # Process all at once
    )

    # Decode results
    translations = []
    for i, result in enumerate(results):
        output_tokens = result.hypotheses[0]
        translation = decode_tokens(output_tokens, requests[i].target_lang)
        translations.append(translation)

    return translations


async def process_batch(requests: List[BatchRequest]):
    """Process a batch of requests asynchronously."""
    if not requests:
        return

    loop = asyncio.get_event_loop()

    try:
        # Step 1: Tokenize all requests in parallel (CPU-bound)
        tokenize_futures = []
        for req in requests:
            future = loop.run_in_executor(
                tokenizer_pool,
                tokenize_text,
                req.text,
                req.source_lang
            )
            tokenize_futures.append(future)

        # Wait for all tokenizations to complete
        tokenized_results = await asyncio.gather(*tokenize_futures)

        # Attach tokens to requests
        for req, tokens in zip(requests, tokenized_results):
            req.source_tokens = tokens

        # Step 2: Run GPU translation (GPU-bound, runs in thread to not block event loop)
        translations = await loop.run_in_executor(
            None,  # Default executor
            translate_batch_sync,
            requests
        )

        # Step 3: Resolve futures with results
        for req, translation in zip(requests, translations):
            if not req.future.done():
                req.future.set_result(translation)

    except Exception as e:
        # On error, fail all pending requests
        for req in requests:
            if not req.future.done():
                req.future.set_exception(e)


async def batch_processor():
    """Background task that collects requests and processes them in batches."""
    global batch_queue

    while True:
        batch: List[BatchRequest] = []

        try:
            # Wait for first request (blocking)
            first_request = await batch_queue.get()
            batch.append(first_request)

            # Collect more requests within timeout window
            deadline = asyncio.get_event_loop().time() + (BATCH_TIMEOUT_MS / 1000.0)

            while len(batch) < MAX_BATCH_SIZE:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break

                try:
                    request = await asyncio.wait_for(
                        batch_queue.get(),
                        timeout=remaining
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    break

            # Process the collected batch
            await process_batch(batch)

        except asyncio.CancelledError:
            # Shutdown - process remaining requests
            while not batch_queue.empty():
                try:
                    batch.append(batch_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if batch:
                await process_batch(batch)
            raise

        except Exception as e:
            print(f"Batch processor error: {e}")
            # Fail any pending requests in batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


async def translate_text_async(text: str, source_lang: str, target_lang: str) -> str:
    """Queue a translation request and wait for result."""
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    request = BatchRequest(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        future=future
    )

    await batch_queue.put(request)
    return await future


def translate_text(text: str, source_lang: str = "eng_Latn", target_lang: str = "ukr_Cyrl") -> str:
    """Translate text using NLLB model (sync version for CLI mode)."""
    source_tokens = tokenize_text(text, source_lang)

    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[target_lang]],
        beam_size=BEAM_SIZE,
        max_decoding_length=256,
        return_scores=False,
    )

    output_tokens = results[0].hypotheses[0]
    return decode_tokens(output_tokens, target_lang)


# ============== API Endpoints ==============

@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate_endpoint(
    request: TranslateRequest,
    token_info: dict = Depends(get_api_token)
):
    """
    Translate text between languages.

    Requires a valid API token in the Authorization header.
    Requests are automatically batched for optimal GPU utilization.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        result = await translate_text_async(request.text, request.source_lang, request.target_lang)
        return TranslateResponse(
            translation=result,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            device=device_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint (no authentication required)."""
    queue_size = batch_queue.qsize() if batch_queue else 0
    return HealthResponse(
        status="ok",
        device=device_used or "not initialized",
        batch_queue_size=queue_size,
        config={
            "beam_size": BEAM_SIZE,
            "batch_timeout_ms": BATCH_TIMEOUT_MS,
            "max_batch_size": MAX_BATCH_SIZE,
            "inter_threads": INTER_THREADS,
        }
    )


@app.get("/languages", tags=["Translation"])
async def languages():
    """
    Get list of all 202 supported language codes.

    NLLB-200 supports 202 languages. Language codes follow the format: xxx_Yyyy
    where xxx is the ISO 639-3 language code and Yyyy is the script.
    """
    return {
        # A
        "ace_Arab": "Acehnese (Arabic script)",
        "ace_Latn": "Acehnese (Latin script)",
        "acm_Arab": "Mesopotamian Arabic",
        "acq_Arab": "Ta'izzi-Adeni Arabic",
        "aeb_Arab": "Tunisian Arabic",
        "afr_Latn": "Afrikaans",
        "ajp_Arab": "South Levantine Arabic",
        "aka_Latn": "Akan",
        "als_Latn": "Albanian (Tosk)",
        "amh_Ethi": "Amharic",
        "apc_Arab": "North Levantine Arabic",
        "arb_Arab": "Modern Standard Arabic",
        "ars_Arab": "Najdi Arabic",
        "ary_Arab": "Moroccan Arabic",
        "arz_Arab": "Egyptian Arabic",
        "asm_Beng": "Assamese",
        "ast_Latn": "Asturian",
        "awa_Deva": "Awadhi",
        "ayr_Latn": "Central Aymara",
        "azb_Arab": "South Azerbaijani",
        "azj_Latn": "North Azerbaijani",
        # B
        "bak_Cyrl": "Bashkir",
        "bam_Latn": "Bambara",
        "ban_Latn": "Balinese",
        "bel_Cyrl": "Belarusian",
        "bem_Latn": "Bemba",
        "ben_Beng": "Bengali",
        "bho_Deva": "Bhojpuri",
        "bjn_Arab": "Banjar (Arabic script)",
        "bjn_Latn": "Banjar (Latin script)",
        "bod_Tibt": "Tibetan",
        "bos_Latn": "Bosnian",
        "bug_Latn": "Buginese",
        "bul_Cyrl": "Bulgarian",
        # C
        "cat_Latn": "Catalan",
        "ceb_Latn": "Cebuano",
        "ces_Latn": "Czech",
        "cjk_Latn": "Chokwe",
        "ckb_Arab": "Central Kurdish",
        "crh_Latn": "Crimean Tatar",
        "cym_Latn": "Welsh",
        # D
        "dan_Latn": "Danish",
        "deu_Latn": "German",
        "dik_Latn": "Southwestern Dinka",
        "dyu_Latn": "Dyula",
        "dzo_Tibt": "Dzongkha",
        # E
        "ell_Grek": "Greek",
        "eng_Latn": "English",
        "epo_Latn": "Esperanto",
        "est_Latn": "Estonian",
        "eus_Latn": "Basque",
        "ewe_Latn": "Ewe",
        # F
        "fao_Latn": "Faroese",
        "fij_Latn": "Fijian",
        "fin_Latn": "Finnish",
        "fon_Latn": "Fon",
        "fra_Latn": "French",
        "fur_Latn": "Friulian",
        "fuv_Latn": "Nigerian Fulfulde",
        # G
        "gaz_Latn": "West Central Oromo",
        "gla_Latn": "Scottish Gaelic",
        "gle_Latn": "Irish",
        "glg_Latn": "Galician",
        "grn_Latn": "Guarani",
        "guj_Gujr": "Gujarati",
        # H
        "hat_Latn": "Haitian Creole",
        "hau_Latn": "Hausa",
        "heb_Hebr": "Hebrew",
        "hin_Deva": "Hindi",
        "hne_Deva": "Chhattisgarhi",
        "hrv_Latn": "Croatian",
        "hun_Latn": "Hungarian",
        "hye_Armn": "Armenian",
        # I
        "ibo_Latn": "Igbo",
        "ilo_Latn": "Ilocano",
        "ind_Latn": "Indonesian",
        "isl_Latn": "Icelandic",
        "ita_Latn": "Italian",
        # J
        "jav_Latn": "Javanese",
        "jpn_Jpan": "Japanese",
        # K
        "kab_Latn": "Kabyle",
        "kac_Latn": "Jingpho",
        "kam_Latn": "Kamba",
        "kan_Knda": "Kannada",
        "kas_Arab": "Kashmiri (Arabic script)",
        "kas_Deva": "Kashmiri (Devanagari script)",
        "kat_Geor": "Georgian",
        "kaz_Cyrl": "Kazakh",
        "kbp_Latn": "Kabiyè",
        "kea_Latn": "Kabuverdianu",
        "khk_Cyrl": "Halh Mongolian",
        "khm_Khmr": "Khmer",
        "kik_Latn": "Kikuyu",
        "kin_Latn": "Kinyarwanda",
        "kir_Cyrl": "Kyrgyz",
        "kmb_Latn": "Kimbundu",
        "kmr_Latn": "Northern Kurdish",
        "knc_Arab": "Central Kanuri (Arabic script)",
        "knc_Latn": "Central Kanuri (Latin script)",
        "kon_Latn": "Kikongo",
        "kor_Hang": "Korean",
        # L
        "lao_Laoo": "Lao",
        "lij_Latn": "Ligurian",
        "lim_Latn": "Limburgish",
        "lin_Latn": "Lingala",
        "lit_Latn": "Lithuanian",
        "lmo_Latn": "Lombard",
        "ltg_Latn": "Latgalian",
        "ltz_Latn": "Luxembourgish",
        "lua_Latn": "Luba-Kasai",
        "lug_Latn": "Ganda",
        "luo_Latn": "Luo",
        "lus_Latn": "Mizo",
        "lvs_Latn": "Standard Latvian",
        # M
        "mag_Deva": "Magahi",
        "mai_Deva": "Maithili",
        "mal_Mlym": "Malayalam",
        "mar_Deva": "Marathi",
        "min_Latn": "Minangkabau",
        "mkd_Cyrl": "Macedonian",
        "mlt_Latn": "Maltese",
        "mni_Beng": "Meitei",
        "mos_Latn": "Mossi",
        "mri_Latn": "Maori",
        "mya_Mymr": "Burmese",
        # N
        "nld_Latn": "Dutch",
        "nno_Latn": "Norwegian Nynorsk",
        "nob_Latn": "Norwegian Bokmål",
        "npi_Deva": "Nepali",
        "nso_Latn": "Northern Sotho",
        "nus_Latn": "Nuer",
        "nya_Latn": "Chichewa",
        # O
        "oci_Latn": "Occitan",
        "ory_Orya": "Odia",
        # P
        "pag_Latn": "Pangasinan",
        "pan_Guru": "Punjabi",
        "pap_Latn": "Papiamento",
        "pbt_Arab": "Southern Pashto",
        "pes_Arab": "Western Persian",
        "plt_Latn": "Plateau Malagasy",
        "pol_Latn": "Polish",
        "por_Latn": "Portuguese",
        "prs_Arab": "Dari",
        # Q
        "quy_Latn": "Ayacucho Quechua",
        # R
        "ron_Latn": "Romanian",
        "run_Latn": "Rundi",
        "rus_Cyrl": "Russian",
        # S
        "sag_Latn": "Sango",
        "san_Deva": "Sanskrit",
        "sat_Beng": "Santali",
        "scn_Latn": "Sicilian",
        "shn_Mymr": "Shan",
        "sin_Sinh": "Sinhala",
        "slk_Latn": "Slovak",
        "slv_Latn": "Slovenian",
        "smo_Latn": "Samoan",
        "sna_Latn": "Shona",
        "snd_Arab": "Sindhi",
        "som_Latn": "Somali",
        "sot_Latn": "Southern Sotho",
        "spa_Latn": "Spanish",
        "srd_Latn": "Sardinian",
        "srp_Cyrl": "Serbian",
        "ssw_Latn": "Swati",
        "sun_Latn": "Sundanese",
        "swe_Latn": "Swedish",
        "swh_Latn": "Swahili",
        "szl_Latn": "Silesian",
        # T
        "tam_Taml": "Tamil",
        "taq_Latn": "Tamasheq (Latin script)",
        "taq_Tfng": "Tamasheq (Tifinagh script)",
        "tat_Cyrl": "Tatar",
        "tel_Telu": "Telugu",
        "tgk_Cyrl": "Tajik",
        "tgl_Latn": "Tagalog",
        "tha_Thai": "Thai",
        "tir_Ethi": "Tigrinya",
        "tpi_Latn": "Tok Pisin",
        "tsn_Latn": "Tswana",
        "tso_Latn": "Tsonga",
        "tuk_Latn": "Turkmen",
        "tum_Latn": "Tumbuka",
        "tur_Latn": "Turkish",
        "twi_Latn": "Twi",
        "tzm_Tfng": "Central Atlas Tamazight",
        # U
        "uig_Arab": "Uyghur",
        "ukr_Cyrl": "Ukrainian",
        "umb_Latn": "Umbundu",
        "urd_Arab": "Urdu",
        "uzn_Latn": "Northern Uzbek",
        # V
        "vec_Latn": "Venetian",
        "vie_Latn": "Vietnamese",
        # W
        "war_Latn": "Waray",
        "wol_Latn": "Wolof",
        # X
        "xho_Latn": "Xhosa",
        # Y
        "ydd_Hebr": "Eastern Yiddish",
        "yor_Latn": "Yoruba",
        "yue_Hant": "Cantonese",
        # Z
        "zho_Hans": "Chinese (Simplified)",
        "zho_Hant": "Chinese (Traditional)",
        "zsm_Latn": "Standard Malay",
        "zul_Latn": "Zulu",
    }


# ============== Admin Endpoints ==============

@app.post("/admin/tokens", response_model=TokenResponse, tags=["Admin"])
async def create_token(
    request: TokenCreate,
    _: bool = Depends(get_admin_token)
):
    """
    Create a new API token (admin only).

    Requires admin token in Authorization header.
    """
    token = secrets.token_urlsafe(32)

    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO tokens (name, token, description) VALUES (?, ?, ?)",
                (request.name, token, request.description)
            )
            conn.commit()
            token_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Token name already exists")

        cursor.execute("SELECT * FROM tokens WHERE id = ?", (token_id,))
        row = cursor.fetchone()

    return TokenResponse(
        id=row["id"],
        name=row["name"],
        token=row["token"],
        description=row["description"],
        created_at=row["created_at"],
        is_active=bool(row["is_active"])
    )


@app.get("/admin/tokens", response_model=List[TokenInfo], tags=["Admin"])
async def list_tokens(_: bool = Depends(get_admin_token)):
    """
    List all API tokens (admin only).

    Token values are not returned for security.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description, created_at, is_active, usage_count FROM tokens")
        rows = cursor.fetchall()

    return [
        TokenInfo(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            is_active=bool(row["is_active"]),
            usage_count=row["usage_count"]
        )
        for row in rows
    ]


@app.delete("/admin/tokens/{token_id}", tags=["Admin"])
async def delete_token(token_id: int, _: bool = Depends(get_admin_token)):
    """
    Deactivate an API token (admin only).
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tokens SET is_active = 0 WHERE id = ?",
            (token_id,)
        )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Token not found")

    return {"message": "Token deactivated"}


@app.post("/admin/tokens/{token_id}/activate", tags=["Admin"])
async def activate_token(token_id: int, _: bool = Depends(get_admin_token)):
    """
    Reactivate a deactivated token (admin only).
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tokens SET is_active = 1 WHERE id = ?",
            (token_id,)
        )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Token not found")

    return {"message": "Token activated"}


# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Initialize model, database, and batch processor on startup."""
    global batch_queue, batch_processor_task

    init_db()
    init_model()

    # Initialize batch processing
    batch_queue = asyncio.Queue()
    batch_processor_task = asyncio.create_task(batch_processor())
    print("Batch processor started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global batch_processor_task

    if batch_processor_task:
        batch_processor_task.cancel()
        try:
            await batch_processor_task
        except asyncio.CancelledError:
            pass
    print("Batch processor stopped")


# ============== Main ==============

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # CLI mode - initialize manually
        init_db()
        init_model()

        text = " ".join(sys.argv[1:])
        source_lang = os.environ.get("SOURCE_LANG", "eng_Latn")
        target_lang = os.environ.get("TARGET_LANG", "ukr_Cyrl")

        print(f"\nTranslating: {text}")
        print(f"From: {source_lang} -> To: {target_lang}")

        result = translate_text(text, source_lang, target_lang)
        print(f"\nResult: {result}")
        print(f"Device: {device_used}")
    else:
        # Server mode
        print("Starting NLLB Translation API server (Optimized)...")
        print(f"Swagger UI: http://0.0.0.0:5000/docs")
        print(f"ReDoc: http://0.0.0.0:5000/redoc")
        uvicorn.run(app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
