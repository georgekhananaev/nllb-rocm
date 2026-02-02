#!/usr/bin/env python3
"""NLLB Translation service using CTranslate2 with FastAPI."""

import ctranslate2
import os
import sys
import threading
import secrets
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List

from transformers import NllbTokenizer
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/nllb-200-3.3B-ct2-int8")
DB_PATH = os.environ.get("DB_PATH", "/data/tokens.db")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "admin-secret-change-me")

# FastAPI app
app = FastAPI(
    title="NLLB Translation API",
    description="Multilingual translation service using Meta's NLLB-200 model on AMD GPU",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

security = HTTPBearer()

# Global instances
translator = None
tokenizer = None
device_used = None
translate_lock = threading.Lock()


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

    tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH)
    print(f"Model loaded successfully on {device_used}!")
    return device_used


def translate_text(text: str, source_lang: str = "eng_Latn", target_lang: str = "ukr_Cyrl") -> str:
    """Translate text using NLLB model (thread-safe)."""
    global translator, tokenizer

    with translate_lock:
        tokenizer.src_lang = source_lang

        encoded = tokenizer(text, return_tensors=None, truncation=True, max_length=512)
        source_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

        results = translator.translate_batch(
            [source_tokens],
            target_prefix=[[target_lang]],
            beam_size=4,
            max_decoding_length=256,
        )

        output_tokens = results[0].hypotheses[0]

        if output_tokens and output_tokens[0] == target_lang:
            output_tokens = output_tokens[1:]

        output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        return output_text


# ============== API Endpoints ==============

@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate_endpoint(
    request: TranslateRequest,
    token_info: dict = Depends(get_api_token)
):
    """
    Translate text between languages.

    Requires a valid API token in the Authorization header.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        result = translate_text(request.text, request.source_lang, request.target_lang)
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
    return HealthResponse(status="ok", device=device_used or "not initialized")


@app.get("/languages", tags=["Translation"])
async def languages():
    """
    Get list of commonly used language codes.

    NLLB supports 200+ languages. This returns a subset of common ones.
    """
    return {
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
        "hin_Deva": "Hindi",
        "vie_Latn": "Vietnamese",
        "tha_Thai": "Thai",
        "heb_Hebr": "Hebrew",
        "ces_Latn": "Czech",
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


# ============== Startup ==============

@app.on_event("startup")
async def startup_event():
    """Initialize model and database on startup."""
    init_db()
    init_model()


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
        print("Starting NLLB Translation API server...")
        print(f"Swagger UI: http://0.0.0.0:5000/docs")
        print(f"ReDoc: http://0.0.0.0:5000/redoc")
        uvicorn.run(app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
